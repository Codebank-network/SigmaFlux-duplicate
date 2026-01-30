import requests
import json
import os
import base64
import mimetypes
import cv2
import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# We use 1.5 Flash as it is currently the most stable for JSON Mode
GEMINI_MODEL = os.getenv("GOOGLE_GEMINI_MODEL", "gemini-2.5-flash")

def encode_image_to_base64(uploaded_image, file_name):
    """
    Encodes a CV2 image (numpy array) to a clean base64 string 
    and determines the mime type.
    """
    mime_type, _ = mimetypes.guess_type(file_name)
    if not mime_type:
        mime_type = "image/jpeg" 
        
    ext = mimetypes.guess_extension(mime_type) or ".jpg"
    
    # Encode image into memory buffer
    success, buffer = cv2.imencode(ext, uploaded_image)
    if not success:
        raise ValueError(f'Failed to encode image as {ext}')
    
    # Get raw base64 string
    b64_string = base64.b64encode(buffer).decode("utf-8")
    
    return mime_type, b64_string

def ocr_reading(uploaded_image, n, file_name, api_key=None):
    """
    Calls Google Gemini with JSON enforcement.
    """
    
    # 1. Prepare Credentials
    keys = []
    if api_key and str(api_key).strip():
        keys.append(str(api_key).strip())
        
    # Try st.secrets first, fall back to os.getenv
    try:
        env_keys = st.secrets["GOOGLE_API_KEYS"]
    except Exception:
        env_keys = os.getenv('GOOGLE_API_KEYS')
        
    if env_keys:
        keys.extend([k.strip() for k in env_keys.split(',') if k.strip()])
    
    keys = list(dict.fromkeys(keys)) # Remove duplicates
    
    if not keys:
        raise ValueError("No API keys found. Please provide api_key.")

    # 2. Prepare Image
    try:
        mime_type, b64_data = encode_image_to_base64(uploaded_image, file_name)
    except Exception as e:
        raise ValueError(f"Image processing failed: {e}")

    # 3. JSON-Focused Prompt
    prompt_text = (
        f"Analyze this attendance sheet image. "
        f"It contains student details and attendance marks for exactly {n} dates/columns. "
        f"Return a JSON object with a key 'records' containing a list of students. "
        f"Each item in the list must have these fields: "
        f"'s_no' (string), 'roll_no' (string), 'name' (string), "
        f"and 'attendance' (a list of exactly {n} strings, either 'Present' or 'Absent'). "
        f"Rules: "
        f"1. Mark 'Absent' if the cell is empty, has a cross (X), or a cut. "
        f"2. Mark 'Present' if there is a signature, text, or scribble. "
        f"3. Ensure the 'attendance' list has exactly {n} entries."
    )

    # 4. Construct Payload with JSON Enforcement
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt_text},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": b64_data
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json"  # <--- CRITICAL: Forces JSON output
        }
    }

    # 5. Iterate Keys (Failsafe)
    errors = []
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

    for index, current_key in enumerate(keys):
        try:
            response = requests.post(
                f"{url}?key={current_key}",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                try:
                    candidates = result.get('candidates', [])
                    if candidates and candidates[0].get('content'):
                        parts = candidates[0]['content'].get('parts', [])
                        if parts:
                            return parts[0].get('text', '')
                    
                    finish_reason = candidates[0].get('finishReason') if candidates else "UNKNOWN"
                    raise ValueError(f"Blocked/Empty. Reason: {finish_reason}")
                    
                except Exception as parse_err:
                    errors.append(f"Key #{index+1} Parse Error: {parse_err}")
                    continue

            elif response.status_code in [429, 500, 503]:
                errors.append(f"Key #{index+1} Server Error: {response.status_code}")
                time.sleep(1) 
                continue
            else:
                errors.append(f"Key #{index+1} Client Error: {response.status_code} {response.text}")
                continue

        except Exception as e:
            errors.append(f"Key #{index+1} Network Error: {e}")
            continue

    raise RuntimeError(f"All keys failed.\nLog:\n" + "\n".join(errors))