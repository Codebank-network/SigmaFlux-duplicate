import pandas as pd
import json
import re

def excel_creation(input_text, n):
    """
    Parses JSON output from Gemini and converts it to a DataFrame.
    """
    try:
        # Clean up potential markdown code blocks if the model included them
        clean_text = input_text.replace("```json", "").replace("```", "").strip()
        
        # Parse JSON
        data = json.loads(clean_text)
        
        # Extract the list of records
        # The prompt asks for a root object with key 'records'
        records = data.get("records", [])
        
        # If the model returned a direct list instead of a dict, handle that
        if isinstance(data, list):
            records = data
            
        if not records:
            return None

        # Build rows for DataFrame
        processed_rows = []
        
        for item in records:
            # Extract basic fields
            s_no = item.get("s_no", "")
            roll = item.get("roll_no", "")
            name = item.get("name", "")
            
            # Extract attendance list
            attendance_list = item.get("attendance", [])
            
            # Pad or truncate attendance list to match 'n' exactly
            # This handles cases where model hallucinated extra/fewer days
            if len(attendance_list) < n:
                attendance_list += ["Unknown"] * (n - len(attendance_list))
            else:
                attendance_list = attendance_list[:n]
                
            row = [s_no, roll, name] + attendance_list
            processed_rows.append(row)

        # Define Columns
        date_columns = [f'Date {i+1}' for i in range(n)]
        columns = ['S.No', 'Roll Number', 'Name'] + date_columns
        
        df = pd.DataFrame(processed_rows, columns=columns)
        return df

    except json.JSONDecodeError as e:
        print(f"JSON Parsing failed: {e}")
        # If JSON fails, the output might be completely malformed
        return None
    except Exception as e:
        print(f"Post-processing error: {e}")
        return None