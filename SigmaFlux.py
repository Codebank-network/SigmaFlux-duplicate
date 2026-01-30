# main code

import pre_processing_pipeline as pre_p         # importing preprocessing part
import llm_code as ocr                          # importing the ocr part
import post_processing as post_p                # importing postprocessing part

def main(input_image,file_name,n,api_key):

    
    output_image = pre_p.run_parallel_image_processing(input_image)
    output_text = ocr.ocr_reading(output_image,n,file_name,api_key)

    df = post_p.excel_creation(output_text,n)

    if df is None:
        snippet = output_text if output_text else "<empty output from OCR/LLM>"
        # Limit snippet length to avoid huge errors
        snippet = snippet[:4000]
        raise ValueError(f"No table could be parsed from LLM output. LLM output (truncated):\n{snippet}")

    return df
