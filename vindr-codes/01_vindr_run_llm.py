import os
import json
import re
import logging
import click
from _constant import *  # Assumes that list_png_files, fix_json, and allowable_models are provided

# --- Setup logging to both console and file ---
log_file = 'process_reports.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'),
        logging.StreamHandler()
    ]
)

# Override print to also log to file
_builtin_print = print

def print(*args, **kwargs):
    message = ' '.join(str(a) for a in args)
    logging.info(message)
    _builtin_print(*args, **kwargs)

# Directories for Left-side
source_file_dir_C_L = '/mnt/data1/Nafiz/MammoGen-RAG/vindr/L_CC'
source_file_dir_M_L = '/mnt/data1/Nafiz/MammoGen-RAG/vindr/L_MLO'
source_file_dir_reports_L = '/mnt/data1/Nafiz/MammoGen-RAG/vindr/GROUND_TRUTH_REPORTS/L'

# Directories for Right-side
source_file_dir_C_R = '/mnt/data1/Nafiz/MammoGen-RAG/vindr/R_CC'
source_file_dir_M_R = '/mnt/data1/Nafiz/MammoGen-RAG/vindr/R_MLO'
source_file_dir_reports_R = '/mnt/data1/Nafiz/MammoGen-RAG/vindr/GROUND_TRUTH_REPORTS/R'

temp = 0
prompt_technique = "base"
prompt_template = """
I will provide you with two mammogram images. First one is the top-view of a breast whereas the second one is the side-view. Your task is to analyze the image and extract key diagnostic information, including breast composition, BIRADS category, and any significant findings. Present the output in a structured JSON format with the following keys: IMG_ID_CC, IMG_ID_MLO, Breast_Composition, BIRADS, and Findings. Ensure the response is precise, medically relevant, and well-organized.

IMPORTANT: Always use forward slashes (/) for any file paths. NEVER use backslashes (\\) in any path. All file paths must use forward slashes.

Present the output in a structured JSON format with the following keys: IMG_ID_CC, IMG_ID_MLO, Breast_Composition, BIRADS, and Findings. Ensure the response is precise, medically relevant, and well-organized.

Please follow the below given JSON format for your response
```json
{
    "IMG_ID_CC": "<Image_Filename>",
    "IMG_ID_MLO": "<Image_Filename>",
    "BREAST_COMPOSITION": "<Description of breast tissue composition>",
    "BIRADS": "<A single value from 0 to 6 indicating the BIRADS category>",
    "FINDINGS": "<Summary of any abnormalities, calcifications, or other observations found in any of the views>"
}
```"""

def extract_json_block(text):
    # Extract JSON inside triple backticks
    triple = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if triple:
        return triple.group(1)
    simple = re.search(r"\{.*\}", text, re.DOTALL)
    return simple.group(0) if simple else None


def process_side(model_name, side, cc_dir, mlo_dir, report_dir, reports_to_process):
    img_files = list_png_files(cc_dir)
    if reports_to_process > 0:
        img_files = img_files[:reports_to_process]

    processed = 0
    for idx, filename in enumerate(img_files, 1):
        cc_path = os.path.join(cc_dir, filename)
        mlo_path = os.path.join(mlo_dir, filename)
        report_path = os.path.join(report_dir, filename)

        if os.path.exists(report_path):
            with open(report_path) as f:
                report_text = f.read().strip()
        else:
            report_text = "No report available"

        query = (
            prompt_template +
            f"\nImage_CC: {cc_path}\nImage_MLO: {mlo_path}\nReport: {report_text}\n"
        )

        response = Ollama(model=model_name, temperature=temp).invoke(query)

        block = extract_json_block(response)
        if block:
            data = fix_json(block)
        else:
            data = {"IMG_ID_CC":"NA","IMG_ID_MLO":"NA","BREAST_COMPOSITION":"NA","BIRADS":"NA","FINDINGS":"NA"}

        out_dir = os.path.join('/mnt/data1/Nafiz/MammoGen-RAG/evaluated-vindr/_zeroshot', f"{model_name}_{side}")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, filename.replace('.png', '.json'))
        with open(out_file, 'w') as jf:
            json.dump(data, jf, indent=4)

        processed += 1
        # Print progress every 20 reports
        if processed % 20 == 0:
            print(f"{processed} reports processed for side {side}...")

    # Final count for this side
    print(f"Completed processing {processed} reports for side {side}.")
    return processed

@click.command()
@click.option("--model_name", default="llama3.1:latest", type=click.Choice(allowable_models))
@click.option("--reports_to_process", default=-1, type=int)
def main(model_name, reports_to_process):
    total = 0
    total += process_side(model_name, 'L', source_file_dir_C_L, source_file_dir_M_L, source_file_dir_reports_L, reports_to_process)
    total += process_side(model_name, 'R', source_file_dir_C_R, source_file_dir_M_R, source_file_dir_reports_R, reports_to_process)
    # Final overall total
    print(f"Total reports processed (both sides): {total}")

if __name__ == '__main__':
    main()

