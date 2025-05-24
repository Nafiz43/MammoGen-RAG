import os
import re
import json
import click
import logging
from ollama import Client            # <-- use the Client class
from _constant import allowable_models

# where your ground-truth JSONs live
SOURCE_DIR = "/mnt/data1/Nafiz/MammoGen-RAG/vindr/ground_truth_reports"
# base folder for evaluated outputs
SAVING_BASE = "/mnt/data1/Nafiz/MammoGen-RAG/evaluated-vindr/"

# load all .json filenames (they’re named after your PNGs)
all_reports = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".json")]

# fixed temperature
temp = 0
prompt_technique = "base"
prompt_template = """
Consider you are helping a radiologist. I will provide you a mammogram image. Your task is to analyze the image and extract key diagnostic information, including breast composition, any significant findings and finally assign a BIRADS category. 

Please analyze the mammogram I’m sending you and respond **only** with a JSON object (no extra text) containing exactly these keys:

{
    "Image": "<the absolute path of the image that is being processed>",
    "Breast Composition": "<one of A, B, C, or D, A indicates almost entirely fatty tissue, B indicates scattered areas of fibroglandular tissue, C indicates heterogeneously dense tissue, and D indicates extremely dense.>",  
    "Findings": "<a summary of any abnormal findings like presence of Mass, Architectural Distortion, Calcification, Nipple Retraction, Focal Asymmetry, Skin Thickening, Global Asymmetry, Asymmetry and so on in any place of the breast or say "No Finding" for nothing abnormal.>", 
    "BIRADS": "<an integer 1–6, where 1 is healthy, 2 is probably benign findings, 3 is highly likely benign findings, 4 is suspicious malignancy, 5 is highly suggestive of malignancy, and 6 is known biopsy-proven malignancy.>"
}
Do not include any other keys or commentary.

Here are some examples of doctor annotated reports to guide you:
Example 1:
{
    "Image": "/mnt/data1/raiyan/breast_cancer/datasets/vindr/images_png/f89214e649de29a9e050564b894f4036/0a0c8b8a32ff32b0feb0d76e93b8dcad.png",
    "Breast Composition": "C",
    "Findings": "['No Finding']",
    "BIRADS": "1"
}

Example 2:
{
    "Image": "/mnt/data1/raiyan/breast_cancer/datasets/vindr/images_png/50a8bfda39adc93f22a174153632b823/0a6f0eac805822cd00f3c25ba99569e8.png",
    "Breast Composition": "C",
    "Findings": "['Mass']",
    "BIRADS": "5"
}

Example 3:
{
    "Image": "/mnt/data1/raiyan/breast_cancer/datasets/vindr/images_png/1a5c56c6f83b913488686f7fca265194/0a4004a005e1d213aed0691106d8772f.png",
    "Breast Composition": "B",
    "Findings": "['No Finding']",
    "BIRADS": "2"
}

Example 4:
{
    "Image": "/mnt/data1/raiyan/breast_cancer/datasets/vindr/images_png/7c51789da6c462e55bcb95c2a7d437ee/f581ef53bb7e61f4575db33eceac8ff8.png",
    "Breast Composition": "C",
    "Findings": "['Nipple Retraction', 'Mass']",
    "BIRADS": "4"
}

Example 5:
{
    "Image": "/mnt/data1/raiyan/breast_cancer/datasets/vindr/images_png/2ad573bf0a419ab289c3a23c6a6db18f/d6db56d5cb90391e234dca76c4a344e4.png",
    "Breast Composition": "A",
    "Findings": "['Global Asymmetry']",
    "BIRADS": "3"
}

"""

@click.command()
@click.option(
    "--model_name",
    default="llama3.1:latest",
    type=click.Choice(allowable_models),
    help="Name of the Ollama vision-capable model",
)
@click.option(
    "--reports_to_process",
    type=int,
    default=None,
    help="Number of reports to process (default: all)",
)
def main(model_name, reports_to_process):
    total = len(all_reports)
    if reports_to_process is None:
        reports_to_process = total
        print(f"No --reports_to_process—processing all {total} reports.")
    else:
        print(f"Processing first {reports_to_process} of {total} reports.")

    client = Client()

    for idx, json_fname in enumerate(all_reports[:reports_to_process], start=1):
        json_path = os.path.join(SOURCE_DIR, json_fname)
        record = json.load(open(json_path, "r"))

        image_path = record.get("Image")
        print(f"\n[{idx}/{reports_to_process}] → {image_path}")

        msg = {
            "role":    "user",
            "content": prompt_template,
            "images": [image_path]
        }

        # pass temperature via options
        resp = client.chat(
            model=model_name,
            messages=[msg],
            options={"temperature": temp}
        )
        text = resp["message"]["content"]
        # print("Model says:", text)

        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            out = {"Image": image_path, "Breast Composition": "NA",
                   "Findings": "NA", "BIRADS": "NA"}
        else:
            out = json.loads(m.group(0))

        model_dir = os.path.join(SAVING_BASE, f"{model_name}_nshot")
        os.makedirs(model_dir, exist_ok=True)
        image_id  = os.path.splitext(json_fname)[0]
        out_path  = os.path.join(model_dir, f"{image_id}.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=4)

        print("Saved →", out_path)

    print(f"\n✅ Done! Processed {reports_to_process} reports.")

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
    )
    main()