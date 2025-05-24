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
I will provide you a mammogram image. Your task is to analyze the image and extract key diagnostic information, including breast composition, any significant findings and finally assign a BIRADS category. 

Please analyze the mammogram I’m sending you and respond **only** with a JSON object (no extra text) containing exactly these keys:


Step 1: First find out the Breast Density Category in ACR Format where 
- A (Almost entirely fatty): The breast is composed mostly of fat with minimal glandular tissue. Low density → Easier to detect abnormalities. Least associated with an increased risk of breast cancer.

- B (Scattered areas of fibroglandular density): Mostly fatty, but with some scattered dense tissue. Abnormalities are still generally well-detected. Slightly increased breast cancer risk compared to ACR A.


- C (Heterogeneously dense): A significant amount of glandular tissue, making the breast more dense. Can obscure small tumors, making detection more challenging. Moderately increased risk of breast cancer.


- D (Extremely dense): The breast is almost entirely composed of dense fibroglandular tissue. High density makes it very difficult to detect abnormalities on a mammogram. Significantly increased risk of breast cancer.


Step 2: Then determine any abnormal findings (or tumors) in the image. Findings are abnormalities or observations detected in a mammogram. Each type indicates different levels of concern: 

- No suspicious findings. Routine screening continues as per guidelines. 

- Calcification: Tiny calcium deposits in the breast tissue. Can be benign (due to aging, injury, or inflammation) or suspicious (clustered, irregular shapes, which may indicate early cancer). Further evaluation is required for suspicious calcifications.

- Architectural Distortion: Distortion of normal breast tissue structure without a clearly defined mass. Can be caused by prior surgery, trauma, or malignancy. Further imaging (MRI, ultrasound) or biopsy is often needed.

- Asymmetry: One breast appears different from the other in density or structure. Can be due to normal variations, prior surgery, or an underlying lesion. If new or developing asymmetry is observed, additional tests may be needed.

- Nipple Retraction

- Global Asymmetry

- Focal Asymmetry

- Mass 

- Skin Thickening

- Suspicious Lymph Node

- Skin Retraction

Step 3: Now finally, determine the BIRADS of the mammogram where 


 "BIRADS": "<BIRADS category; any values between 1 to 6. BI-RADS category is a standardized classification for breast imaging findings, ranging from 1 to 6, where: BI-RADS 1 indicates a negative result with no abnormalities; BI-RADS 2 signifies benign findings with no suspicion of cancer; BI-RADS 3 suggests a benign lesion, requiring short-term follow-up to confirm stability; BI-RADS 4 represents a suspicious abnormality needing biopsy, further divided into 4A (low suspicion), 4B (moderate suspicion), and 4C (high suspicion); BI-RADS 5 is highly suggestive of malignancy with a high probability of cancer; and BI-RADS 6 confirms a known malignancy with a biopsy-proven cancer diagnosis.",



Step 4: Please follow the below given JSON format for your response:

- "Image": the absolute path of the image that is being processed.
- "Breast Composition": one of A, B, C, or D, A indicates almost entirely fatty tissue, B indicates scattered areas of fibroglandular tissue, C indicates heterogeneously dense tissue, and D indicates extremely dense.  
- "Findings": a summary of any abnormal findings like presence of Mass, Architectural Distortion, Calcification, Nipple Retraction, Focal Asymmetry, Skin Thickening, Global Asymmetry, Asymmetry and so on in any place of the breast or say "No Finding" for nothing abnormal. 
- "BIRADS": an integer 1–6, where 1 is healthy, 2 is probably benign findings, 3 is highly likely benign findings, 4 is suspicious malignancy, 5 is highly suggestive of malignancy, and 6 is known biopsy-proven malignancy.

Do not include any other keys or commentary.



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

        model_dir = os.path.join(SAVING_BASE, f"{model_name}_zeroshot")
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