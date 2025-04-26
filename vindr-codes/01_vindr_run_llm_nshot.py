# from _constant import *


# # source_file_dir = '/mnt/data1/raiyan/breast_cancer/datasets/dmid/png_images/all_images/IMG'

# # source_file_dir =  '/mnt/data1/raiyan/breast_cancer/datasets/dmid/pixel_level_annotations/png_images/IMG'
# source_file_dir_C =  '/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/vindr/L_CC'
# source_file_dir_M =  '/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/vindr/L_MLO'
# source_file_dir_reports =  '/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/vindr/GROUND_TRUTH_REPORTS'
# # saving_dir = '/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/evaluated/llava_base/'


# img_files = list_png_files(source_file_dir)

# temp = 0
# prompt_technique = "base"
# prompt_template = """
# I will provide you with two mammogram images. First one is the top-view of a breast whereas the second one is the side-view. Your task is to analyze the image and extract key diagnostic information, including breast composition, BIRADS category, and any significant findings. Present the output in a structured JSON format with the following keys: IMG_ID_CC, IMG_ID_MLO, Breast_Composition, BIRADS, and Findings. Ensure the response is precise, medically relevant, and well-organized.
# Please follow the below given JSON format for your response
# {
#     "IMG_ID_CC": "<Image_Filename>",
#     "IMG-ID-MLO": "<Image_Filename>",
#     "BREAST-COMPOSITION" "<Description of breast tissue composition>",
#     "BIRADS": "<BIRADS category; any values between 1 to 6. BI-RADS category is a standardized classification for breast imaging findings, ranging from 1 to 6, where: BI-RADS 1 indicates a negative result with no abnormalities; BI-RADS 2 signifies benign findings with no suspicion of cancer; BI-RADS 3 suggests a benign lesion, requiring short-term follow-up to confirm stability; BI-RADS 4 represents a suspicious abnormality needing biopsy, further divided into 4A (low suspicion), 4B (moderate suspicion), and 4C (high suspicion); BI-RADS 5 is highly suggestive of malignancy with a high probability of cancer; and BI-RADS 6 confirms a known malignancy with a biopsy-proven cancer diagnosis.",
#     "FINDINGS": "<Summary of any abnormalities, calcifications, or other observations for both the views>"
# }

# """

# @click.command()
# @click.option(
#     "--model_name",
#     default="llama3.1:latest",
#     type=click.Choice(allowable_models),
#     help="name of the model to be used for processing",
# )
# @click.option(
#     "--reports_to_process", 
#     default=-1,  # Default value
#     type=int, 
#     help="An extra integer to be passed via command line"
# )

# def main(model_name, reports_to_process):
#     print(f"Received model_name: {model_name}")
#     print(f"Received value for reports_to_process: {reports_to_process}")

#     global data 

#     if(reports_to_process > 0):
#         # data = data.head(reports_to_process)
#         print(f"Processing only {reports_to_process} reports")
    
#     if(reports_to_process == -1):
#         reports_to_process = len(img_files)


#     for report in range(0, reports_to_process):
#         # report_id = source_file_dir + str(report+1).zfill(3)+'.png'
#         report_id = source_file_dir_reports + img_files[report]

#         print(report_id)
#         # image_id = 'IMG'+ str(report+1).zfill(3)
#         image_id =  img_files[report].replace('.png', '')

        
#         # query = 'image ID: ' + report_id
#         query = prompt_template+ 'image ID: '+  report_id

#         print("QUERY: ", query)

#         ollama = Ollama(model=model_name, temperature=temp)
#         logging.getLogger().setLevel(logging.ERROR)  # Suppress INFO logs
#         response = ollama.invoke(query)
#         print("RESPONSE: ",response)



#         json_match = re.search(r"\{.*\}", response, re.DOTALL)
#         if json_match in [None, ""]:
#             json_match = {"IMG_ID": "NA", "Breast_Composition": "NA", "BIRADS": "NA", "Findings": "NA"}
#         else:
#             json_match = fix_json(json_match.group(0))
        
#         print(json_match)
#         # global saving_dir
#         #constructing the saving dir here
#         saving_dir = 'evaluated-vindr/'+model_name+'_/'
#         print(saving_dir)

#         image_saving_dir = saving_dir +image_id + '.json'

#         os.makedirs(os.path.dirname(image_saving_dir), exist_ok=True)
#         with open(image_saving_dir, 'w') as json_file:
#             json.dump(json_match, json_file, indent=4)
        

#         print("Data has been written to", image_saving_dir)

#     print("\nTotal Reports Processed", reports_to_process)


# if __name__ == "__main__":
#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
#     )
#     main()

import os
import json
import re
import logging
import click
from _constant import *  # Assumes that list_png_files, fix_json, and allowable_models are provided

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
```

Here are some examples of doctor annotated reports to guide you:
Example 1:
{
    "IMG_ID_CC": "/mnt/data1/Nafiz/MammoGen-RAG/vindr/L_CC/0a0c5108270e814818c1ad002482ce74.png",
    "IMG_ID_MLO": "/mnt/data1/Nafiz/MammoGen-RAG/vindr/L_MLO/0a0c5108270e814818c1ad002482ce74.png",
    "Breast_Composition": "Heterogeneously dense breast parenchyma (ACR C)",
    "BIRADS": "1",
    "Findings": "No findings in both views"
}

Example 2:
{
    "IMG_ID_CC": "/mnt/data1/Nafiz/MammoGen-RAG/vindr/L_CC/0a936e2a8042114394878bee2c7ef727.png",
    "IMG_ID_MLO": "/mnt/data1/Nafiz/MammoGen-RAG/vindr/L_MLO/0a936e2a8042114394878bee2c7ef727.png",
    "Breast_Composition": "Extremely dense breast parenchyma",
    "BIRADS": "3",
    "Findings": "Focal Asymmetry in both views"
}

Example 3:
{
    "IMG_ID_CC": "/mnt/data1/Nafiz/MammoGen-RAG/vindr/R_CC/0ac8b6988e7443dc5e45d601a2f813f1.png",
    "IMG_ID_MLO": "/mnt/data1/Nafiz/MammoGen-RAG/vindr/R_MLO/0ac8b6988e7443dc5e45d601a2f813f1.png",
    "Breast_Composition": "Heterogeneously dense breast parenchyma (ACR C)",
    "BIRADS": "3",
    "Findings": "Focal Asymmetry, Mass in CC view. Focal Asymmetry in MLO view"
}
"""

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

        out_dir = os.path.join('/mnt/data1/Nafiz/MammoGen-RAG/evaluated-vindr/_nshot/', f"{model_name}_{side}")
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

