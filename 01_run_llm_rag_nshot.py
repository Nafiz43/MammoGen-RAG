import os
import logging
import click
import pandas as pd
import csv
from datetime import datetime
from langchain_ollama import OllamaLLM as Ollama
import sys
from pydantic import BaseModel
import random
import os
import json
import re
from glob import glob
from PIL import Image
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from matplotlib import pyplot as plt
from torchvision import transforms
import torch
import open_clip

# source_file_dir = '/mnt/data1/raiyan/breast_cancer/datasets/dmid/png_images/all_images/IMG'

chroma_client = chromadb.PersistentClient(path="./chroma")  # Persistent storage
multimodal_db = chroma_client.get_or_create_collection(name="multimodal_db_all")


source_file_dir =  '/mnt/data1/raiyan/breast_cancer/datasets/dmid/pixel_level_annotations/png_images/IMG'


temp = 0
prompt_technique = "base"
prompt_template = """
I am providing you a mammogram image. Your task is to analyze the image and extract key diagnostic information, including breast composition, any significant findings, and assign the BIRADS score. Present the output in a structured JSON format with the following keys: IMG_ID, Breast_Composition, Findings, and BIRADS. Ensure the response is precise, medically relevant, and well-organized.
Please follow the below given JSON format for your response and only output a valid json object:
{
    "IMG-ID" "<Image_Filename>",
    "BREAST-COMPOSITION" "<Provide the tissue density in ACR format where ACR A is almost entirely fatty, ACR B is scattered fibroglandular densities, ACR C is heterogeneously dense, and ACR D is extremely dense>",
    "FINDINGS": "<Summary of any abnormalities, calcifications, or other observations>",
    "BIRADS": "<BIRADS category; any values between 1 to 6. BI-RADS category is a standardized classification for breast imaging findings, ranging from 1 to 6, where: BI-RADS 1 indicates a negative result with no abnormalities; BI-RADS 2 signifies benign findings with no suspicion of cancer; BI-RADS 3 suggests a benign lesion, requiring short-term follow-up to confirm stability; BI-RADS 4 represents a suspicious abnormality needing biopsy, further divided into 4A (low suspicion), 4B (moderate suspicion), and 4C (high suspicion); BI-RADS 5 is highly suggestive of malignancy with a high probability of cancer; and BI-RADS 6 confirms a known malignancy with a biopsy-proven cancer diagnosis.",
}

"""



allowable_models = ["meditron:latest", "jyan1/paligemma-mix-224:latest", "qwen2.5:latest", "medllama2:latest", "llama3.1:latest", "gemma:7b-instruct", "mistral:latest", "mixtral:8x7b-instruct-v0.1-q4_K_M", 
         "llama2:latest", "llama2:70b-chat-q4_K_M", "llama2:13b-chat", "llama3.8b-instruct-q4_K_M", "llama3.3:70b", "llama3.2:latest", "meditron:70b", "tinyllama", "mistral", "mistral-nemo:latest", 
          'vanilj/llama-3-8b-instruct-32k-v0.1:latest', "mistrallite:latest", "mistral-nemo:12b-instruct-2407-q4_K_M", "llama3.2:3b-instruct-q4_K_M", "deepseek-r1:1.5b",
          "deepseek-r1:7b", "deepseek-r1:70b", "qordmlwls/llama3.1-medical:latest", "mixtral:latest","llava:latest"]


# Load OpenCLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = open_clip.create_model("ViT-B-32", pretrained="openai").to(device)
# open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")


# Preprocessing for OpenCLIP (manual transform)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.481, 0.457, 0.408), std=(0.268, 0.261, 0.275))  # CLIP normalization
])

# Function to generate image embeddings
def get_image_embedding(image):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image)
    return embedding.cpu().numpy().flatten().tolist()  # Convert to list of floats


# Function to retrieve top 3 similar images from ChromaDB
def retrieve_similar_images(query_image_path):
    # Load and preprocess the query image
    query_image = Image.open(query_image_path).convert("RGB")
    query_embedding = get_image_embedding(query_image)
    
    # Query ChromaDB for the most similar embeddings
    results = multimodal_db.query(
        query_embeddings=[query_embedding],  # Query using the passed image embedding
        n_results=3  # Retrieve top 3 results
    )

    # print(results)
    return results
    
    # # Inspect and log the full results to understand their structure
    # print("Full Query Results:")
    # print(results)
    
    # Extracting the IDs and metadata and storing them in the required format.
    # formatted_data = []

    # for img_id, metadata in zip(results['ids'][0], results['metadatas'][0]):
    #     formatted_entry = {
    #         "IMG-ID": img_id,
    #         "BREAST-COMPOSITION": metadata['Breast_Composition'].replace('\n', ' ').strip(),
    #         "BIRADS": str(metadata['BIRADS']),
    #         "FINDINGS": metadata['Findings'].replace('\n', ' ').strip()
    #     }
    #     formatted_data.append(formatted_entry)
        
    # example_1=formatted_data[0]
    # example_2=formatted_data[1]
    



# Define the expected JSON schema using Pydantic
class ClassificationResponse(BaseModel):
    reason_for_the_label: str
    label: int  # Assuming 'Label' is an integer

import json


def remove_invalid_control_chars(input_string):
    """
    Removes control characters (ASCII 0-31 and 127), except for tab, newline, and carriage return.
    Also removes backslashes (\) explicitly.
    """
    # Remove control characters and backslashes
    cleaned_string = re.sub(r'[\x00-\x1F\x7F\\]', '', input_string)
    return cleaned_string



def fix_json(json_input):
    """
    Ensures the input is a JSON string or a dictionary and always returns a dictionary.
    If input is a dictionary, return it as-is.
    If input is a valid JSON string, return parsed JSON as a dictionary.
    If input is an invalid JSON string, attempts to fix it by trimming trailing characters.
    """
    # If input is already a dictionary, return it directly
    if isinstance(json_input, dict):
        return json_input  

    # Ensure input is a string (or bytes), otherwise return error JSON
    if not isinstance(json_input, (str, bytes, bytearray)):
        return {"IMG-ID": "NA", "BREAST-COMPOSITION": "NA", "Findings": "NA", "BIRADS": "NA"}

    # First, check if the JSON is already valid
    try:
        parsed_json = json.loads(json_input)
        if isinstance(parsed_json, dict):
            return parsed_json  # Ensure it's a dictionary
        else:
            return {"IMG-ID": "NA", "BREAST-COMPOSITION": "NA", "Findings": "NA", "BIRADS": "NA"}
    except json.JSONDecodeError:
        pass  # If invalid, proceed with fixing

    # Try trimming trailing characters
    for i in range(len(json_input), 0, -1):  
        try:
            parsed_json = json.loads(json_input[:i])  # Try parsing progressively shorter substrings
            if isinstance(parsed_json, dict):
                return parsed_json  # Ensure it's a dictionary
        except Exception as e:
            print(f"Unexpected error: {e}")  # Catch other unforeseen errors
            continue  # Keep trimming

    # If all attempts fail, return error JSON
    return {"IMG-ID": "NA", "BREAST-COMPOSITION": "NA", "Findings": "NA", "BIRADS": "NA"}

@click.command()
@click.option(
    "--model_name",
    default="llama3.1:latest",
    type=click.Choice(allowable_models),
    help="name of the model to be used for processing",
)
@click.option(
    "--reports_to_process", 
    default=-1,  # Default value
    type=int, 
    help="An extra integer to be passed via command line"
)





def main(model_name, reports_to_process):
    print(f"Received model_name: {model_name}")
    print(f"Received value for reports_to_process: {reports_to_process}")

    global data 

    if(reports_to_process > 0):
        # data = data.head(reports_to_process)
        print(f"Processing only {reports_to_process} reports")
    
    if(reports_to_process == -1):
        reports_to_process = 510

    for report in range(0, reports_to_process):
        report_id = source_file_dir + str(report+1).zfill(3)+'.png'
        print(report_id)
        original_image_id = 'IMG'+ str(report+1).zfill(3)
        
        similar_images = retrieve_similar_images(report_id)
        
        # print("printing similar images::::::::", similar_images)


        context = "Here are some examples of doctor annoted reports to guide you: "
        for index in range(1, 3):
            image_id = similar_images["ids"][0][index]  # Get the image ID
            img_findings = similar_images["metadatas"][0][index]['Findings']  # Get the findings
            img_birads = similar_images["metadatas"][0][index]['BIRADS']  # Get the BIRADS score
            img_breast_composition = similar_images["metadatas"][0][index]['Breast_Composition']  # Get the breast composition

            # Dynamically build the context string with the actual values
            context += f"""
            Example :{index}
            {{
            "IMG-ID": /mnt/data1/raiyan/breast_cancer/datasets/dmid/pixel_level_annotations/png_images/{image_id},
            "BREAST-COMPOSITION": "{img_breast_composition}",
            "FINDINGS": "{img_findings}",
            "BIRADS": "{img_birads}"
            }}
            """

        # print(context)

        query = prompt_template+ 'image ID: '+  report_id + "  \n  " +context
        query = remove_invalid_control_chars(query)
        query = query.replace('\\', '')
        print(query)


        ollama = Ollama(model=model_name, temperature=temp)
        logging.getLogger().setLevel(logging.ERROR)  # Suppress INFO logs

        response = ollama.invoke(query)
        response = remove_invalid_control_chars(response)
        response = response.replace('\\', '')
        print(response)

        # response = ""
        # print(response)

        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match in [None, ""]:
            json_match = {"IMG-ID": "NA", "BREAST-COMPOSITION": "NA", "Findings": "NA", "BIRADS": "NA"}

        else:
            json_match = fix_json(json_match.group(0))
        
        print(json_match)
        
        #constructing the saving dir here
        saving_dir = 'evaluated/'+model_name+'_nshot_rag/'
        print(saving_dir)

        image_saving_dir = saving_dir +original_image_id + '.json'

        os.makedirs(os.path.dirname(image_saving_dir), exist_ok=True)
        with open(image_saving_dir, 'w') as json_file:
            json.dump(json_match, json_file, indent=4)
        

        print("Data has been written to", image_saving_dir)

    print("\nTotal Reports Processed", reports_to_process)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
