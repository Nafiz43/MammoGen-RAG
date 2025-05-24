import os
import logging
import click
import json
import re
from glob import glob
from datetime import datetime
from PIL import Image
import torch
import open_clip
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from torchvision import transforms
from langchain_ollama import OllamaLLM as Ollama

# Directories for images (recursive)
IMAGE_ROOT = '/mnt/data1/raiyan/breast_cancer/datasets/vindr/images_png'
JSON_ROOT  = '/mnt/data1/Nafiz/MammoGen-RAG/vindr/ground_truth_reports'

# Initialize ChromaDB with persistent storage
chroma_client = chromadb.PersistentClient(path="./chroma")
multimodal_db = chroma_client.get_or_create_collection(
    name="multimodal_db_all",
    embedding_function=OpenCLIPEmbeddingFunction(),
    data_loader=ImageLoader()
)

# Load OpenCLIP model and preprocessing
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess, _ = open_clip.create_model_and_transforms(
    "ViT-B-32-quickgelu", pretrained="openai"
)
model.to(device)

# Embedding helper
def get_image_embedding(pil_img):
    img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img_tensor)
    return emb.cpu().numpy().flatten().tolist()

# RAG retrieval helper
def retrieve_similar_images(query_path):
    img = Image.open(query_path).convert("RGB")
    q_emb = get_image_embedding(img)
    results = multimodal_db.query(
        query_embeddings=[q_emb],
        n_results=5  # fetch extra to allow skipping the query itself
    )
    return results

allowable_models = ["meditron:latest", "jyan1/paligemma-mix-224:latest", "qwen2.5:latest", "medllama2:latest", "llama3.1:latest", "gemma:7b-instruct", "mistral:latest", "mixtral:8x7b-instruct-v0.1-q4_K_M", 
         "llama2:latest", "llama2:70b-chat-q4_K_M", "llama2:13b-chat", "llama3.8b-instruct-q4_K_M", "llama3.3:70b", "llama3.2:latest", "meditron:70b", "tinyllama", "mistral", "mistral-nemo:latest", 
          'vanilj/llama-3-8b-instruct-32k-v0.1:latest', "mistrallite:latest", "mistral-nemo:12b-instruct-2407-q4_K_M", "llama3.2:3b-instruct-q4_K_M", "deepseek-r1:1.5b",
          "deepseek-r1:7b", "deepseek-r1:70b", "qordmlwls/llama3.1-medical:latest", "mixtral:latest","llava:latest"]

# Prompt template for LLM
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

# Utility to clean JSON-like strings
def remove_invalid_control_chars(s):
    return re.sub(r'[\x00-\x1F\x7F\\]', '', s)

def fix_json(json_input):
    if isinstance(json_input, dict):
        return json_input
    try:
        parsed = json.loads(json_input)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    # Fallback: trim until valid
    for i in range(len(json_input), 0, -1):
        try:
            parsed = json.loads(json_input[:i])
            if isinstance(parsed, dict):
                return parsed
        except:
            continue
    return {"IMG-ID":"NA","BREAST-COMPOSITION":"NA","FINDINGS":"NA","BIRADS":"NA"}

@click.command()
@click.option(
    "--model_name",
    default="llama3.1:latest",
    type=click.Choice([
        "meditron:latest","jyan1/paligemma-mix-224:latest","qwen2.5:latest",
        "medllama2:latest","llama3.1:latest","gemma:7b-instruct","mistral:latest",
        # ... other models ...
        "llava:latest"
    ]),
    help="Model to use for LLM processing"
)
@click.option(
    "--reports_to_process",
    default=-1,
    type=int,
    help="How many images to process (use -1 for all)"
)
def main(model_name, reports_to_process):
    # Discover all images recursively and build full ID-to-path map
    full_images = glob(os.path.join(IMAGE_ROOT, '**', '*.png'), recursive=True)
    full_images.sort()
    id_to_path = {os.path.basename(p): p for p in full_images}

    # Subset to process
    images_to_process = full_images if reports_to_process <= 0 else full_images[:reports_to_process]
    # print(f"Processing {len(images_to_process)} images from {IMAGE_ROOT}")
    
    cnt=0

    for img_path in images_to_process:
        img_name = os.path.basename(img_path)
        original_id = os.path.splitext(img_name)[0]
        # print(f"\n=== Processing {img_name} ===")

        # Retrieve similar embeddings
        sims = retrieve_similar_images(img_path)
        sim_ids = sims.get('ids', [[]])[0]
        metas   = sims.get('metadatas', [[]])[0]

        # Collect up to 2 examples, skipping the query itself
        examples = []
        for sim_id, meta in zip(sim_ids, metas):
            if sim_id == img_name:
                continue
            sim_path = id_to_path.get(sim_id)
            if sim_path:
                examples.append((sim_path, meta))
            if len(examples) == 2:
                break

        # Build context
        context = "Here are some examples of doctor-annotated reports to guide you:"
        for idx, (doc_uri, meta) in enumerate(examples, start=1):
            context += f"""

        Example {idx}:
        {{
            \"IMG-ID\": \"{doc_uri}\",
            \"BREAST-COMPOSITION\": \"{meta.get('BREAST-COMPOSITION','N/A')}\",
            \"FINDINGS\": \"{meta.get('FINDINGS','N/A')}\",
            \"BIRADS\": \"{meta.get('BIRADS','N/A')}\"
        }}"""

        # Assemble query
        query = prompt_template + f"\nImage ID: {img_path}\n" + context
        query = remove_invalid_control_chars(query)
        # print(query)

        # Invoke the LLM
        ollama = Ollama(model=model_name, temperature=0)
        response = ollama.invoke(query)
        response = remove_invalid_control_chars(response)
        # print(response)

        # Extract and fix JSON
        match = re.search(r"\{.*\}", response, re.DOTALL)
        result_json = fix_json(match.group(0) if match else "")
        # print(result_json)

        # Save to file
        save_dir = os.path.join('evaluated-vindr', f"{model_name}_rag_nshot")
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{original_id}.json")
        with open(out_path, 'w') as f:
            json.dump(result_json, f, indent=4)
        # print(f"Written → {out_path}")
        
        cnt+=1
        if(cnt%20==0):
            print("Processed", cnt, "reports with model", model_name)

    print(f"\nTotal processed: {len(images_to_process)}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    main()
