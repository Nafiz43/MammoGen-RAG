import os
import logging
import click
import json
import re
from glob import glob
from PIL import Image
import torch
import open_clip
import chromadb
from torchvision import transforms
from pydantic import BaseModel
from langchain_ollama import OllamaLLM as Ollama

# ——— Paths for paired views ———
BASE_IMG_DIR = '/mnt/data1/Nafiz/MammoGen-RAG/vindr'
JSON_BASE_DIR = '/mnt/data1/Nafiz/MammoGen-RAG/vindr/GROUND_TRUTH_REPORTS'

# ChromaDB setup
ochroma_client = chromadb.PersistentClient(path="./chroma")
multimodal_db = chroma_client.get_or_create_collection(
    name="multimodal_db_all"
)

# Load OpenCLIP model & preprocess
device = "cuda" if torch.cuda.is_available() else "cpu"
model = open_clip.create_model("ViT-B-32", pretrained="openai").to(device)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.481, 0.457, 0.408), std=(0.268, 0.261, 0.275))
])

def get_image_embedding(image: Image.Image) -> list:
    img_t = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img_t)
    return emb.cpu().numpy().flatten().tolist()

# Modify: accept two view paths and concatenate embeddings
def retrieve_similar_images(cc_path: str, mlo_path: str, top_k: int = 3) -> dict:
    # load both views
    img_cc  = Image.open(cc_path).convert("RGB")
    img_mlo = Image.open(mlo_path).convert("RGB")

    # get embeddings
    emb_cc  = get_image_embedding(img_cc)
    emb_mlo = get_image_embedding(img_mlo)

    # concatenate for 1024-d query
    combined = emb_cc + emb_mlo

    # query vector store
    results = multimodal_db.query(
        query_embeddings=[combined],
        n_results=top_k
    )
    return results

# JSON cleanup helpers

def remove_invalid_control_chars(input_string: str) -> str:
    return re.sub(r'[\x00-\x1F\x7F\\]', '', input_string)

class ClassificationResponse(BaseModel):
    IMG_ID_CC: str
    IMG_ID_MLO: str
    BREAST_COMPOSITION: str
    BIRADS: str
    FINDINGS: str

@click.command()
@click.option("--model_name", default="llama3.1:latest", type=str)
@click.option("--reports_to_process", default=-1, type=int)
def main(model_name, reports_to_process):
    logging.basicConfig(level=logging.ERROR)
    temp = 0

    # Determine how many to process
    total = reports_to_process if reports_to_process > 0 else 510

    for idx in range(total):
        # build IDs & paths (example: side 'L' or 'R' can be inferred or parameterized)
        side = 'L'  # or set dynamically per your dataset split logic
        base_name = f"IMG{str(idx+1).zfill(3)}.png"
        cc_path  = os.path.join(BASE_IMG_DIR, f"{side}_CC", base_name)
        mlo_path = os.path.join(BASE_IMG_DIR, f"{side}_MLO", base_name)
        report_json_dir = os.path.join(JSON_BASE_DIR, side)

        if not (os.path.exists(cc_path) and os.path.exists(mlo_path)):
            continue

        # Retrieve neighbors
        results = retrieve_similar_images(cc_path, mlo_path, top_k=3)

        # Build RAG context with paired-view examples
        context = "Here are some examples of doctor-annotated reports:\n"
        for i in range(len(results['ids'][0])):
            uri_dict = results['uris'][0][i]
            meta     = results['metadatas'][0][i]
            context += f"Example {i+1}: {{\n"
            context += f"  \"IMG_ID_CC\": \"{uri_dict['cc']}\",\n"
            context += f"  \"IMG_ID_MLO\": \"{uri_dict['mlo']}\",\n"
            context += f"  \"BREAST_COMPOSITION\": \"{meta['Breast_Composition']}\",\n"
            context += f"  \"BIRADS\": \"{meta['BIRADS']}\",\n"
            context += f"  \"FINDINGS\": \"{meta['Findings']}\"\n}}\n"

        # Build the prompt
        prompt = (
            "I will provide you with two mammogram images…"
            f"\n\nIMG_ID_CC: {cc_path}\n"
            f"IMG_ID_MLO: {mlo_path}\n"
            f"{context}"
        )

        prompt = remove_invalid_control_chars(prompt)
    

        # print(context)

        # Invoke LLM via Ollama
        ollama = Ollama(model=model_name, temperature=temp)
        response = ollama.invoke(prompt)
        response = remove_invalid_control_chars(response)

        # Extract JSON
        m = re.search(r"\{.*\}", response, re.DOTALL)
        parsed = json.loads(m.group(0)) if m else {}

        # Save output
        save_dir = os.path.join('evaluated', f"{model_name}_paired_view")
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{side}_{str(idx+1).zfill(3)}.json")
        with open(out_path, 'w') as f:
            json.dump(parsed, f, indent=4)

        print(f"Written: {out_path}")

if __name__ == '__main__':
    main()
