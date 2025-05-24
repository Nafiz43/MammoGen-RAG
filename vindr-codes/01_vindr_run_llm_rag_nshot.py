import os
import logging
import json
import re
import click
from PIL import Image
import torch
import open_clip
import chromadb
from torchvision import transforms
from pydantic import BaseModel
from langchain_ollama import OllamaLLM as Ollama

# ——— Paths for paired views ———
BASE_IMG_DIR = '/mnt/data1/Nafiz/MammoGen-RAG/vindr'

# Side-specific directories
SIDE_DIRS = {
    'L': {'cc': os.path.join(BASE_IMG_DIR, 'L_CC'), 'mlo': os.path.join(BASE_IMG_DIR, 'L_MLO')},
    'R': {'cc': os.path.join(BASE_IMG_DIR, 'R_CC'), 'mlo': os.path.join(BASE_IMG_DIR, 'R_MLO')}
}

# Prompt template with placeholders
PROMPT_TEMPLATE = r"""
I will provide you with two mammogram images. First one is the top-view of a breast whereas the second one is the side-view. Your task is to analyze the image and extract key diagnostic information, including breast composition, BIRADS category, and any significant findings. Present the output in a structured JSON format with the following keys: IMG_ID_CC, IMG_ID_MLO, Breast_Composition, BIRADS, and Findings. Ensure the response is precise, medically relevant, and well-organized.

IMPORTANT: Always use forward slashes (/) for any file paths. NEVER use backslashes (\\) in any path. All file paths must use forward slashes.

Please follow the below given JSON format for your response
```json
{{
    "IMG_ID_CC": "<Image_Filename>",
    "IMG_ID_MLO": "<Image_Filename>",
    "BREAST_COMPOSITION": "<Description of breast tissue composition>",
    "BIRADS": "<A single value from 0 to 6 indicating the BIRADS category>",
    "FINDINGS": "<Summary of any abnormalities, calcifications, or other observations found in any of the views>"
}}
```

Here are some examples of doctor annotated reports to guide you:
{context}
"""

# ChromaDB setup
chroma_client = chromadb.PersistentClient(path="./chroma")
multimodal_db = chroma_client.get_or_create_collection(
    name="multimodal_db_all",
    embedding_function=None
)

# Verify collection
try:
    num_entries = multimodal_db.count()
    print(f"📊 ChromaDB collection contains {num_entries} embeddings")
    if num_entries == 0:
        raise RuntimeError("ChromaDB collection is empty—run the embedding indexer first.")
except AttributeError:
    print("⚠️ Cannot determine collection size; ensure embeddings have been indexed.")

# Load OpenCLIP model & preprocess
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32-quickgelu", pretrained="openai"
)
model = model.to(device)

# Helpers

def get_image_embedding(image: Image.Image) -> list:
    img_t = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img_t)
    return emb.cpu().numpy().flatten().tolist()


def retrieve_similar_images(cc_path: str, mlo_path: str, top_k: int = 3) -> list:
    """
    Fetches the top_k neighbors for a CC/MLO pair and splits their embeddings into two 512-vectors.

    Returns a list of dicts with keys:
      - id: entry identifier
      - distance: similarity score
      - uri: {'cc': path, 'mlo': path}
      - metadata: clinical labels
      - cc_embedding: 512-dim vector for the CC view
      - mlo_embedding: 512-dim vector for the MLO view
    """
    # Load and embed both input views
    img_cc = Image.open(cc_path).convert("RGB")
    img_mlo = Image.open(mlo_path).convert("RGB")
    emb_cc = get_image_embedding(img_cc)
    emb_mlo = get_image_embedding(img_mlo)
    combined = emb_cc + emb_mlo

    # Query ChromaDB for full-pair embeddings
    result = multimodal_db.query(
        query_embeddings=[combined],
        n_results=top_k,
        include=["documents", "metadatas", "distances", "embeddings", "uris"]
    )

    # Parse results
    ids = result.get('ids', [[]])[0]
    docs = result.get('documents', [[]])[0]
    metas = result.get('metadatas', [[]])[0]
    dists = result.get('distances', [[]])[0]
    embs = result.get('embeddings', [[]])[0]

    neighbors = []
    half = len(embs[0]) // 2  # should be 512
    for i, nid in enumerate(ids):
        full_emb = embs[i]
        cc_emb_nb = full_emb[:half]
        mlo_emb_nb = full_emb[half:]
        neighbors.append({
            'id': nid,
            'distance': dists[i],
            'uri': docs[i],
            'metadata': metas[i],
            'cc_embedding': cc_emb_nb,
            'mlo_embedding': mlo_emb_nb
        })
    return neighbors


def remove_invalid_control_chars(text: str) -> str:
    return re.sub(r'[\x00-\x1F\x7F\\]', '', text)

class ClassificationResponse(BaseModel):
    IMG_ID_CC: str
    IMG_ID_MLO: str
    BREAST_COMPOSITION: str
    BIRADS: str
    FINDINGS: str

# Main processing function

def process_side(model_name: str, side: str) -> int:
    cc_dir = SIDE_DIRS[side]['cc']
    mlo_dir = SIDE_DIRS[side]['mlo']
    processed = 0
    json_dir = os.path.join(BASE_IMG_DIR, 'GROUND_TRUTH_REPORTS', side)
    json_files = [f for f in os.listdir(json_dir) if f.endswith(f"_{side}.json")]

    for json_file in json_files:
        case_id = json_file[:-len(f"_{side}.json")]
        base_name = f"{case_id}.png"
        cc_path = os.path.join(cc_dir, base_name)
        mlo_path = os.path.join(mlo_dir, base_name)

        if not (os.path.exists(cc_path) and os.path.exists(mlo_path)):
            print(f"⚠️ Missing view for {base_name}, skipping.")
            continue

        # Retrieve neighbors from ChromaDB
        neighbors = retrieve_similar_images(cc_path, mlo_path, top_k=3)
        if not neighbors:
            print(f"⚠️ No neighbors for {base_name}, skipping.")
            continue

        # Build RAG context with three examples
        context = "Here are three examples of doctor-annotated reports:\n"
        for idx, nbr in enumerate(neighbors, 1):
            uri = nbr.get('uri', {}) or {}
            meta = nbr.get('metadata', {}) or {}
            context += (
                f"Example {idx}: {{\n"
                f"  \"IMG_ID_CC\": \"{uri.get('cc','')}\",\n"
                f"  \"IMG_ID_MLO\": \"{uri.get('mlo','')}\",\n"
                f"  \"BREAST_COMPOSITION\": \"{meta.get('Breast_Composition','')}\",\n"
                f"  \"BIRADS\": \"{meta.get('BIRADS','')}\",\n"
                f"  \"FINDINGS\": \"{meta.get('Findings','')}\"\n}}\n"
            )

        # Inject neighbors into prompt and call the model
        prompt_text = PROMPT_TEMPLATE.format(context=context)
        raw_resp = Ollama(model=model_name, temperature=0).invoke(prompt_text)
        response = remove_invalid_control_chars(raw_resp)

        # Parse and save LLM output
        m = re.search(r"\{.*\}", response, re.DOTALL)
        parsed = json.loads(m.group(0)) if m else {}

        out_dir = os.path.join(
            '/mnt/data1/Nafiz/MammoGen-RAG/evaluated-vindr/_rag_nshot',
            f"{model_name}_{side}_3shot"
        )
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{case_id}.json"), 'w') as f:
            json.dump(parsed, f, indent=4)

        processed += 1
        if processed % 1000 == 0:
            print(f"[{side}] Processed {processed} reports...")

    print(f"Completed {processed} reports for side {side}.")
    return processed


@click.command()
@click.option("--model_name", default="llama3.1:latest", type=str)
def main(model_name: str):
    logging.basicConfig(level=logging.ERROR)
    total_l = process_side(model_name, 'L')
    total_r = process_side(model_name, 'R')
    print(f"Total reports processed: {total_l + total_r}")

if __name__ == '__main__':
    main()
