import os
import json
from PIL import Image
import open_clip
import torch
import chromadb
from chromadb.utils.data_loaders import ImageLoader

# Base paths
img_base  = '/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/vindr'
json_base = '/mnt/data1/Nafiz/MammoGen-RAG/vindr/GROUND_TRUTH_REPORTS'

# ChromaDB setup (as before)
chroma_client  = chromadb.PersistentClient(path="chroma")
multimodal_db  = chroma_client.get_or_create_collection(
    name="multimodal_db_all",
    embedding_function=None
)

# Load CLIP model + preprocess
model, preprocess, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

all_ids, all_uris, all_meta, all_embs = [], [], [], []
count = 0

for side in ('L', 'R'):
    # Directories for this side
    cc_dir   = os.path.join(img_base, f"{side}_CC")
    mlo_dir  = os.path.join(img_base, f"{side}_MLO")
    json_dir = os.path.join(json_base, side)

    # Gather filenames
    cc_pngs   = {f for f in os.listdir(cc_dir)  if f.lower().endswith('.png')}
    mlo_pngs  = {f for f in os.listdir(mlo_dir) if f.lower().endswith('.png')}
    json_pngs = {f[:-5]+'.png' for f in os.listdir(json_dir) if f.lower().endswith('.json')}

    # Only those names present in all three
    common = sorted(cc_pngs & mlo_pngs & json_pngs)

    for fname in common:
        cc_path   = os.path.join(cc_dir,   fname)
        mlo_path  = os.path.join(mlo_dir,  fname)
        json_path = os.path.join(json_dir, fname.replace('.png', '.json'))

        # Load metadata
        with open(json_path, 'r') as jf:
            raw = json.load(jf)
        meta = {
            'Breast_Composition': raw.get('Breast_Composition', 'N/A'),
            'BIRADS':            raw.get('BIRADS', 'N/A'),
            'Findings':          raw.get('Findings', 'N/A')
        }

        # Preprocess & encode both views
        img_cc  = preprocess(Image.open(cc_path).convert("RGB")).unsqueeze(0).to(device)
        img_mlo = preprocess(Image.open(mlo_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            e_cc  = model.encode_image(img_cc)
            e_mlo = model.encode_image(img_mlo)

        # Concatenate embeddings
        comb = torch.cat([e_cc, e_mlo], dim=-1)
        emb_list = comb.cpu().numpy().flatten().tolist()

        # Record
        all_ids.append(f"{side}_{fname[:-4]}")  
        all_uris.append({'cc': cc_path, 'mlo': mlo_path})
        all_meta.append(meta)
        all_embs.append(emb_list)

        count += 1
        if count % 20 == 0:
            print(f"Indexed {count} cases so far…")

print(f"Total indexed: {count}")

# Push to ChromaDB
multimodal_db.add(
    ids=all_ids,
    uris=all_uris,
    embeddings=all_embs,
    metadatas=all_meta
)
print("Done.")
