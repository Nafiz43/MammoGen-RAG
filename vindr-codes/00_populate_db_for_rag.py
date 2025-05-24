import os
import json
from PIL import Image
import open_clip
import torch
import chromadb

# Base paths
vindr_base = '/mnt/data1/Nafiz/MammoGen-RAG/vindr'
img_dirs = {
    'L': {
        'CC': os.path.join(vindr_base, 'L_CC'),
        'MLO': os.path.join(vindr_base, 'L_MLO')
    },
    'R': {
        'CC': os.path.join(vindr_base, 'R_CC'),
        'MLO': os.path.join(vindr_base, 'R_MLO')
    }
}
json_base = os.path.join(vindr_base, 'GROUND_TRUTH_REPORTS')

# ChromaDB setup
chroma_client = chromadb.PersistentClient(path="chroma")
multimodal_db = chroma_client.get_or_create_collection(
    name="multimodal_db_all",
    embedding_function=None
)

# Load CLIP model
model, preprocess, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Parameters
BATCH_SIZE = 50

# Helper function to safely push batch to ChromaDB
def safe_batch_push(db, ids, uris, embeddings, metadatas):
    if embeddings:
        print(f"✅ Uploading batch of {len(embeddings)} entries...")
        db.add(ids=ids, uris=uris, embeddings=embeddings, metadatas=metadatas)

# Initialize batch
batch_ids, batch_uris, batch_meta, batch_embs = [], [], [], []
total_count = 0

for side in ('L', 'R'):
    side_json_dir = os.path.join(json_base, side)
    json_files = [f for f in os.listdir(side_json_dir) if f.endswith('.json')]

    print(f"Processing {len(json_files)} reports for side {side}...")

    for json_file in json_files:
        case_id = json_file.replace('.json', '')
        
        # Remove '_L' or '_R' suffix for image matching
        case_id = case_id.replace('_L', '').replace('_R', '')

        cc_path = os.path.join(img_dirs[side]['CC'], f"{case_id}.png")
        mlo_path = os.path.join(img_dirs[side]['MLO'], f"{case_id}.png")
        json_path = os.path.join(side_json_dir, json_file)

        # Check if both images exist
        if not (os.path.exists(cc_path) and os.path.exists(mlo_path)):
            print(f"⚠️ Missing image(s) for {case_id}, skipping...")
            continue

        try:
            # Load metadata
            with open(json_path, 'r') as jf:
                raw = json.load(jf)
            meta = {
                'Breast_Composition': raw.get('Breast_Composition', 'N/A'),
                'BIRADS': raw.get('BIRADS', 'N/A'),
                'Findings': raw.get('Findings', 'N/A')
            }

            # Preprocess & encode
            img_cc = preprocess(Image.open(cc_path).convert("RGB")).unsqueeze(0).to(device)
            img_mlo = preprocess(Image.open(mlo_path).convert("RGB")).unsqueeze(0).to(device)

            with torch.no_grad():
                e_cc = model.encode_image(img_cc)
                e_mlo = model.encode_image(img_mlo)

            comb = torch.cat([e_cc, e_mlo], dim=-1)
            emb_list = comb.cpu().numpy().flatten().tolist()

            # Add to batch
            batch_ids.append(f"{side}_{case_id}")
            batch_uris.append({'cc': cc_path, 'mlo': mlo_path})
            batch_meta.append(meta)
            batch_embs.append(emb_list)

            total_count += 1
            if total_count % 20 == 0:
                print(f"Processed {total_count} cases so far...")

            # Push batch if full
            if len(batch_ids) >= BATCH_SIZE:
                safe_batch_push(multimodal_db, batch_ids, batch_uris, batch_embs, batch_meta)
                batch_ids, batch_uris, batch_meta, batch_embs = [], [], [], []  # reset batch

        except Exception as e:
            print(f"⚠️ Error processing {case_id}: {e}")

# Push any remaining items
if batch_ids:
    safe_batch_push(multimodal_db, batch_ids, batch_uris, batch_embs, batch_meta)

print(f"🎯 Finished! Total indexed: {total_count}")