import os
from PIL import Image
import torch
import open_clip
import chromadb
import json

# ─── CONFIG ──────────────────────────────────────────────────────────────────
BASE_DIR    = '/mnt/data1/Nafiz/MammoGen-RAG/vindr'
REPORT_DIR  = os.path.join(BASE_DIR, 'GROUND_TRUTH_REPORTS')
CC_DIRS     = {'L': os.path.join(BASE_DIR, 'L_CC'),
               'R': os.path.join(BASE_DIR, 'R_CC')}
MLO_DIRS    = {'L': os.path.join(BASE_DIR, 'L_MLO'),
               'R': os.path.join(BASE_DIR, 'R_MLO')}

# ─── LOAD CLIP MODEL ─────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32-quickgelu", pretrained="openai"
)
model.to(device).eval()

def get_image_embedding(path: str) -> list[float]:
    img = Image.open(path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(tensor)
    return emb.cpu().numpy().flatten().tolist()


# ─── OPEN CHROMADB ───────────────────────────────────────────────────────────
client     = chromadb.PersistentClient(path="chroma")
collection = client.get_or_create_collection(
    name="multimodal_db_all",
    embedding_function=None
)
print(f"🔍 ChromaDB contains {collection.count()} entries\n")


# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
for side in ("L","R"):
    side_dir = os.path.join(REPORT_DIR, side)
    for fname in os.listdir(side_dir):
        # only process files like "12345_L.json" or "67890_R.json"
        if not fname.endswith(f"_{side}.json"):
            continue

        case_id = fname[:-len(f"_{side}.json")]
        json_path = os.path.join(side_dir, fname)

        # build image paths
        cc_path  = os.path.join(CC_DIRS[side],  f"{case_id}.png")
        mlo_path = os.path.join(MLO_DIRS[side], f"{case_id}.png")
        if not (os.path.exists(cc_path) and os.path.exists(mlo_path)):
            print(f"⚠️ Skipping {case_id}_{side}: missing image")
            continue

        # compute embeddings
        emb_cc  = get_image_embedding(cc_path)
        emb_mlo = get_image_embedding(mlo_path)
        query_emb = emb_cc + emb_mlo  # 1024-d

        # query the 3 nearest neighbors (excluding self)
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=4,  # fetch 4 so we can drop the first (it’s usually itself)
            include=["distances","uris","metadatas"]
        )
        ids       = results["ids"][0]
        dists     = results["distances"][0]
        uris      = results["uris"][0]
        metas     = results["metadatas"][0]

        # print(f"🔎 Neighbors for {side}_{case_id}:")
        # for rank, (nid, dist, uri, meta) in enumerate(zip(ids, dists, uris, metas), start=1):
        #     # skip the first match if it’s exactly our own ID
        #     if rank == 1 and nid == f"{side}_{case_id}":
        #         continue
        #     print(f" {rank-1}. ID={nid}, Dist={dist:.4f}")
        #     print(f"    Paths: {uri}")
        #     print(f"    Meta : {meta}")
        #     if rank-1 == 3:
        #         break
        # print()
        
        print(f"🔎 Neighbors for {side}_{case_id} as JSON:\n")
        example_idx = 1
        for nid, dist, uri, meta in zip(ids, dists, uris, metas):
            # skip itself
            if nid == f"{side}_{case_id}":
                continue

            out = {
                "IMG_ID_CC": uri.get("cc"),
                "IMG_ID_MLO": uri.get("mlo"),
                "Breast_Composition": meta.get("Breast_Composition"),
                "BIRADS": meta.get("BIRADS"),
                "Findings": meta.get("Findings")
            }

            print(f"Example {example_idx}:")
            print(json.dumps(out, ensure_ascii=False, indent=4))
            print()  # blank line between examples

            example_idx += 1
            if example_idx > 3:
                break
