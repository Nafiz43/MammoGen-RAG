# import chromadb

# # 1) Connect and open your collection
# client     = chromadb.PersistentClient(path="chroma")
# collection = client.get_or_create_collection(
#     name="multimodal_db_all",
#     embedding_function=None
# )

# print(f"🔍 Starting with {collection.count()} entries in ChromaDB")

# # 2) Peek at the first 10 entries
# peeked = collection.peek(10)
# print("Returned keys:", list(peeked.keys()))

# ids         = peeked.get("ids", [])
# embeddings  = peeked.get("embeddings", [])
# documents   = peeked.get("documents")
# uris        = peeked.get("uris", [])
# metadatas   = peeked.get("metadatas", [])
# distances   = peeked.get("distances", [])

# # If documents is None (some versions return None), fall back to uris
# if documents is None:
#     documents = uris

# n = len(ids)
# print(f"\nPrinting up to {n} entries:\n")

# for i in range(n):
#     print(f"Entry #{i+1}")
#     print(" ID           :", ids[i])

#     # file‐paths
#     if i < len(documents) and documents[i] is not None:
#         print(" Paths        :", documents[i])
#     else:
#         print(" Paths        : (none)")

#     # metadata
#     if i < len(metadatas) and metadatas[i] is not None:
#         print(" Metadata     :", metadatas[i])
#     else:
#         print(" Metadata     : (none)")

#     # embedding length
#     if i < len(embeddings):
#         print(" Embedding dims:", len(embeddings[i]))
#     else:
#         print(" Embedding dims: (none)")

#     # distance (if you stored/query it)
#     if i < len(distances):
#         print(" Distance     :", distances[i])
#     print("-" * 40)


# import chromadb

# # 1) connect to your ChromaDB
# client     = chromadb.PersistentClient(path="chroma")
# collection = client.get_or_create_collection(
#     name="multimodal_db_all",
#     embedding_function=None
# )

# # 2) figure out how many entries you have
# total = collection.count()
# print(f"🔢 Total entries in ChromaDB: {total}")

# # 3) peek all IDs, then slice the last 10
# all_ids = collection.peek(total)["ids"]
# last_ids = all_ids[-10:]

# # 4) fetch those 10 entries with their URIs, metadata, embeddings
# records = collection.get(
#     ids=last_ids,
#     include=["uris", "metadatas", "embeddings"]
# )

# print("\n📋 Last 10 entries:\n")
# for idx, case_id in enumerate(last_ids, start=1):
#     uri  = records["uris"][idx-1]        # {'cc':…, 'mlo':…}
#     meta = records["metadatas"][idx-1]   # your clinical labels
#     emb  = records["embeddings"][idx-1]  # 1024-dim vector

#     print(f"Entry #{total-10+idx} (ID={case_id}):")
#     print("  CC path :", uri.get("cc"))
#     print("  MLO path:", uri.get("mlo"))
#     print("  Metadata:", meta)
#     print("  Embedding dims:", len(emb))
#     print("-" * 40)

import shutil

shutil.rmtree("chroma")    # Deletes the entire directory and everything in it
print("🎯 ChromaDB folder ‘chroma’ has been removed.")


# import chromadb

# # 1) Connect to your collection
# client     = chromadb.PersistentClient(path="chroma")
# collection = client.get_or_create_collection(
#     name="multimodal_db_all",
#     embedding_function=None
# )

# def get_top_k_neighbors(input_id: str, k: int = 3):
#     # — Step 1: fetch your input entry’s embedding —
#     rec = collection.get(
#         ids=[input_id],
#         include=["embeddings"]
#     )
#     embs = rec.get("embeddings", [])
#     if len(embs) == 0:
#         raise ValueError(f"No embedding found for ID {input_id}")
#     query_emb = embs[0]    # first (and only) embedding

#     # — Step 2: run a nearest‐neighbor search on that embedding —
#     results = collection.query(
#         query_embeddings=[query_emb],
#         n_results=k+1,                # +1 to include itself
#         include=["ids","distances","uris","metadatas"]
#     )

#     all_ids       = results["ids"][0]
#     all_distances = results["distances"][0]
#     all_uris      = results["uris"][0]
#     all_meta      = results["metadatas"][0]

#     # drop the first result (it’s the query itself)
#     neighbors = []
#     for nid, dist, uri, meta in zip(all_ids[1:], all_distances[1:], all_uris[1:], all_meta[1:]):
#         neighbors.append({
#             "ID":       nid,
#             "Distance": dist,
#             "Paths":    uri,
#             "Metadata": meta
#         })

#     return neighbors


# # Example usage:
# input_entry = "L_6702d1c138b7b8de229c0377ff13eb92"
# top3 = get_top_k_neighbors(input_entry, k=3)
# for i, nbr in enumerate(top3, start=1):
#     print(f"Neighbor #{i}")
#     print(" ID       :", nbr["ID"])
#     print(" Distance :", nbr["Distance"])
#     print(" Paths    :", nbr["Paths"])
#     print(" Metadata :", nbr["Metadata"])
#     print("-" * 40)


# import torch

# print("CUDA available:", torch.cuda.is_available())
# print("GPU count:   ", torch.cuda.device_count())
# if torch.cuda.is_available():
#     print("Current device:", torch.cuda.get_device_name(0))

# #!/usr/bin/env python3
# import sys
# from ollama import Client

# def describe_image(image_path: str, model_name: str = "llava:34b") -> str:
#     """
#     Sends the image at image_path to the Ollama vision model and returns its description.
#     """
#     client = Client()  # assumes Ollama daemon is running locally
#     message = {
#         "role":    "user",
#         "content": "Please describe what you see in this image in one or two sentences. Describe if you see any abnormalities in breast tissue. Also what is the tissue density? And what would you consider the BIRADS?",
#         "images":  [image_path]
#     }
#     response = client.chat(
#         model=model_name,
#         messages=[message]
#     )
#     return response["message"]["content"]

# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print(f"Usage: {sys.argv[0]} /absolute/path/to/image.png")
#         sys.exit(1)

#     img_path = sys.argv[1]
#     description = describe_image(img_path)
#     print("Model description:")
#     print(description)
