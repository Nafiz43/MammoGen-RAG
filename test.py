import os
import json
from collections import OrderedDict

# ——— CONFIGURATION ———
json_dir = "/mnt/data1/Nafiz/MammoGen-RAG/vindr/ground_truth_reports"
# ——————————————

for fname in os.listdir(json_dir):
    if not fname.lower().endswith(".json"):
        continue

    path = os.path.join(json_dir, fname)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # determine image-ID key
    img_key = "IMG-ID" if "IMG-ID" in data else "Image"

    # build new dict: 1) img_key, 2) BREAST-COMPOSITION, 3) FINDINGS, then rest
    new_data = OrderedDict()
    new_data[img_key] = data.get(img_key)
    new_data["BREAST-COMPOSITION"] = data.get("BREAST-COMPOSITION")

    # rename "Findings" to "FINDINGS" and place 3rd
    findings_val = data.get("Findings")
    new_data["FINDINGS"] = findings_val

    # add remaining keys, skipping those already handled
    for k, v in data.items():
        if k in (img_key, "BREAST-COMPOSITION", "Findings"):
            continue
        new_data[k] = v

    # overwrite file in place
    with open(path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

    print(f"Updated {fname}")
