#!/usr/bin/env python3
import os
import re
import json

from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

# ─── Configuration (hard-coded) ───────────────────────────────────────────────
GROUND_TRUTH_DIR = "/mnt/data1/Nafiz/MammoGen-RAG/vindr/GROUND_TRUTH_REPORTS"
EVALUATED_DIR    = "/mnt/data1/Nafiz/MammoGen-RAG/evaluated-vindr"
RESULTS_DIR      = "/mnt/data1/Nafiz/MammoGen-RAG/results-vindr"
# ────────────────────────────────────────────────────────────────────────────────

def parse_birads(s: str) -> int:
    """
    Extract the first numeric substring (0–6) from a BIRADS string.
    E.g. "BI-RADS 4A" → 4
    """
    matches = re.findall(r'\d+', s or "")
    if not matches:
        raise ValueError(f"No digits found in BIRADS field: {s!r}")
    return int(matches[0])

def main():
    for prompt_style in sorted(os.listdir(EVALUATED_DIR)):
        prompt_path = os.path.join(EVALUATED_DIR, prompt_style)
        if not os.path.isdir(prompt_path):
            continue

        # Aggregate predictions & truths per model_name (stripping off _L/_R)
        aggregated = {}  # model_name -> {'y_true': [], 'y_pred': []}

        for model_run in sorted(os.listdir(prompt_path)):
            run_path = os.path.join(prompt_path, model_run)
            if not os.path.isdir(run_path):
                continue

            # Expect folder names like "llava:latest_L" or "otherModel_R"
            parts = model_run.rsplit("_", 1)
            if len(parts) != 2 or parts[1] not in ("L", "R"):
                print(f"[!] Skipping unexpected folder name: {model_run}")
                continue

            model_name, view = parts
            agg = aggregated.setdefault(model_name, {"y_true": [], "y_pred": []})

            for fname in sorted(os.listdir(run_path)):
                if not fname.endswith(".json"):
                    continue

                case_id = fname[:-5]
                gen_path = os.path.join(run_path, fname)
                gt_path  = os.path.join(GROUND_TRUTH_DIR, view, f"{case_id}_{view}.json")

                if not os.path.isfile(gt_path):
                    print(f"    [!] Missing GT for {case_id} ({view})")
                    continue

                with open(gen_path, "r", encoding="utf-8") as f:
                    pred = json.load(f)
                with open(gt_path, "r", encoding="utf-8") as f:
                    gt   = json.load(f)

                try:
                    agg["y_pred"].append(parse_birads(pred.get("BIRADS", "")))
                    agg["y_true"].append(parse_birads(gt.get("BIRADS", "")))
                except ValueError as e:
                    print(f"    [!] Skipping {gen_path}: {e}")
                    continue

        # Compute and save one‐row CSV per model_name
        for model_name, data in aggregated.items():
            y_true = data["y_true"]
            y_pred = data["y_pred"]
            if not y_true:
                print(f"[!] No valid cases for model '{model_name}' under '{prompt_style}' – skipping.")
                continue

            labels = sorted(set(y_true + y_pred))
            precision = precision_score(y_true, y_pred, labels=labels,
                                        average="macro", zero_division=0)
            recall    = recall_score(   y_true, y_pred, labels=labels,
                                        average="macro", zero_division=0)
            f1        = f1_score(       y_true, y_pred, labels=labels,
                                        average="macro", zero_division=0)

            out_dir = os.path.join(RESULTS_DIR, prompt_style)
            os.makedirs(out_dir, exist_ok=True)
            out_csv = os.path.join(out_dir, f"{model_name}.csv")

            df = pd.DataFrame([{
                "Prompt Type":      prompt_style,
                "BIRADS_Precision": precision,
                "BIRADS_Recall":    recall,
                "BIRADS_F1-score":  f1
            }])
            df.to_csv(out_csv, index=False)
            print(f"→ Saved {out_csv}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

if __name__ == "__main__":
    main()
