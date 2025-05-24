import os
import pandas as pd
import json

def main():
    # ——— Paths ———
    BASE_DIR    = "/mnt/data1/raiyan/breast_cancer/datasets/vindr"
    IMAGES_DIR  = os.path.join(BASE_DIR, "images_png")
    BREAST_CSV  = os.path.join(BASE_DIR, "breast-level_annotations.csv")
    FINDING_CSV = os.path.join(BASE_DIR, "finding_annotations.csv")
    OUTPUT_DIR  = "/mnt/data1/Nafiz/MammoGen-RAG/vindr/ground_truth_reports"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ——— Load annotations ———
    breast_df  = pd.read_csv(BREAST_CSV, dtype=str).set_index("image_id")
    finding_df = pd.read_csv(FINDING_CSV, dtype=str)

    # ——— Counters for logging ———
    images_scanned   = 0
    reports_written  = 0

    for study_id in os.listdir(IMAGES_DIR):
        study_path = os.path.join(IMAGES_DIR, study_id)
        if not os.path.isdir(study_path):
            continue

        for img_name in os.listdir(study_path):
            if not img_name.lower().endswith(".png"):
                continue

            images_scanned += 1
            image_id   = os.path.splitext(img_name)[0]
            image_path = os.path.join(study_path, img_name)

            # — lookup breast-level info —
            if image_id not in breast_df.index:
                print(f"⚠️  {image_id} missing in breast-level CSV; skipping")
                continue
            binfo         = breast_df.loc[image_id]
            birads_last   = str(binfo["breast_birads"]).strip()[-1]
            density_last  = str(binfo["breast_density"]).strip()[-1]

            # — lookup finding_categories by BOTH study_id & image_id —
            findings = ""
            mask = (
                (finding_df["study_id"] == study_id) &
                (finding_df["image_id"] == image_id)
            )
            matched = finding_df.loc[mask, "finding_categories"]
            if not matched.empty:
                findings = ";".join(matched.astype(str).tolist())

            # — assemble record & write JSON —
            rec = {
                "Image":               image_path,
                "Breast Composition":  density_last,
                "Findings":            findings,
                "BIRADS":              birads_last
            }
            out_path = os.path.join(OUTPUT_DIR, f"{image_id}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=4)

            reports_written += 1

            # periodic progress update
            if images_scanned % 1000 == 0:
                print(f"▶️  Scanned {images_scanned} images, wrote {reports_written} reports so far...")

    # final summary
    print(f"\n✅ Done! Scanned {images_scanned} images; generated {reports_written} JSON reports.")
    print("→ All reports in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
