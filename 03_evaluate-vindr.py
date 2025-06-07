import os
import re
import json
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from bert_score import score as bert_score
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import ast

# --- CONFIGURATION ---
gt_dir        = "/mnt/data1/Nafiz/MammoGen-RAG/vindr/ground_truth_reports"
pred_root     = "/mnt/data1/Nafiz/MammoGen-RAG/evaluated-vindr"
output_dir    = "/mnt/data1/Nafiz/MammoGen-RAG/results-vindr"
# ----------------------

def normalize_text(s: str) -> str:
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
            return " ".join(obj)
    except Exception:
        pass
    return s

def calculate_birads_metrics(gt_dir, pred_dir, file_list):
    digit_re = re.compile(r"(\d)")
    y_true, y_pred = [], []

    for fn in file_list:
        gt_path = os.path.join(gt_dir, fn)
        pr_path = os.path.join(pred_dir, fn)

        # load both JSONs once
        gt = json.load(open(gt_path))
        pr = json.load(open(pr_path))

        # extract and validate both BIRADS values
        s_gt = gt.get("BIRADS", "").strip()
        s_pr = pr.get("BIRADS", "").strip()

        m_gt = digit_re.search(s_gt)
        m_pr = digit_re.search(s_pr)

        if not m_gt or not m_pr:
            # print a warning for whichever is missing
            if not m_gt:
                print(f"Warning: no digit in GT BIRADS for {gt_path} (value: '{s_gt}')")
            if not m_pr:
                print(f"Warning: no digit in PR BIRADS for {pr_path} (value: '{s_pr}')")
            # skip this file entirely
            continue

        # append the extracted digits
        y_true.append(int(m_gt.group(1)))
        y_pred.append(int(m_pr.group(1)))

    if not y_true:
        raise RuntimeError("No valid BIRADS entries found to compute metrics.")

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    return p, r, f1



def calculate_composition_metrics(gt_dir, pred_dir, file_list):
    comp_re = re.compile(r"\b([ABCD])\b")
    y_true, y_pred = [], []

    for fn in file_list:
        gt_path = os.path.join(gt_dir, fn)
        pr_path = os.path.join(pred_dir, fn)
        gt = json.load(open(gt_path))
        pr = json.load(open(pr_path))

        gt_s = gt.get("BREAST-COMPOSITION", "").strip()
        pr_s = pr.get("BREAST-COMPOSITION", "").strip()

        if not gt_s or not pr_s:
            missing = []
            if not gt_s:
                missing.append("ground truth")
            if not pr_s:
                missing.append("prediction")
            missing_str = " & ".join(missing)
            print(f"Skipping {fn}: missing BREAST-COMPOSITION in {missing_str}.")
            continue

        m1 = comp_re.search(gt_s)
        m2 = comp_re.search(pr_s)
        if not m1 or not m2:
            print(f"Skipping {fn}: cannot extract comp A/B/C/D from '{gt_s}' vs '{pr_s}'")
            continue

        y_true.append(m1.group(1))
        y_pred.append(m2.group(1))

    # if nothing valid, warn and return zeros so caller can continue
    if not y_true:
        print("Warning: no valid BREAST-COMPOSITION entries to evaluate → returning 0.0 for all metrics.")
        return 0.0, 0.0, 0.0

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    return p, r, f1



def calculate_composition_bertscore(gt_dir, pred_dir, file_list):
    refs, cands = [], []
    for fn in file_list:
        gt_path = os.path.join(gt_dir, fn)
        pr_path = os.path.join(pred_dir, fn)
        gt = json.load(open(gt_path))
        pr = json.load(open(pr_path))

        ref = gt.get("BREAST-COMPOSITION", "").strip()
        cand = pr.get("BREAST-COMPOSITION", "").strip()

        if not ref or not cand:
            print(f"Skipping {fn} for BERTScore: missing text in "
                  f"{'ground truth' if not ref else ''}"
                  f"{' & ' if not ref and not cand else ''}"
                  f"{'prediction' if not cand else ''}.")
            continue

        refs.append(ref)
        cands.append(cand)

    if not refs:
        print("Warning: no valid texts for BERTScore → returning 0.0 for P, R, F1.")
        return 0.0, 0.0, 0.0

    P, R, F1 = bert_score(cands, refs, lang="en", rescale_with_baseline=False)
    return float(P.mean()), float(R.mean()), float(F1.mean())


def calculate_findings_metrics(gt_dir, pred_dir, file_list):
    refs, cands = [], []
    rouge_f = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    for fn in file_list:
        gt_path = os.path.join(gt_dir, fn)
        pr_path = os.path.join(pred_dir, fn)
        gt = json.load(open(gt_path))
        pr = json.load(open(pr_path))

        # pull raw values
        raw_ref  = gt.get("FINDINGS", "")
        raw_cand = pr.get("FINDINGS", "")

        # if it's a list, join it; otherwise normalize_text+strip
        if isinstance(raw_ref, list):
            ref = " ".join(raw_ref).strip()
        else:
            ref = normalize_text(raw_ref).strip()

        if isinstance(raw_cand, list):
            cand = " ".join(raw_cand).strip()
        else:
            cand = normalize_text(raw_cand).strip()

        if not ref or not cand:
            missing = []
            if not ref:  missing.append("ground truth")
            if not cand: missing.append("prediction")
            print(f"Skipping {fn} for Findings metrics: missing text in {' & '.join(missing)}.")
            continue

        refs.append([ref.split()])
        cands.append(cand.split())

    if not refs:
        print("Warning: no valid FINDINGS entries to evaluate → returning zeros for all metrics.")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # BLEU-4
    bleu = corpus_bleu(refs, cands)
    # METEOR (average)
    meteor = sum(meteor_score([r[0]], c) for r, c in zip(refs, cands)) / len(refs)
    # ROUGE-L F1 (average)
    rouge_l = sum(
        rouge_f.score(" ".join(r[0]), " ".join(c))['rougeL'].fmeasure
        for r, c in zip(refs, cands)
    ) / len(refs)
    # BERTScore
    P, R, F1 = bert_score(
        [" ".join(c) for c in cands],
        [" ".join(r[0]) for r in refs],
        lang="en",
        rescale_with_baseline=False
    )

    return (
        bleu,
        meteor,
        rouge_l,
        float(P.mean()),
        float(R.mean()),
        float(F1.mean())
    )



def main():
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    # loop through every sub-directory under evaluated-vindr
    for base in sorted(os.listdir(pred_root)):
        pred_dir = os.path.join(pred_root, base)
        if not os.path.isdir(pred_dir):
            continue

        # find matching JSONs
        gt_files   = {f for f in os.listdir(gt_dir)   if f.endswith(".json")}
        pred_files = {f for f in os.listdir(pred_dir) if f.endswith(".json")}
        common = sorted(gt_files & pred_files)
        if not common:
            print(f"→ no matching JSONs in '{base}', skipping.")
            continue

        # compute all metrics
        birads_p, birads_r, birads_f1 = calculate_birads_metrics(gt_dir, pred_dir, common)
        comp_p, comp_r, comp_f1       = calculate_composition_metrics(gt_dir, pred_dir, common)
        bs_p, bs_r, bs_f1             = calculate_composition_bertscore(gt_dir, pred_dir, common)
        bleu, meteor, rouge_l, f_bs_p, f_bs_r, f_bs_f1 = calculate_findings_metrics(gt_dir, pred_dir, common)

        # parse Model and Prompt Type out of the directory name
        if "_" in base:
            model, prompt = base.split("_", 1)
        else:
            model, prompt = base, ""

        rows.append({
            "Directory":                       base,
            "Model":                           model,
            "Prompt Type":                     prompt,
            "BIRADS_Precision":                birads_p,
            "BIRADS_Recall":                   birads_r,
            "BIRADS_F1-score":                 birads_f1,
            "Composition_Precision":           comp_p,
            "Composition_Recall":              comp_r,
            "Composition_F1-score":            comp_f1,
            "Composition_BERTScore_Precision": bs_p,
            "Composition_BERTScore_Recall":    bs_r,
            "Composition_BERTScore_F1-score":  bs_f1,
            "Findings_BLEU":                   bleu,
            "Findings_METEOR":                 meteor,
            "Findings_ROUGE-L":                rouge_l,
            "Findings_BERTScore_Precision":    f_bs_p,
            "Findings_BERTScore_Recall":       f_bs_r,
            "Findings_BERTScore_F1-score":     f_bs_f1
        })

    # write out a single CSV with one row per model/prompt
    df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, "results.csv")
    df.to_csv(out_path, index=False, float_format='%.4f')
    print(f"Saved aggregated evaluation to {out_path}")

if __name__ == "__main__":
    main()

