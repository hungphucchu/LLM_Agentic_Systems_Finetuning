from transformers import AutoTokenizer

from src.utils.io_utils import read_jsonl, write_jsonl

import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bertscore_score
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"


def _task_completed_heuristic(pred_text: str, *, min_chars: int = 20) -> bool:
    t = (pred_text or "").strip()
    if not t or len(t) < min_chars:
        return False
    lowered = t.lower()
    failure_markers = [
        "i can't",
        "i cannot",
        "unable to",
        "as an ai",
        "i'm an ai",
        "sorry",
        "{}",
        "null",
        "[]",
    ]
    if any(m in lowered for m in failure_markers):
        return False
    return True


def main() -> None:
    stage2_ckpt = os.getenv("STAGE2_CKPT_LABEL", "ckpt2_stage2")
    checkpoints = os.getenv("ALPACA_EVAL_CKPTS", "ckpt0_base,ckpt1_stage1," + stage2_ckpt).split(",")

    tables_dir = Path("artifacts/tables")
    metrics_dir = Path("artifacts/metrics")
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False)

    # ROUGE: use F-measure across samples.
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    bertscore_model_type = os.getenv("BERTSCORE_MODEL_TYPE", "roberta-large")
    bertscore_lang = os.getenv("BERTSCORE_LANG", "en")
    bertscore_batch_size = int(os.getenv("BERTSCORE_BATCH_SIZE", "16"))

    min_chars = int(os.getenv("TASK_COMPLETION_MIN_CHARS", "20"))

    # Compatibility fix:
    # The bert_score package expects `tokenizer.build_inputs_with_special_tokens`.
    # In your cluster's older Transformers version, some tokenizers may miss this
    # method, causing crashes like:
    #   "RobertaTokenizer has no attribute build_inputs_with_special_tokens"
    if not hasattr(PreTrainedTokenizerBase, "build_inputs_with_special_tokens"):
        def _build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
            cls = getattr(self, "cls_token_id", None) or getattr(self, "bos_token_id", None)
            sep = getattr(self, "sep_token_id", None) or getattr(self, "eos_token_id", None)

            token_ids_0 = list(token_ids_0)
            if token_ids_1 is None:
                if cls is None or sep is None:
                    return token_ids_0
                return [cls] + token_ids_0 + [sep]

            token_ids_1 = list(token_ids_1)
            if cls is None or sep is None:
                return token_ids_0 + token_ids_1
            return [cls] + token_ids_0 + [sep] + token_ids_1 + [sep]

        PreTrainedTokenizerBase.build_inputs_with_special_tokens = _build_inputs_with_special_tokens  # type: ignore[attr-defined]

    rows_summary: List[Dict[str, Any]] = []

    for ckpt in checkpoints:
        rows = read_jsonl(f"artifacts/predictions/{ckpt}_alpaca_eval_outputs.jsonl")
        if not rows:
            print(f"[alpaca-auto] No rows found for {ckpt}; skipping.")
            continue

        preds = [r.get("prediction", "") or "" for r in rows]
        refs = [r.get("reference", "") or "" for r in rows]

        token_lens: List[int] = []
        completed = 0
        for p in preds:
            token_lens.append(len(tokenizer.encode(p)))
            completed += int(_task_completed_heuristic(p, min_chars=min_chars))

        avg_output_tokens = float(np.mean(token_lens)) if token_lens else 0.0
        task_completion_rate = completed / len(rows) if rows else 0.0

        rouge1_f: List[float] = []
        rouge2_f: List[float] = []
        rougeL_f: List[float] = []
        for pred, ref in zip(preds, refs):
            # rouge_scorer.score(target, prediction)
            scores = scorer.score(ref, pred)
            rouge1_f.append(scores["rouge1"].fmeasure)
            rouge2_f.append(scores["rouge2"].fmeasure)
            rougeL_f.append(scores["rougeL"].fmeasure)

        rouge1_avg = float(np.mean(rouge1_f)) if rouge1_f else 0.0
        rouge2_avg = float(np.mean(rouge2_f)) if rouge2_f else 0.0
        rougeL_avg = float(np.mean(rougeL_f)) if rougeL_f else 0.0

        # Compute BERTScore directly (avoid HF `evaluate`, which can import
        # `transformers.pipelines` -> `torchvision` and fail with torch/vision
        # mismatches on the cluster).
        _P, _R, F1 = bertscore_score(
            cands=preds,
            refs=refs,
            lang=bertscore_lang,
            model_type=bertscore_model_type,
            batch_size=bertscore_batch_size,
            # Use CPU by default; the cluster driver may be too old for the
            # installed torch CUDA build, which can crash metric computation.
            device=os.getenv("BERTSCORE_DEVICE", "cpu"),
            verbose=False,
        )
        bert_f1_avg = float(F1.mean().item()) if F1 is not None else 0.0

        metrics = {
            "checkpoint": ckpt,
            "samples": len(rows),
            "avg_output_tokens": avg_output_tokens,
            "task_completion_rate": task_completion_rate,
            "rouge1_f1": rouge1_avg,
            "rouge2_f1": rouge2_avg,
            "rougeL_f1": rougeL_avg,
            "bertscore_f1_avg": bert_f1_avg,
        }
        write_jsonl(str(metrics_dir / f"alpaca_auto_metrics_{ckpt}.json"), [metrics])

        print(
            f"[alpaca-auto] {ckpt}: ROUGE-L={rougeL_avg:.4f} "
            f"BERTScoreF1={bert_f1_avg:.4f} completion={task_completion_rate:.3f} "
            f"tokens={avg_output_tokens:.1f} samples={len(rows)}"
        )

        rows_summary.append(
            {
                "checkpoint": ckpt,
                "samples": len(rows),
                "avg_output_tokens": avg_output_tokens,
                "task_completion_rate": task_completion_rate,
                "rouge1_f1": rouge1_avg,
                "rouge2_f1": rouge2_avg,
                "rougeL_f1": rougeL_avg,
                "bertscore_f1_avg": bert_f1_avg,
            }
        )

    import csv

    t_path = tables_dir / "alpaca_metrics_by_checkpoint.csv"
    with t_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "checkpoint",
                "samples",
                "avg_output_tokens",
                "task_completion_rate",
                "rouge1_f1",
                "rouge2_f1",
                "rougeL_f1",
                "bertscore_f1_avg",
            ],
        )
        w.writeheader()
        for row in rows_summary:
            w.writerow(row)

    print(f"[alpaca-auto] wrote {t_path}")


if __name__ == "__main__":
    main()
