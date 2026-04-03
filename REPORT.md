# Assignment 3 Report: Sequential Instruction Tuning & Structured Output Imitation

*Submission format: **Option 1** — this file is the web-readable report (Markdown in the GitHub repo). For grading, open this file on GitHub (`…/blob/main/REPORT.md`) or clone the repository.*

## 1. Summary

This project fine-tunes **Microsoft Phi-3.5-mini-instruct** in two stages: **Stage 1** on Alpaca-style instruction data (QLoRA), then **Stage 2** on teacher-generated JSON imitation data (QLoRA continuation). We evaluate three checkpoints—base model (**ckpt0**), after Stage 1 (**ckpt1**), after Stage 2 (**ckpt2**)—using automatic metrics (ROUGE-L, BERTScore F1, JSON validity/schema/exact match) and an **LLM-as-judge** with pairwise Alpaca comparisons and dimension-scored JSON evaluation. We analyze **catastrophic forgetting** between Stage 1 and Stage 2 on Alpaca-style tasks and report an **ablation** on **Stage 2 training duration** (1 vs 2 epochs on the JSON stage, with separate adapters and prediction runs).

---

## 2. Methodology

### 2.1 Models and training

| Item | Setting |
|------|---------|
| Student | `microsoft/Phi-3.5-mini-instruct` |
| Adaptation | QLoRA (4-bit NF4, LoRA on attention/MLP projections) |
| Stage 1 data | Processed Alpaca-style JSONL (`data/processed/alpaca_train.jsonl`) |
| Stage 2 data | Teacher-generated JSON imitation set (built from prompt pool + teacher API) |
| Stage 1 (config) | LR `2e-5`, 2 epochs, batch 2 × grad accum 8 (see `config/config.yaml`) |
| Stage 2 (config default) | LR `1e-5`, 2 epochs, same batch settings |

**Student model choice (course §2.1):** We use **Phi-3.5-mini-instruct** as the recommended default: strong instruction-following for its size class, well-documented chat template, and practical memory footprint on **UTSA V100** nodes with **4-bit QLoRA** and rank-16 LoRA. Alternatives such as Llama 3.2 3B or Qwen2.5 3B are viable but were not necessary given Phi-3.5’s balance of quality and HPC fit.

**Ablation:** A second Stage 2 adapter was trained with **`STAGE2_EPOCHS=2`** into `artifacts/checkpoints/stage2_json_adapter_ep2_ablate` (see Slurm `STAGE2_OUT_DIR`). The **baseline** table (`ckpt2_stage2`) uses the **primary** Stage 2 adapter trained with **1** Stage 2 epoch; the ablation table uses the **2-epoch** adapter for `ckpt2_stage2_ablate_ep2` only.

**Teacher and judge (API):** Teacher generations use `TEACHER_MODEL` (OpenAI-compatible API; course default comparable to Llama 3.1 70B Instruct). Evaluation judges use `JUDGE_MODEL` and `BASE_URL` / `API_KEY` from `.env`—document the **exact** names you ran in your HPC `.env` in an appendix footnote for reproducibility.

### 2.2 Inference

- Batched generation with left padding, `max_new_tokens` controlled via env (e.g. `GEN_MAX_NEW_TOKENS`).
- Subsets for cost control: e.g. `INFERENCE_MAX_ALPACA`, `INFERENCE_MAX_JSON` (200 / 100 in the reported runs).
- Predictions saved under `artifacts/predictions/` as `{checkpoint_label}_{alpaca|json}_eval_outputs.jsonl`.

### 2.3 Evaluation

**Automatic (Alpaca):** ROUGE-1/2/L (F1), BERTScore F1 (RoBERTa-large), heuristic task-completion rate, mean output length.

**Automatic (JSON):** JSON validity, schema compliance vs reference shape, exact match, field-level F1 on extraction-style items (subset count reported in metrics), error taxonomy counts.

**Judge:** OpenAI-compatible API (`BASE_URL`, `API_KEY`, `JUDGE_MODEL` in `.env`). Alpaca: pairwise Self-Instruct–style preference with optional **response-order randomization** (`JUDGE_RANDOMIZE_ORDER`). JSON: multi-dimensional scores. Judge outputs parsed with tolerant JSON extraction (`src/utils/json_schema_utils.py`).

**Aggregation:** `src/evaluation/aggregate_results.py` merges auto metrics and judge-derived win rates into `artifacts/tables/three_checkpoint_comparison*.csv`.

---

## 3. Main Results (Baseline Checkpoint Labels)

Source: `artifacts/tables/three_checkpoint_comparison.csv`.

| Checkpoint | Judge win rate† | ROUGE-L F1 | BERTScore F1 | JSON validity | Schema compliance | Exact match |
|------------|-----------------|------------|--------------|---------------|-------------------|-------------|
| ckpt0_base | 0.636 | 0.1230 | 0.8323 | 1.00 | 0.35 | 0.35 |
| ckpt1_stage1 | 0.417 | 0.1160 | 0.8365 | 1.00 | 0.18 | 0.18 |
| ckpt2_stage2 | 0.500 | 0.1146 | 0.8365 | 1.00 | 0.17 | 0.17 |

†**Judge win rate** is the script’s aggregate win rate from pairwise comparisons across configured checkpoint pairs (non-tie wins / non-tie decisions); see `aggregate_results.py` for the exact aggregation.

**Observations:**

- **Stage 1** improves BERTScore vs base on Alpaca references while **judge win rate vs base drops**, consistent with style shift and judge preference for the pretrained model on some pairs.
- **Stage 2 (JSON imitation)** keeps **valid JSON** at **100%** on this eval slice but **schema compliance and exact match fall** vs base, reflecting specialization toward teacher JSON formats that may not match every reference example byte-for-byte.
- **ROUGE-L** drifts slightly downward after fine-tuning, which is common when the model’s phrasing diverges from Alpaca references even if quality is acceptable.

### 3.1 Alpaca judge dimensions (baseline)

Source: `artifacts/tables/alpaca_judge_summary_by_checkpoint.csv` (averages over judge calls).

- **ckpt0_base** shows the highest average **clarity** and **hallucination_risk** score (latter as defined in the judge rubric).
- **ckpt1_stage1** scores lowest on several dimensions on average; **ckpt2_stage2** partially recovers vs ckpt1 on some dimensions while judge win rate sits between ckpt0 and ckpt1.

---

## 4. Ablation: Stage 2 Training (2 Epochs)

We re-ran inference with labels `ckpt0_base_ablate_ep2`, `ckpt1_stage1_ablate_ep2`, `ckpt2_stage2_ablate_ep2` (same base and Stage 1 adapter, **Stage 2 adapter trained for 2 epochs**), then re-evaluated and aggregated with `RUN_TAG=ablate_ep2`.

Source: `artifacts/tables/three_checkpoint_comparison_ablate_ep2.csv`.

| Checkpoint | Judge win rate | ROUGE-L F1 | BERTScore F1 | JSON validity | Schema compliance | Exact match |
|------------|----------------|------------|--------------|---------------|-------------------|-------------|
| ckpt0_base_ablate_ep2 | 0.536 | 0.1230 | 0.8323 | 1.00 | 0.35 | 0.35 |
| ckpt1_stage1_ablate_ep2 | 0.421 | 0.1160 | 0.8365 | 1.00 | 0.18 | 0.18 |
| ckpt2_stage2_ablate_ep2 | 0.535 | 0.1149 | 0.8366 | 1.00 | 0.17 | 0.17 |

**Compared to baseline `ckpt2_stage2`:**

- **ROUGE-L:** 0.1146 → **0.1149** (small increase).
- **BERTScore F1:** 0.8365 → **0.8366** (negligible).
- **Schema / exact match:** unchanged at **0.17** on this eval (JSON task remains difficult; extra Stage 2 epoch did not fix reference alignment in this setup).
- **Judge win rate (aggregate):** context-dependent across all pairs; the ablation row for **ckpt2** is **0.535** vs **0.500** baseline—interpret cautiously because judge win rates are **not** solely “quality of ckpt2” but aggregation over multiple pairwise files.

**JSON auto detail (last run, ablation labels):** `artifacts/tables/json_metrics_by_checkpoint.csv` reports extraction F1 on 20 extraction-style prompts: **0.067** (base), **0** (stage1/stage2) in the synced snapshot—use this as a diagnostic, not the sole JSON quality measure.

---

## 5. Catastrophic Forgetting (Stage 1 vs Stage 2)

**Pairwise judge file (HPC / local):** `artifacts/judge/alpaca_ckpt1_stage1_vs_ckpt2_stage2.jsonl`.

**Machine-readable summary (synced):** `artifacts/metrics/forgetting_alpaca_ckpt1_stage1_vs_ckpt2_stage2.json` (also `alpaca_forgetting_summary.csv` as a short KV view).

| Metric | Value |
|--------|-------|
| Total pairs | 200 |
| ckpt1 (Stage 1) wins | 38 |
| ckpt2 (Stage 2) wins | 52 |
| Ties | 110 |
| ckpt1 win rate | 0.19 |
| ckpt2 win rate | 0.26 |
| Tie rate | 0.55 |
| Δ judge win rate (ckpt2 − ckpt1) | **+0.07** |
| ROUGE-L F1 ckpt1 → ckpt2 | 0.1160 → 0.1146 (**Δ ≈ −0.00143**) |
| BERTScore F1 ckpt1 → ckpt2 | 0.8365 → 0.8365 (**Δ ≈ −1.5e−5**) |

**Per-category judge outcomes** (heuristic buckets from instruction text; see JSON for full counts): open-ended is the largest bucket; Stage 2 **wins more often** there in pairwise comparisons, while rewriting shows mostly ties with few Stage 1 wins. Full `category_breakdown`, **representative_regression_example**, and **representative_improvement_example** (with truncated predictions) are in the JSON file above—paste one of each into the blog if the rubric asks for exemplars.

**Interpretation:** Automatic Alpaca metrics move **slightly down** from Stage 1 to Stage 2 on ROUGE-L; BERTScore is **flat**. The judge gives Stage 2 a **modest edge** in win rate on this slice, alongside many ties—so this is **not** a clean “catastrophic collapse” story, but **task-dependent** tradeoffs (see §6).

---

## 6. Analysis (post-training concepts)

**Sequential fine-tuning:** Stage 1 aligns the student to broad instruction-following; Stage 2 adds a **distribution shift** toward teacher-style JSON. Metrics show JSON validity remains high while **schema compliance and exact match** are limited—consistent with specialization that does not perfectly match every held-out reference.

**Imitation learning:** Stage 2 targets are **discrete teacher strings**, not logits (black-box / synthetic supervision). Gains in formatting discipline can coexist with **Alpaca regression** on n-gram overlap (ROUGE-L) or judge preferences, as seen from ckpt1 → ckpt2.

**Catastrophic forgetting:** Pairwise judge results are **tie-heavy** (55% ties); Stage 2 **wins slightly more** head-to-head (52 vs 38) while ROUGE-L **ticks down**—so we see **mixed signals**, not a one-sided collapse of Alpaca ability. The ablation (extra Stage 2 epoch) moves automatic metrics only slightly, suggesting **diminishing returns** on this setup.

---

## 7. Limitations

- **Judge variance and bias:** position bias mitigated by randomization, but the judge model and API can still be inconsistent or tie-heavy.
- **Evaluation subsets** for cost (e.g. 200 Alpaca / 100 JSON prompts): confidence intervals not computed; numbers are point estimates.
- **Schema/exact match** are strict vs reference JSON; valid but differently shaped JSON is penalized.
- **Environment drift:** PyTorch/CUDA and `transformers` versions on HPC can affect reproducibility; the repo documents workarounds (e.g. BERTScore on CPU, tokenizer compatibility patch).
- **Config vs Slurm:** Effective epochs and learning rates follow **environment variables and Slurm exports** when set; always log `STAGE2_EPOCHS` and adapter paths in the appendix for exact reproducibility.

---

## 8. Reproducibility (HPC-oriented)

Project root: `LLM_Agentic_Systems_Finetuning`. Typical flow:

```bash
module load anaconda3  # or your cluster module
cd /work/<user>/LLM_Agentic_Systems_Finetuning
source .venv/bin/activate  # if used
set -a && source .env && set +a
export PYTHONPATH="$PWD"
# Optional: bash hpc/install_torch_cu121.sh for driver-matched torch
```

**Training:** `sbatch hpc/stage1_train.slurm`, `sbatch hpc/stage2_train.slurm` (ablation: `--export=ALL,STAGE2_EPOCHS=2,STAGE2_OUT_DIR=artifacts/checkpoints/stage2_json_adapter_ep2_ablate,...`).

**Inference:** `python3 src/inference/generate_checkpoint_outputs.py` with `STAGE2_ADAPTER_PATH`, `CKPT*_LABEL`, and limits as needed.

**Evaluation (ablation-aligned example):**

```bash
export CKPT0_LABEL=ckpt0_base_ablate_ep2
export CKPT1_LABEL=ckpt1_stage1_ablate_ep2
export STAGE2_CKPT_LABEL=ckpt2_stage2_ablate_ep2
export ALPACA_EVAL_CKPTS="${CKPT0_LABEL},${CKPT1_LABEL},${STAGE2_CKPT_LABEL}"
export JSON_EVAL_CKPTS="${CKPT0_LABEL},${CKPT1_LABEL},${STAGE2_CKPT_LABEL}"
export JSON_JUDGE_CKPTS="${CKPT0_LABEL},${CKPT1_LABEL},${STAGE2_CKPT_LABEL}"

python3 src/evaluation/eval_alpaca_auto.py
python3 src/evaluation/eval_json_auto.py
python3 src/evaluation/eval_alpaca_judge.py
python3 src/evaluation/eval_json_judge.py
python3 src/evaluation/forgetting_analysis.py
RUN_TAG=ablate_ep2 python3 src/evaluation/aggregate_results.py
```

**Important:** For `*_ablate_ep2.csv` tables, `RUN_TAG` only changes filenames; **`CKPT0_LABEL`, `CKPT1_LABEL`, `STAGE2_CKPT_LABEL` must match** the prediction file prefixes so metrics and judge stats align.

Full layout and local/HPC notes: `README.md`.

---

## 9. Appendix: Prompt assets (full templates)

Course §5 / §6 require editable templates. **Authoritative text** is in the repo and is **loaded at runtime** (not duplicated for editing in Python):

| File | Role |
|------|------|
| `prompts/teacher_json_generation.md` | Teacher imitation-learning user prompt (`__JSON_EXAMPLE__`, `__INSTRUCTION__`, `__INPUT__`) |
| `prompts/judge_system_message.md` | Shared system message for JSON-only judge replies |
| `prompts/judge_pairwise_eval.md` | Alpaca pairwise user prompt (`__RESPONSE_A__`, `__CHECKPOINT_A__`, …) |
| `prompts/judge_json_eval.md` | Per-example JSON judge user prompt |

Override paths with env vars: `TEACHER_PROMPT_TEMPLATE`, `JUDGE_SYSTEM_PROMPT`, `JUDGE_PAIRWISE_PROMPT`, `JUDGE_JSON_PROMPT`.

**Prompt engineering (brief):** Early judge runs returned chain-of-thought or empty parses; we tightened the **system** message and user instructions to demand a **single JSON object** and no reasoning tags, and added tolerant parsing in `json_schema_utils.py`. Further iteration = edit the `.md` files above and rerun judge eval.

---

## 10. Artifacts Index (this submission)

Artifacts are grouped by topic under **`artifacts/`**: `predictions/`, `judge/`, `logs/`, `tables/`, `metrics/`, `archives/`.

| Path | Contents |
|------|----------|
| `artifacts/tables/three_checkpoint_comparison.csv` | Baseline three-checkpoint summary |
| `artifacts/tables/three_checkpoint_comparison_ablate_ep2.csv` | Ablation (2-epoch Stage 2) summary |
| `artifacts/tables/alpaca_judge_summary_by_checkpoint*.csv` | Per-dimension judge averages |
| `artifacts/tables/alpaca_forgetting_summary*.csv` | Stage1 vs Stage2 forgetting KV summary |
| `artifacts/tables/json_metrics_by_checkpoint.csv` | JSON auto metrics (last eval run’s checkpoint names) |
| `artifacts/tables/json_error_taxonomy_by_checkpoint.csv` | Error category counts |
| `artifacts/metrics/alpaca_auto_metrics_*.json` | Per-checkpoint Alpaca auto metrics (ROUGE, BERTScore, …) |
| `artifacts/metrics/json_auto_metrics_*.json` | Per-checkpoint JSON auto metrics |
| `artifacts/metrics/forgetting_alpaca_ckpt1_stage1_vs_ckpt2_stage2.json` | Forgetting: Δ metrics, categories, exemplars |

---

## 11. Conclusion

We built a **two-stage QLoRA** pipeline (Alpaca → teacher JSON imitation) on **UTSA HPC** and evaluated **three checkpoints** with **automatic metrics** and an **LLM judge**. Stage 2 keeps **valid JSON** high but **strict schema/exact match** and some **Alpaca-side** signals (e.g. ROUGE-L vs references) **shift**; pairwise forgetting analysis shows **many ties** and a **modest** Stage 2 edge in judge wins on our slice, not a sharp catastrophic collapse. An **extra Stage 2 epoch** (ablation) yields **small** automatic-metric changes—useful for discussing **diminishing returns** when pushing JSON specialization. Future work could loosen JSON references, enlarge judge power, or tune LR/epochs against a clearer retention target.

---

## 12. PDF rubric crosswalk

See **`ASSIGNMENT_COMPLIANCE.md`** for a line-by-line checklist against the assignment PDF (what is done in code, what is only on HPC, and what you must still submit on the course portal).

---

*Report generated to match artifacts synced on 2026-04-03; numeric rounding to ~3–4 decimals in tables.*
