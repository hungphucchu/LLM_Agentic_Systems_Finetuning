# Report: Sequential Instruction Tuning & Structured Output Imitation

## 1. Methodology

### 1.1 Student model choice

We use **`microsoft/Phi-3.5-mini-instruct`** as the student. Rationale: strong instruction-following for its size, a documented chat template for optional chat-formatted inference, and a practical memory footprint on **UTSA V100** GPUs with **4-bit QLoRA** and rank-16 LoRA. Other allowed options (e.g. Llama 3.2 3B, Qwen2.5 3B) were not required given Phi-3.5’s balance of quality and HPC fit.

### 1.2 Alpaca data source

General instruction data come from the Hugging Face dataset **`tatsu-lab/alpaca`**. The preparation script (`src/data_prep/prepare_alpaca.py`) loads the train split, **shuffles with seed 42**, normalizes each row to **`instruction` / `input` / `output`**, drops rows without a non-empty instruction, and writes:

- **`data/processed/alpaca_train.jsonl`** — **95%** of rows, used only in Stage 1 training  
- **`data/processed/alpaca_eval.jsonl`** — **5%** held out, used only for evaluation (never in Stage 1 loss)

### 1.3 Imitation-learning pipeline for the JSON Instruct dataset

Structured-output supervision is built in four steps:

1. **Prompt pool** — `src/data_prep/build_json_prompts.py` constructs a diverse pool covering five task families: JSON extraction, schema-constrained generation, exact-label classification with JSON output, JSON repair, and tool-call argument generation. Each row includes `instruction`, `input`, `json_example` (target shape), and `task_type`. Output: **`data/processed/json_prompt_pool.jsonl`**.
2. **Teacher generation** — `src/data_prep/generate_teacher_json.py` calls a **teacher** LLM via an OpenAI-compatible API for each pool row, using the template in **`prompts/teacher_json_generation.md`**. Invalid JSON responses trigger retries (`TEACHER_MAX_INVALID_RETRIES`).
3. **Pairing** — Valid teacher outputs are serialized to the **`output`** field; each training row keeps the original **`instruction`** and **`input`**, plus **`task_type`**.
4. **Train / eval split** — After shuffling (seed from `SEED`, default 42), the first **`JSON_EVAL_SIZE`** rows (default **100**) become **`json_eval.jsonl`**; up to **`JSON_TRAIN_CAP`** rows (default **80**) become **`json_train_teacher.jsonl`** for Stage 2. The sets are **disjoint** by construction.

A final filter **`src/data_prep/validate_json_dataset.py`** keeps only rows whose **`output`** parses as valid JSON.

### 1.4 Teacher model setup

Teacher calls use **`TEACHER_MODEL`** (or **`UTSA_MODEL`** if the former is unset) with **`BASE_URL`** / **`API_KEY`** (or **`UTSA_BASE_URL`** / **`UTSA_API_KEY`**), loaded from **`.env`**. The service is **Llama 3.1 70B Instruct–class** in our deployment; exact endpoint and model id should match what was used on HPC for reproducibility.

### 1.5 Training design (Stage 1 and Stage 2)

| Item | Setting |
|------|---------|
| Adaptation | QLoRA (4-bit NF4, LoRA on attention and MLP projections; see `config/config.yaml` for target modules) |
| Stage 1 data | `data/processed/alpaca_train.jsonl` |
| Stage 2 data | `data/processed/json_train_teacher.jsonl` (continues from Stage 1 adapter) |
| Stage 1 | LR `2e-5`, 2 epochs, per-device batch 2, gradient accumulation 8 (`train_stage1_alpaca.py`) |
| Stage 2 | LR `1e-5`, 2 epochs by default; **baseline ckpt2** uses **1** Stage 2 epoch in the primary run; ablation uses **2** epochs into a separate adapter directory |

Training uses causal language modeling on a single string format (see **Appendix: Student formatting**). Checkpoints are saved under **`artifacts/checkpoints/`** (adapters on HPC; see `README.md`).

### 1.6 UTSA HPC setup

Jobs are submitted with Slurm: **`hpc/stage1_train.slurm`** (Stage 1 + Alpaca prep) and **`hpc/stage2_train.slurm`** (JSON prep, optional teacher generation, validation, Stage 2). Typical allocation: **one GPU** on the **`gpu1v100`** partition, with CPUs and wall time set in the `#SBATCH` headers. The project root is taken from **`SLURM_SUBMIT_DIR`** so paths like **`artifacts/checkpoints/`** resolve correctly. Environment variables and optional **`module load`** / venv steps are documented in **`README.md`** and **Section 2 (reproducibility note)** below.

### 1.7 Judge model choice

Alpaca and JSON judge evaluations call **`JUDGE_MODEL`** (fallback **`UTSA_MODEL`**) on the same OpenAI-compatible **`BASE_URL`** as the teacher, with **`API_KEY`**. Temperature is **0.0** for judge calls. Pairwise Alpaca evaluation optionally **randomizes which checkpoint appears as Response A vs B** (`JUDGE_RANDOMIZE_ORDER`).

### 1.8 Evaluation protocol

- **Checkpoints:** **ckpt0** = base weights from Hugging Face; **ckpt1** = after Stage 1 adapter (merged for inference); **ckpt2** = after Stage 2 adapter (merged).
- **Generations:** `src/inference/generate_checkpoint_outputs.py` writes **`artifacts/predictions/{label}_{alpaca|json}_eval_outputs.jsonl`**. Reported runs used subset limits **`INFERENCE_MAX_ALPACA` / `INFERENCE_MAX_JSON`** (e.g. 200 / 100) where noted.
- **Alpaca — automatic:** ROUGE-1/2/L (F1), BERTScore F1, task-completion heuristic, mean output length (`eval_alpaca_auto.py`).
- **Alpaca — judge:** Pairwise comparisons ckpt0–ckpt1, ckpt1–ckpt2, ckpt0–ckpt2 (`eval_alpaca_judge.py`) → **`artifacts/judge/alpaca_*.jsonl`**.
- **JSON — automatic:** Validity, schema compliance vs reference, exact match, field-level F1 on extraction subset, error taxonomy (`eval_json_auto.py`).
- **JSON — judge:** Per-example dimension scores (`eval_json_judge.py`).
- **Forgetting:** Stage 1 vs Stage 2 on Alpaca (`forgetting_analysis.py`).
- **Aggregation:** `aggregate_results.py` produces **`artifacts/tables/three_checkpoint_comparison*.csv`** and related CSVs.

### 1.9 Hyperparameters

Authoritative defaults and names for models, LoRA, epochs, batch sizes, and evaluation settings are in **`config/config.yaml`**. **Effective** Stage 2 epoch count and adapter output path can be overridden with environment variables (**`STAGE2_EPOCHS`**, **`STAGE2_OUT_DIR`**) and Slurm **`--export`**, as in the ablation run.

---

## 2. Experiments

### 2.1 Three-checkpoint comparison (primary table)

Source: **`artifacts/tables/three_checkpoint_comparison.csv`**.

| Checkpoint | Judge win rate† | ROUGE-L F1 | BERTScore F1 | JSON validity | Schema compliance | Exact match |
|------------|-----------------|------------|--------------|---------------|-------------------|-------------|
| ckpt0_base | 0.636 | 0.1230 | 0.8323 | 1.00 | 0.35 | 0.35 |
| ckpt1_stage1 | 0.417 | 0.1160 | 0.8365 | 1.00 | 0.18 | 0.18 |
| ckpt2_stage2 | 0.500 | 0.1146 | 0.8365 | 1.00 | 0.17 | 0.17 |

†**Judge win rate** = aggregate over configured pairwise files (non-tie wins / non-tie decisions); see **`src/evaluation/aggregate_results.py`**.

### 2.2 Alpaca evaluation results

**Automatic metrics** are reflected in the ROUGE-L and BERTScore columns above. **Judge dimension averages** (baseline) are summarized from **`artifacts/tables/alpaca_judge_summary_by_checkpoint.csv`**: **ckpt0_base** has the highest average **clarity** and **hallucination_risk** among the six dimensions; **ckpt1_stage1** is lowest on several dimensions on average; **ckpt2_stage2** partially recovers vs ckpt1 on some dimensions while aggregate win rate sits between ckpt0 and ckpt1.

**Reading the primary table:** Stage 1 raises BERTScore vs references while **aggregate judge win rate vs base falls**—consistent with stylistic shift and judge preference for the pretrained model on some pairs. **ROUGE-L** drifts down after fine-tuning, which often happens when wording diverges from reference answers.

### 2.3 JSON evaluation results

On the held-out JSON set, **valid JSON** stays at **100%** for all three checkpoints in this snapshot, while **schema compliance** and **exact match** are **lower after fine-tuning than at base** (0.35 → 0.18 → 0.17), reflecting strict matching against reference objects and teacher–student format variation. **Finer-grained JSON metrics** (per-checkpoint validity, schema, exact match, extraction F1, prompt counts) are in **`artifacts/tables/json_metrics_by_checkpoint.csv`**; **error categories** are in **`artifacts/tables/json_error_taxonomy_by_checkpoint.csv`**. Extraction F1 on the 20 extraction-style prompts (last ablation-aligned run cited previously): **0.067** (base), **0** (stage1/stage2)—a diagnostic only, not a substitute for validity/schema/judge signals.

### 2.4 Forgetting analysis (Stage 1 vs Stage 2 on Alpaca)

Pairwise judge file: **`artifacts/judge/alpaca_ckpt1_stage1_vs_ckpt2_stage2.jsonl`**. Structured summary: **`artifacts/metrics/forgetting_alpaca_ckpt1_stage1_vs_ckpt2_stage2.json`** and **`artifacts/tables/alpaca_forgetting_summary.csv`**.

| Metric | Value |
|--------|-------|
| Total pairs | 200 |
| ckpt1 wins | 38 |
| ckpt2 wins | 52 |
| Ties | 110 |
| ckpt1 win rate | 0.19 |
| ckpt2 win rate | 0.26 |
| Tie rate | 0.55 |
| Δ judge win rate (ckpt2 − ckpt1) | **+0.07** |
| ROUGE-L F1 ckpt1 → ckpt2 | 0.1160 → 0.1146 (**Δ ≈ −0.00143**) |
| BERTScore F1 ckpt1 → ckpt2 | ~flat (**Δ ≈ −1.5e−5**) |

Per-category breakdowns and **representative regression / improvement examples** (truncated predictions) are stored in that JSON file and support the qualitative discussion in **Section 3**.

### 2.5 Ablation results (Stage 2: 1 vs 2 epochs)

A second Stage 2 adapter was trained with **`STAGE2_EPOCHS=2`** into **`artifacts/checkpoints/stage2_json_adapter_ep2_ablate`**. Inference labels **`ckpt0_base_ablate_ep2`**, **`ckpt1_stage1_ablate_ep2`**, **`ckpt2_stage2_ablate_ep2`**; aggregation uses **`RUN_TAG=ablate_ep2`**.

Source: **`artifacts/tables/three_checkpoint_comparison_ablate_ep2.csv`**.

| Checkpoint | Judge win rate | ROUGE-L F1 | BERTScore F1 | JSON validity | Schema compliance | Exact match |
|------------|----------------|------------|--------------|---------------|-------------------|-------------|
| ckpt0_base_ablate_ep2 | 0.536 | 0.1230 | 0.8323 | 1.00 | 0.35 | 0.35 |
| ckpt1_stage1_ablate_ep2 | 0.421 | 0.1160 | 0.8365 | 1.00 | 0.18 | 0.18 |
| ckpt2_stage2_ablate_ep2 | 0.535 | 0.1149 | 0.8366 | 1.00 | 0.17 | 0.17 |

Compared to baseline **ckpt2_stage2**: ROUGE-L and BERTScore move only slightly; schema and exact match unchanged at **0.17** on this eval. The aggregate judge win rate for the ckpt2 row differs from the baseline table (**0.535** vs **0.500**) but must be interpreted in context of **multiple pairwise files** feeding the aggregate statistic.

### 2.6 Tables and figures

All quantitative tables above are generated from **`artifacts/tables/*.csv`** and the evaluation scripts in **`src/evaluation/`**. This submission does not rely on separate figure files; any future plots (e.g. loss curves from Slurm logs) would live under **`artifacts/`** and be referenced here.

---

## 3. Analysis

### 3.1 Qualitative comparison across checkpoints

Quantitatively, the base model is strongest on **strict JSON schema/exact match** against references in this benchmark, while fine-tuned models produce **valid JSON** with **different** object shapes than some references—so automatic “compliance” drops even when outputs are usable. On Alpaca, **judge preferences** and **ROUGE-L** need not move together: Stage 1 can improve **BERTScore** vs gold text while **pairwise win rate vs base** falls, if the judge favors fluency or style closer to the base model.

For **side-by-side text**, the repository stores full predictions under **`artifacts/predictions/`** and forgetting exemplars under **`artifacts/metrics/forgetting_alpaca_ckpt1_stage1_vs_ckpt2_stage2.json`**. Readers should open those artifacts for verbatim comparisons on specific prompt ids.

### 3.2 Failure cases

Common failure modes visible in metrics and manual spot checks include: **reference mismatch** on JSON (valid output, wrong keys or extra fields vs the eval reference), **Alpaca reference mismatch** (reasonable answer, low ROUGE), and **judge noise** (high tie rate, borderline dimension scores). The JSON error taxonomy CSV lists **category counts** per checkpoint for formatting and structural issues.

### 3.3 Forgetting vs retention

Head-to-head on Alpaca (**Stage 1 vs Stage 2**), the judge gives Stage 2 a **small win-rate edge** with **many ties**—not a sharp collapse of instruction-following on this slice. **ROUGE-L** decreases slightly from Stage 1 to Stage 2; **BERTScore** is nearly unchanged. That combination is best described as **mixed signals**: some retention, some regression on n-gram overlap to references, and judge outcomes that depend on task type (see per-category fields in the forgetting JSON).

### 3.4 What this implies for sequential fine-tuning

**Sequential fine-tuning** here adds a **second distribution** (teacher-style JSON). Imitation on discrete targets improves **format discipline** but can **misalign** strict schema/exact metrics if references are not identical to the teacher’s convention. **Extra Stage 2 epochs** in the ablation produce **diminishing** changes on automatic metrics, suggesting that more JSON training alone does not automatically fix reference alignment or Alpaca retention in this setup. Practical takeaway: **compose objectives explicitly** (retention regularizers, mixed replay, or reference harmonization) if both open-ended quality and strict JSON metrics must move together.

### 3.5 Limitations

Judge variance and **position bias** (mitigated but not removed), **evaluation subset** size without confidence intervals, **strict** JSON matching, and **software/hardware drift** on HPC all limit how strongly one can generalize from point estimates. Effective hyperparameters may differ when **`STAGE2_EPOCHS`** or adapter paths are overridden in Slurm.

---

## 4. Prompt engineering

### 4.1 Teacher generation prompts

The teacher sees a **fixed instruction block** requiring **only valid JSON**, **no** markdown fences, **no** chain-of-thought, and shape guidance from **`__JSON_EXAMPLE__`**, followed by the task **`__INSTRUCTION__`** and **`__INPUT__`**. The full template is in the **Appendix** (`prompts/teacher_json_generation.md`). Design goal: reduce prose wrappers so **`is_valid_json`** succeeds and retries stay rare.

### 4.2 Judge prompts (Alpaca and JSON)

**Pairwise Alpaca judging** uses a user template that defines **six dimensions** (1–5), requires a **single JSON object** with scores for A and B, **`winner`**, and **`justification`**, and forbids extra keys and thinking tags (**`prompts/judge_pairwise_eval.md`**). A **shared system message** (`prompts/judge_system_message.md`) reinforces “JSON only.”

**JSON judging** uses a parallel schema with one set of **`scores`** and **`justification`** (`prompts/judge_json_eval.md`).

### 4.3 Changes after failure analysis

Early judge runs sometimes returned **empty content**, **markdown fences**, or **chain-of-thought** before JSON, which broke parsing. Mitigations: (1) **stricter system and user instructions** (no fences, no reasoning tags); (2) **tolerant extraction** of the first JSON object in **`src/utils/json_schema_utils.py`** when the model still emits minor wrappers. Further tuning is done by **editing the `.md` templates** and re-running **`eval_alpaca_judge.py`** / **`eval_json_judge.py`**, not by hard-coding prose in Python.

---

## Appendix: Full prompt templates and student formatting

### A.1 Teacher — JSON imitation (`prompts/teacher_json_generation.md`)

```
You are generating a structured JSON output.
Return ONLY a single valid JSON object.
Do NOT include any reasoning, <redacted_thinking> tags, markdown fences, or extra text.
Do NOT wrap JSON in code blocks.
Use double quotes for all JSON strings.
If you cannot produce valid JSON, output exactly: {}

Required output shape (must match keys/value types):
__JSON_EXAMPLE__

Now complete the task.
Instruction: __INSTRUCTION__
Input: __INPUT__
```

### A.2 Judge — system message (`prompts/judge_system_message.md`)

```
You must reply with exactly one valid JSON object and no markdown fences, no code blocks, and no text before or after the JSON. Do not use chain-of-thought, reasoning tags, or prose; output only the JSON object.
```

### A.3 Judge — Alpaca pairwise (`prompts/judge_pairwise_eval.md`)

```
You are an expert judge for instruction-following quality.
You will see one instruction (and optional input) plus two candidate responses.
Your job is to score each response on multiple dimensions and then pick a winner.

Dimensions (1-5, higher is better):
- instruction_following
- correctness
- clarity
- completeness
- structured_output_validity (for JSON-like outputs; otherwise use 3 as neutral)
- hallucination_risk (1 = very hallucinated, 5 = minimal hallucination)

Return ONLY a single JSON object with this schema:
{
  "prompt_id": "...",
  "checkpoint_a": "...",
  "checkpoint_b": "...",
  "response_a_scores": {
    "instruction_following": int,
    "correctness": int,
    "clarity": int,
    "completeness": int,
    "structured_output_validity": int,
    "hallucination_risk": int
  },
  "response_b_scores": { ... same keys ... },
  "winner": "A" | "B" | "tie",
  "justification": "short natural language string"
}

Do not include any extra keys, comments, or markdown. Do not output chain-of-thought or XML-style thinking blocks before the JSON.

Instruction: __INSTRUCTION__
Input: __INPUT__

Response A (checkpoint __CHECKPOINT_A__):
__RESPONSE_A__

Response B (checkpoint __CHECKPOINT_B__):
__RESPONSE_B__

Now return the JSON object.
```

### A.4 Judge — JSON per example (`prompts/judge_json_eval.md`)

```
You are an expert judge for JSON-structured outputs.
You will see an instruction (and optional input), the model's JSON prediction, and the reference JSON.
Score the prediction on the following dimensions (1-5, higher is better):
- instruction_following
- correctness
- clarity
- completeness
- structured_output_validity
- hallucination_risk

Return ONLY a JSON object with this schema:
{
  "prompt_id": "...",
  "checkpoint": "...",
  "scores": {
    "instruction_following": int,
    "correctness": int,
    "clarity": int,
    "completeness": int,
    "structured_output_validity": int,
    "hallucination_risk": int
  },
  "justification": "short natural language string"
}

Do not include any extra keys or markdown. Do not output chain-of-thought or XML-style thinking blocks before the JSON.

Instruction: __INSTRUCTION__
Input: __INPUT__

Prediction (checkpoint __CHECKPOINT__):
__PREDICTION__

Reference JSON:
__REFERENCE__

Now return the JSON object.
```

### A.5 Student training formatting (SFT string)

Stage 1 and Stage 2 training both map each example to a single causal-LM string (see `format_row` in **`train_stage1_alpaca.py`** and **`train_stage2_json.py`**):

```
Instruction: {instruction}
Input: {input}
Response: {output}
```

Placeholders **`__...__`** in the appendix templates are filled at runtime by **`src/utils/prompt_loader.py`**. Optional overrides: **`TEACHER_PROMPT_TEMPLATE`**, **`JUDGE_SYSTEM_PROMPT`**, **`JUDGE_PAIRWISE_PROMPT`**, **`JUDGE_JSON_PROMPT`**.

---

## Reproducibility (HPC)

```bash
module load anaconda3   # or your cluster module
cd /work/<user>/LLM_Agentic_Systems_Finetuning
source .venv/bin/activate   # if used
set -a && source .env && set +a
export PYTHONPATH="$PWD"
```

**Training:** `sbatch hpc/stage1_train.slurm`, `sbatch hpc/stage2_train.slurm` (ablation: export **`STAGE2_EPOCHS=2`**, **`STAGE2_OUT_DIR=artifacts/checkpoints/stage2_json_adapter_ep2_ablate`**, etc.).

**Inference:** `python3 src/inference/generate_checkpoint_outputs.py` (set **`STAGE1_ADAPTER_PATH`**, **`STAGE2_ADAPTER_PATH`**, **`CKPT*_LABEL`**, limits as needed).

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

For **`RUN_TAG`**, prediction file prefixes must match **`CKPT0_LABEL`**, **`CKPT1_LABEL`**, **`STAGE2_CKPT_LABEL`**. Full layout: **`README.md`**.

---

## Artifacts index

| Path | Role |
|------|------|
| `artifacts/tables/three_checkpoint_comparison.csv` | Baseline three-checkpoint summary |
| `artifacts/tables/three_checkpoint_comparison_ablate_ep2.csv` | Ablation summary |
| `artifacts/tables/alpaca_judge_summary_by_checkpoint*.csv` | Per-dimension judge averages |
| `artifacts/tables/alpaca_forgetting_summary*.csv` | Forgetting summary |
| `artifacts/tables/json_metrics_by_checkpoint.csv` | JSON auto metrics |
| `artifacts/tables/json_error_taxonomy_by_checkpoint.csv` | Error categories |
| `artifacts/metrics/alpaca_auto_metrics_*.json` | Alpaca auto metrics |
| `artifacts/metrics/json_auto_metrics_*.json` | JSON auto metrics |
| `artifacts/metrics/forgetting_alpaca_ckpt1_stage1_vs_ckpt2_stage2.json` | Forgetting detail + exemplars |
| `artifacts/judge/*.jsonl` | Raw judge outputs |
| `artifacts/predictions/*.jsonl` | Per-checkpoint generations |

---

## Conclusion

We implemented **two-stage QLoRA** (Alpaca then teacher JSON imitation) on **UTSA HPC** and evaluated **three checkpoints** with **automatic metrics** and an **LLM judge**. Stage 2 maintains **high JSON validity** while **strict schema and exact match** relative to references **soften** vs the base model; Alpaca-side signals show **mixed** forgetting and retention, with **tie-heavy** judge outcomes and only **small** gains from an **extra Stage 2 epoch** in the ablation. Future work: align eval references with teacher conventions, enlarge or stratify eval sets, and explore replay or regularization if Alpaca retention must improve jointly with JSON specialization.
