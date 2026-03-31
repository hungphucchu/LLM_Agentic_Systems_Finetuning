import os

from datasets import load_dataset
from peft import PeftModel
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from src.training.qlora_utils import load_4bit_model, load_tokenizer


BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
STAGE1_ADAPTER = "artifacts/checkpoints/stage1_alpaca_adapter"
OUT_DIR = "artifacts/checkpoints/stage2_json_adapter"

NUM_CORES = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))


def format_row(ex):
    text = f"Instruction: {ex['instruction']}\nInput: {ex['input']}\nResponse: {ex['output']}"
    return {"text": text}


def main() -> None:
    ds = load_dataset("json", data_files="data/processed/json_train_teacher.jsonl", split="train")
    # Use multiple CPU cores for faster text formatting.
    ds = ds.map(format_row, num_proc=NUM_CORES)

    tokenizer = load_tokenizer(BASE_MODEL)
    base = load_4bit_model(BASE_MODEL)
    model = PeftModel.from_pretrained(base, STAGE1_ADAPTER, is_trainable=True)

    def tokenize(batch):
        # Match Stage 1 context length for consistency and speed.
        return tokenizer(batch["text"], truncation=True, max_length=512)

    # Multi-core batched tokenization to keep GPUs fed efficiently.
    tokenized = ds.map(
        tokenize,
        batched=True,
        num_proc=NUM_CORES,
        remove_columns=ds.column_names,
    )

    args = TrainingArguments(
        output_dir=OUT_DIR,
        learning_rate=1e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        bf16=False,
        # Feed the GPU efficiently from all allocated CPU cores.
        dataloader_num_workers=NUM_CORES,
        dataloader_pin_memory=True,
        # Older transformers on HPC may not support newer flags like
        # group_by_length/gradient_checkpointing/optim; keep this minimal.
        report_to=[],
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )
    trainer.train()
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)


if __name__ == "__main__":
    main()
