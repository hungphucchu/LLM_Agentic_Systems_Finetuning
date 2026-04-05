import os

from src.utils.project_paths import ensure_repo_on_path

ensure_repo_on_path()

from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from src.training.qlora_utils import attach_lora, load_4bit_model, load_tokenizer
from src.training.sft_format import format_sft_text_row
from src.utils.app_config import student_model_id

OUT_DIR = "artifacts/checkpoints/stage1_alpaca_adapter"
STAGE1_LR = float(os.getenv("STAGE1_LR", "2e-5"))
STAGE1_EPOCHS = int(os.getenv("STAGE1_EPOCHS", "2"))
STAGE1_MAX_LENGTH = int(os.getenv("STAGE1_MAX_LENGTH", "512"))


def main() -> None:
    model_name = student_model_id()
    ds = load_dataset("json", data_files="data/processed/alpaca_train.jsonl", split="train")
    ds = ds.map(format_sft_text_row)

    tokenizer = load_tokenizer(model_name)
    model = attach_lora(load_4bit_model(model_name))

    def tokenize(batch):
        # Shorter context for faster training; explain in report.
        return tokenizer(batch["text"], truncation=True, max_length=STAGE1_MAX_LENGTH)

    tokenized = ds.map(tokenize, batched=True, remove_columns=ds.column_names)

    args = TrainingArguments(
        output_dir=OUT_DIR,
        learning_rate=STAGE1_LR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=STAGE1_EPOCHS,
        logging_steps=10,
        save_strategy="epoch",
        # V100 (gpu1v100) has no BF16 tensor cores; use FP16 instead.
        fp16=True,
        bf16=False,
        dataloader_pin_memory=False,
        report_to=[],
    )

    # Variable-length tokenized rows cannot be stacked by the default collator.
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
