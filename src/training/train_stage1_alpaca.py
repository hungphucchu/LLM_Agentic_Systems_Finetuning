from datasets import load_dataset
from transformers import Trainer, TrainingArguments

from src.training.qlora_utils import attach_lora, load_4bit_model, load_tokenizer


MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
OUT_DIR = "artifacts/checkpoints/stage1_alpaca_adapter"


def format_row(ex):
    text = f"Instruction: {ex['instruction']}\nInput: {ex['input']}\nResponse: {ex['output']}"
    return {"text": text}


def main() -> None:
    ds = load_dataset("json", data_files="data/processed/alpaca_train.jsonl", split="train")
    ds = ds.map(format_row)

    tokenizer = load_tokenizer(MODEL_NAME)
    model = attach_lora(load_4bit_model(MODEL_NAME))

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=1024)

    tokenized = ds.map(tokenize, batched=True, remove_columns=ds.column_names)

    args = TrainingArguments(
        output_dir=OUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to=[],
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized)
    trainer.train()
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)


if __name__ == "__main__":
    main()
