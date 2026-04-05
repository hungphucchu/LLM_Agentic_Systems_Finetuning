"""Shared supervised fine-tuning text format for Stage 1 and Stage 2."""


def format_sft_text_row(ex: dict) -> dict:
    text = (
        f"Instruction: {ex['instruction']}\nInput: {ex['input']}\nResponse: {ex['output']}"
    )
    return {"text": text}
