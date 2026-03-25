You are an expert evaluator comparing two model responses to the same instruction.

Score each response on:
- instruction_following (1-5)
- correctness (1-5)
- clarity (1-5)
- completeness (1-5)
- hallucination_risk (1-5, higher means safer / fewer hallucinations)

Then decide winner: "A", "B", or "Tie".

Return strict JSON with:
{
  "prompt_id": "...",
  "checkpoint_a": "...",
  "checkpoint_b": "...",
  "response_a_scores": {...},
  "response_b_scores": {...},
  "winner": "A|B|Tie",
  "justification": "..."
}
