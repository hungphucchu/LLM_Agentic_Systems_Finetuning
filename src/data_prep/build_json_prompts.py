import json
from typing import Dict, List

from src.utils.io_utils import write_jsonl


def build_prompt_pool() -> List[Dict]:
    # Build a diverse prompt pool for teacher imitation.
    # Output targets must be valid JSON, so each row includes a json_example
    # that fixes the *shape* (keys + value types) expected in the response.

    tasks: List[Dict] = []

    # ---------- Task type 1: JSON extraction ----------
    extraction_rows = [
        ("Alice", "Bob", "San Antonio", "2024-05-12"),
        ("Mia", "Noah", "Austin", "2024-06-03"),
        ("Ethan", "Zoe", "Houston", "2024-07-21"),
        ("Liam", "Olivia", "Dallas", "2024-08-09"),
        ("Sophia", "Lucas", "San Diego", "2024-09-15"),
        ("Ava", "Henry", "Seattle", "2024-10-02"),
        ("Isabella", "Jack", "Chicago", "2024-11-18"),
        ("Amir", "Sara", "Boston", "2024-12-01"),
        ("Chloe", "David", "Denver", "2025-01-14"),
        ("Grace", "Ryan", "Phoenix", "2025-02-26"),
        ("Ella", "William", "Atlanta", "2025-03-07"),
        ("Nora", "James", "Miami", "2025-04-19"),
        ("Zara", "Leo", "Portland", "2025-05-23"),
        ("Hannah", "Kai", "Las Vegas", "2025-06-30"),
        ("Maya", "Owen", "Tampa", "2025-07-08"),
        ("Ruby", "Caleb", "Orlando", "2025-08-12"),
        ("Lily", "Gabriel", "Charlotte", "2025-09-25"),
        ("Aria", "Mason", "Raleigh", "2025-10-05"),
        ("Eva", "Eli", "Baltimore", "2025-11-17"),
        ("Tessa", "Cole", "Minneapolis", "2025-12-24"),
    ]
    for a, b, loc, date in extraction_rows:
        tasks.append(
            {
                "task_type": "json_extraction",
                "instruction": "Extract entities and dates into JSON.",
                "input": f"{a} met {b} in {loc} on {date}.",
                "json_example": json.dumps(
                    {"entities": [a, b], "location": loc, "date": date}
                ),
            }
        )

    # ---------- Task type 2: Schema-constrained generation ----------
    # Here the shape is a profile with name (string), age (int), city (string).
    schema_instruction = (
        "You are given a JSON schema. Generate a profile object that matches the schema exactly. "
        "Return only valid JSON."
    )
    schema_input = json.dumps({"required": ["name", "age", "city"]})
    profile_rows = [
        ("Avery", 28, "Austin"),
        ("Noelle", 34, "Boston"),
        ("Quinn", 22, "Chicago"),
        ("Jordan", 45, "Denver"),
        ("Riley", 29, "Seattle"),
        ("Taylor", 31, "Miami"),
        ("Morgan", 19, "Phoenix"),
        ("Cameron", 26, "Dallas"),
        ("Rowan", 37, "Atlanta"),
        ("Parker", 24, "Portland"),
        ("Reese", 41, "Tampa"),
        ("Skyler", 27, "Orlando"),
        ("Devon", 23, "Charlotte"),
        ("Casey", 38, "Raleigh"),
        ("Finley", 21, "Baltimore"),
        ("Harper", 33, "Minneapolis"),
        ("Emerson", 25, "San Diego"),
        ("Hayden", 20, "Houston"),
        ("Dakota", 36, "Las Vegas"),
        ("Charlie", 30, "San Antonio"),
    ]
    for name, age, city in profile_rows:
        tasks.append(
            {
                "task_type": "schema_generation",
                "instruction": schema_instruction,
                "input": f"Schema: {schema_input}\nUser preference: generate a profile for {name} aged {age} living in {city}.",
                "json_example": json.dumps({"name": name, "age": age, "city": city}),
            }
        )

    # ---------- Task type 3: Exact-label classification with JSON ----------
    # Allowed labels are strings; we keep the key stable: {"label": "..."}.
    class_instruction = (
        "Classify the sentiment into exactly one label from [positive, neutral, negative]. "
        "Return JSON in the required shape only."
    )
    classification_rows = [
        ("The product is okay, but shipping was slow.", "neutral"),
        ("Amazing quality and fast delivery. Would buy again!", "positive"),
        ("Terrible experience—arrived damaged and late.", "negative"),
        ("It works fine, but the battery life could be better.", "neutral"),
        ("Great value for the price.", "positive"),
        ("Not worth the money. Very disappointed.", "negative"),
        ("Service was average, nothing special.", "neutral"),
        ("Loved it—super reliable and easy to use.", "positive"),
        ("Broke after one week. Do not recommend.", "negative"),
        ("Decent product overall, but a few issues.", "neutral"),
        ("Fantastic! Exceeded expectations.", "positive"),
        ("Horrible support and confusing setup.", "negative"),
        ("Okay experience; might try again later.", "neutral"),
        ("Best purchase I've made this year.", "positive"),
        ("Waste of time—keeps freezing.", "negative"),
        ("Not bad, but also not great.", "neutral"),
        ("Customer service resolved my issue quickly.", "positive"),
        ("The update made everything worse.", "negative"),
        ("It’s acceptable though not impressive.", "neutral"),
        ("Completely satisfied with the results.", "positive"),
    ]
    for text, label in classification_rows:
        tasks.append(
            {
                "task_type": "exact_label_classification",
                "instruction": class_instruction,
                "input": text,
                "json_example": json.dumps({"label": label}),
            }
        )

    # ---------- Task type 4: JSON repair ----------
    repair_instruction = (
        "Repair malformed JSON. Return ONLY valid JSON (no explanations)."
    )
    # Provide malformed input with a known valid output shape.
    repair_rows = [
        ('{"name":"Sam","age": 22,}', {"name": "Sam", "age": 22}),
        ('{"name":"Jamie", "age": 31,}', {"name": "Jamie", "age": 31}),
        ('{"name":"Ari","age": 19,}', {"name": "Ari", "age": 19}),
        ('{"name":"Nina","age": 27,}', {"name": "Nina", "age": 27}),
        ('{"name":"Omar","age": 40,}', {"name": "Omar", "age": 40}),
        ('{"name":"Kara","age": 24,}', {"name": "Kara", "age": 24}),
        ('{"name":"Ben","age": 18,}', {"name": "Ben", "age": 18}),
        ('{"name":"Mina","age": 35,}', {"name": "Mina", "age": 35}),
        ('{"name":"Zed","age": 29,}', {"name": "Zed", "age": 29}),
        ('{"name":"Elena","age": 26,}', {"name": "Elena", "age": 26}),
        ('{"name":"Tariq","age": 32,}', {"name": "Tariq", "age": 32}),
        ('{"name":"Sofia","age": 23,}', {"name": "Sofia", "age": 23}),
        ('{"name":"Noah","age": 21,}', {"name": "Noah", "age": 21}),
        ('{"name":"Emma","age": 38,}', {"name": "Emma", "age": 38}),
        ('{"name":"Leo","age": 25,}', {"name": "Leo", "age": 25}),
        ('{"name":"Ivy","age": 17,}', {"name": "Ivy", "age": 17}),
        ('{"name":"Maya","age": 33,}', {"name": "Maya", "age": 33}),
        ('{"name":"Oscar","age": 28,}', {"name": "Oscar", "age": 28}),
        ('{"name":"Grace","age": 20,}', {"name": "Grace", "age": 20}),
        ('{"name":"Lena","age": 36,}', {"name": "Lena", "age": 36}),
    ]
    for bad, good in repair_rows:
        tasks.append(
            {
                "task_type": "json_repair",
                "instruction": repair_instruction,
                "input": bad,
                "json_example": json.dumps(good),
            }
        )

    # ---------- Task type 5: Tool-call argument generation ----------
    tool_instruction = (
        "Create JSON arguments for a function call. "
        "Return ONLY the JSON argument object."
    )
    tool_function = "weather"
    tool_rows = [
        ("Austin", "tomorrow"),
        ("Dallas", "today"),
        ("Seattle", "tomorrow"),
        ("Chicago", "next week"),
        ("Miami", "today"),
        ("Denver", "tomorrow"),
        ("Phoenix", "next week"),
        ("Boston", "today"),
        ("Atlanta", "tomorrow"),
        ("Portland", "next week"),
        ("Tampa", "today"),
        ("Orlando", "tomorrow"),
        ("Charlotte", "next week"),
        ("Raleigh", "today"),
        ("Baltimore", "tomorrow"),
        ("Minneapolis", "next week"),
        ("San Diego", "today"),
        ("Houston", "tomorrow"),
        ("Las Vegas", "next week"),
        ("San Antonio", "today"),
    ]
    for city, date in tool_rows:
        tasks.append(
            {
                "task_type": "tool_call_args",
                "instruction": tool_instruction,
                "input": f"Get {tool_function} for {city} {date}.",
                "json_example": json.dumps(
                    {"function": tool_function, "arguments": {"city": city, "date": date}}
                ),
            }
        )

    # Sanity check: we built 20 per task type => 100 total.
    return tasks


def main() -> None:
    rows = build_prompt_pool()
    write_jsonl("data/processed/json_prompt_pool.jsonl", rows)
    print(f"Saved json_prompt_pool={len(rows)}")


if __name__ == "__main__":
    main()
