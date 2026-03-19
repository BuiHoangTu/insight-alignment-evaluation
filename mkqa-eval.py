import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
from collections import Counter


# -----------------------------
# 1. Utilities for EM & F1
# -----------------------------
def normalize_answer(s):
    """Lower text, remove punctuation/articles/extra whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match_score(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


# -----------------------------
# 2. Load MKQA dataset (subset)
# -----------------------------
mkqa = load_dataset("apple/mkqa", trust_remote_code=True)

# Pick 3 languages: English, French, German
languages = ["en", "fr", "de"]

# -----------------------------
# 3. Load HF decoder-only model
# -----------------------------
model_name = "../llama3_8b_lacomsa/checkpoint-94/"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# -----------------------------
# 4. Generate predictions
# -----------------------------
def generate_answer(question, max_length=64):
    inputs = tokenizer(question, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# -----------------------------
# 5. Evaluate per language
# -----------------------------
results = {}

for lang in languages:
    em_scores = []
    f1_scores = []
    queries = mkqa["test"]["queries"][lang]
    answers_list = mkqa["test"]["answers"][
        lang
    ]  # list of lists (multiple correct answers)

    for i, question in enumerate(queries):
        pred = generate_answer(question)
        # For multiple references, take max score
        em = max([exact_match_score(pred, ref) for ref in answers_list[i]])
        f1 = max([f1_score(pred, ref) for ref in answers_list[i]])
        em_scores.append(em)
        f1_scores.append(f1)

        if i < 3:  # print first few examples
            print(f"Lang={lang} | Q: {question} | P: {pred} | EM={em} | F1={f1:.2f}")

    results[lang] = {
        "EM": sum(em_scores) / len(em_scores),
        "F1": sum(f1_scores) / len(f1_scores),
    }

# -----------------------------
# 6. Print metrics
# -----------------------------
print("\n=== MKQA Evaluation Results ===")
for lang, metrics in results.items():
    print(f"{lang} -> EM: {metrics['EM']*100:.2f}%, F1: {metrics['F1']*100:.2f}%")
