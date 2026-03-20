# run_eval_hf.py
import collections
import logging

import torch
from tqdm import tqdm
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM

from mkqa_eval.mkqa_eval import evaluate, MKQAAnnotation, MKQAPrediction

# ----------------------------
# Configuration
# ----------------------------
MODEL_NAME = "../llama3_8b_lacomsa/checkpoint-94/"
LANGS = ["en", "es", "de", "fr", "ar"]
MAX_TOKENS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Debug subset settings
DEBUG_MODE = True  # set to False for full dataset run
DEBUG_MAX_EXAMPLES_PER_LANG = 10

# ----------------------------
# Load model & tokenizer
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# ----------------------------
# Load MKQA dataset from Hugging Face
# ----------------------------
dataset = load_dataset("mkqa", trust_remote_code=True)
print("MKQA dataset loaded:", dataset)


# ----------------------------
# Helper functions, derive from mkqa_eval.mkqa_eval
# ----------------------------
def prepare_gold_annotations(dataset_split, lang):
    annotations = {}
    for example in dataset_split:
        valid_answers, answer_types = [], []
        for ans in example["answers"][lang]:
            valid_answers.append(ans["text"] or "")
            valid_answers.extend(ans.get("aliases", []))
            answer_types.append(ans["type"])

        annotation = MKQAAnnotation(
            example_id=str(example["example_id"]),
            types=list(set(answer_types)),
            answers=list(set(valid_answers)),
        )

        annotations[annotation.example_id] = annotation

    if len(annotations) != 10000:
            logging.warning(
                f"The annotations file you've provided contains {len(annotations)} for language {lang} examples, where 10000 are expected."
            )
    return annotations


def generate_prediction(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
        )
    pred_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    )
    return pred_text.strip()


def prepare_predictions(dataset_split, lang):
    predictions = {}
    for example in tqdm(dataset_split, desc=f"Generating predictions ({lang})"):
        # Use the actual MKQA question text
        prompt = f"Question ({lang}): {example['queries'][lang]} "
        pred_text = generate_prediction(prompt)
        predictions[str(example["example_id"])] = MKQAPrediction(
            example_id=str(example["example_id"]),
            prediction=pred_text,
            binary_answer=None,
            no_answer_prob=0.0,
        )
    return predictions


# ----------------------------
# Main evaluation loop
# ----------------------------
all_metrics = {}
for lang in LANGS:
    print(f"\nEvaluating language: {lang.upper()}")
    dataset_split = dataset["train"]
    if DEBUG_MODE:
        dataset_split = dataset_split.select(
            range(min(DEBUG_MAX_EXAMPLES_PER_LANG, len(dataset_split)))
        )
    gold_annotations = prepare_gold_annotations(dataset_split, lang)
    predictions = prepare_predictions(dataset_split, lang)

    metrics = evaluate(
        annotations=gold_annotations,
        predictions=predictions,
        language=lang,
        out_dir=f"results/{lang}",
        verbose=False,
        print_metrics=True,
    )
    all_metrics[lang] = metrics

# Macro-average across languages
macro_em = sum(m["best_em"] for m in all_metrics.values()) / len(LANGS)
macro_f1 = sum(m["best_f1"] for m in all_metrics.values()) / len(LANGS)
print(f"\n=== Macro-Average MKQA ===")
print(f"Macro EM: {macro_em:.2f}, Macro F1: {macro_f1:.2f}")
