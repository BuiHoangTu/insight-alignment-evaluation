import json
import logging

import torch
from tqdm import tqdm
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM

from mkqa_eval.mkqa_eval import evaluate, MKQAAnnotation, MKQAPrediction

DEFAULT_MODEL_NAME = "../llama3_8b_lacomsa/checkpoint-94/"
DEFAULT_LANGS = ["en", "de", "ru", "es", "fr", "th", "zh", "ja", "vi", "tr", "it"]  # , "sw"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_OUTPUT = "results/results-mkqa.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG_MAX_EXAMPLES = 10


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


def prepare_predictions(dataset_split, lang, tokenizer, model, device, max_tokens):
    predictions = {}
    for example in tqdm(dataset_split, desc=f"Generating predictions ({lang})"):
        # Use the actual MKQA question text
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["queries"][lang]}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens
            )

        pred_token = outputs[0][inputs.input_ids.shape[1] :]

        pred_text = tokenizer.decode(pred_token, skip_special_tokens=True)

        predictions[str(example["example_id"])] = MKQAPrediction(
            example_id=str(example["example_id"]),
            prediction=pred_text,
            binary_answer=None,
            no_answer_prob=0.0,
        )
    return predictions


def main(args):
    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(DEVICE)
    model.eval()

    # Load MKQA dataset from Hugging Face
    dataset = load_dataset("mkqa", trust_remote_code=True)
    print("MKQA dataset loaded:", dataset)

    # Main evaluation loop
    all_metrics = {}
    for lang in args.langs:
        print(f"\nEvaluating language: {lang.upper()}")
        dataset_split = dataset["train"]
        if args.debug:
            dataset_split = dataset_split.select(
                range(min(DEBUG_MAX_EXAMPLES, len(dataset_split)))
            )
        gold_annotations = prepare_gold_annotations(dataset_split, lang)
        predictions = prepare_predictions(dataset_split, lang, tokenizer, model, DEVICE, args.max_tokens)

        metrics = evaluate(
            annotations=gold_annotations,
            predictions=predictions,
            language=lang,
            out_dir=f"results/{lang}",
            verbose=False,
            print_metrics=True,
        )
        all_metrics[lang] = metrics

    # save result
    with open(args.output, "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Macro-average across languages
    macro_em = sum(m["best_em"] for m in all_metrics.values()) / len(args.langs)
    macro_f1 = sum(m["best_f1"] for m in all_metrics.values()) / len(args.langs)
    print("\n=== Macro-Average MKQA ===")
    print(f"Macro EM: {macro_em:.2f}, Macro F1: {macro_f1:.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate MKQA question answering.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Model name or path")
    parser.add_argument("--langs", nargs="+", default=DEFAULT_LANGS, help="Languages to evaluate")
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max new tokens")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with fewer examples")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output file for results")

    args = parser.parse_args()

    main(args)
