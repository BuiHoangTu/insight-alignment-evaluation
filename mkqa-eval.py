import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM

from mkqa_eval.mkqa_eval import evaluate, MKQAAnnotation, MKQAPrediction

DEFAULT_MODEL_NAME = "../llama3_8b_lacomsa/checkpoint-94/"
DEFAULT_LANGS = ["en", "de", "ru", "es", "fr", "th", "zh", "ja", "vi", "tr", "it"]  # , "sw"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_OUTPUT = "results"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG_MAX_EXAMPLES = 10
ACTIVE_MAX_EXAMPLES = 1000
BATCH_SIZE = 64


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
    """
    Generate predictions in batches with padding for efficient inference.
    """
    predictions = {}
    for i in tqdm(
        range(0, len(dataset_split), BATCH_SIZE),
        desc=f"Generating predictions ({lang})",
        mininterval=50,
        maxinterval=300,
    ):
        batch = dataset_split[i : i + BATCH_SIZE]

        # Format each example as a separate chat conversation
        batch_conversations = [
            [{"role": "user", "content": example["queries"][lang]}] for example in batch
        ]

        # Apply chat template
        formatted_batch = [
            tokenizer.apply_chat_template(
                conv, 
                tokenize=False, 
                add_generation_prompt=True,
            )
            for conv in batch_conversations
        ]

        # Tokenize the batch with padding
        tokenized_batch = tokenizer(
            formatted_batch,
            padding="longest",  # pad to max length in this batch
            truncation=True,  # truncate if too long
            return_tensors="pt",
        ).to(device)

        # Step 4: Generate predictions
        with torch.no_grad():
            outputs = model.generate(
                **tokenized_batch,
                max_new_tokens=max_tokens,
                num_return_sequences=1
            )

        # Step 5: Decode predictions, removing the input prompt
        input_lengths = tokenized_batch["input_ids"].shape[1]
        generated_tokens = [out[input_lengths:] for out in outputs]

        pred_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        for example, pred_text in zip(batch, pred_texts):
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

    tokenizer.padding_side = "left"
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
        else:
            dataset_split = dataset_split.select(
                range(min(ACTIVE_MAX_EXAMPLES, len(dataset_split)))
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
    (Path(args.output_path) / "results-mkqa.json").write_text(
        json.dumps(all_metrics, indent=2)
    )

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
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT, help="Output file for results")

    args = parser.parse_args()

    main(args)
