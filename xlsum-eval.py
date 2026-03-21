import json
import os
import subprocess
import tempfile
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODEL_PATH = "../llama3_8b_lacomsa/checkpoint-94/"
DEFAULT_LANGS = [
    "thai",
    "english",
    "russian",
    "spanish",
    "chinese_simplified",
    "swahili",
    "french",
    "japanese",
    "vietnamese",
    "turkish",
]
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_OUTPUT = "results/results-xlsum.json"
DEFAULT_CHECKPOINT_DIR = "./checkpoints"

ROUGE_SCRIPT_PATH = "./multilingual_rouge_scoring"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG_MAX_EXAMPLES = 5

def main(args):
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(DEVICE)
    model.eval()

    results = []

    for lang in args.langs:
        print(f"Evaluating language: {lang}")
        dataset = load_dataset("csebuetnlp/xlsum", lang)
        test_split = dataset["test"]

        if args.debug:
            test_split = test_split.select(range(min(args.max_examples, len(test_split))))

        # Create temporary files for references and predictions
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False, encoding="utf-8", dir=args.checkpoint_dir
        ) as ref_file, tempfile.NamedTemporaryFile(
            mode="w+", delete=False, encoding="utf-8", dir=args.checkpoint_dir
        ) as pred_file:

            ref_path = ref_file.name
            pred_path = pred_file.name

            # Write references
            for item in test_split:
                ref_file.write(item["summary"].strip() + "\n")
            ref_file.flush()

            # Generate predictions
            for item in test_split:
                prompt = f"Summarize the following article in {lang}:\n{item['text']}\nSummary:"
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(DEVICE)

                inputs = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(DEVICE)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )
                    
                pred_token = outputs[0][inputs.input_ids.shape[1] :]
                pred_text = tokenizer.decode(pred_token, skip_special_tokens=True)
                pred_file.write(pred_text.strip() + "\n")
            pred_file.flush()

            # Run XLSum ROUGE CLI
            temp_csv = tempfile.NamedTemporaryFile(
                mode="w+", delete=False, encoding="utf-8"
            ).name
            cmd = [
                "python",
                "-m",
                "rouge_score.rouge",
                f"--target_filepattern={ref_path}",
                f"--prediction_filepattern={pred_path}",
                f"--output_filename={temp_csv}",
                "--use_stemmer=false",
                f"--lang={lang}",
            ]
            subprocess.run(cmd, cwd=ROUGE_SCRIPT_PATH, check=True)

            # Extract ROUGE-L F1
            rouge_l_f = 0.0
            with open(temp_csv, "r", encoding="utf-8") as f:
                for line in f:
                    if "rougeL" in line.lower() and "f" in line.lower():
                        rouge_l_f = float(line.strip().split(",")[-1])
                        break

            results.append((lang, rouge_l_f))

            # Remove temporary files
            # os.remove(ref_path)
            # os.remove(pred_path)
            os.remove(temp_csv)

    # Save results
    with open(args.output_path, "w", newline="", encoding="utf-8") as f_out:
        json.dump(results, f_out)

    print(f"Evaluation complete! Results saved to {args.output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate XLSum summarization.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Model path")
    parser.add_argument("--langs", nargs="+", default=DEFAULT_LANGS, help="Languages to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Max new tokens")
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT, help="Output JSON path")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with fewer examples")
    parser.add_argument("--max_examples", type=int, default=DEBUG_MAX_EXAMPLES, help="Max examples per lang in debug mode")
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR, help="Directory for temporary files")

    args = parser.parse_args()
    main(args)
