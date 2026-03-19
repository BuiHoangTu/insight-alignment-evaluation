import os
import subprocess
import csv
import tempfile
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============ CONFIG ============
MODEL_PATH = "../llama3_8b_lacomsa/checkpoint-94/"
LANGS = ["english", "spanish", "amharic"]  # languages to evaluate
MAX_NEW_TOKENS = 128
TEMP = 0.0
ROUGE_SCRIPT_PATH = "./multilingual_rouge_scoring"  # path to XLSum repo script
OUTPUT_CSV = "results-xlsum.csv"
# ================================

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

# Load XLSum dataset

results = []

for lang in LANGS:
    print(f"Evaluating language: {lang}")
    dataset = load_dataset("csebuetnlp/xlsum", lang)
    test_split = dataset["validation"]

    # Create temporary files for references and predictions
    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False, encoding="utf-8"
    ) as ref_file, tempfile.NamedTemporaryFile(
        mode="w+", delete=False, encoding="utf-8"
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
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMP,
                do_sample=False,
            )
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred_file.write(summary.strip() + "\n")
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
        subprocess.run(cmd, cwd=ROUGE_SCRIPT_PATH)

        # Extract ROUGE-L F1
        rouge_l_f = 0.0
        with open(temp_csv, "r", encoding="utf-8") as f:
            for line in f:
                if "rougeL" in line.lower() and "f" in line.lower():
                    rouge_l_f = float(line.strip().split(",")[-1])
                    break

        results.append((lang, rouge_l_f))

        # Remove temporary files
        os.remove(ref_path)
        os.remove(pred_path)
        # os.remove(temp_csv)
        print(temp_csv)

# Save results
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f_out:
    writer = csv.writer(f_out)
    writer.writerow(["language", "ROUGE-L_F"])
    writer.writerows(results)

print(f"Evaluation complete! Results saved to {OUTPUT_CSV}")
