import json
from pathlib import Path
import pandas as pd
import numpy as np

def convert_score(score, max):
    proportion = score / max
    converted = np.log(proportion / (1 - proportion))

    return converted


def read_score(results_path):
    """
    Convert evaluation results from JSON files to a pandas DataFrame.

    Args:
        results_path (str or Path): Path to the results folder containing the JSON files.

    Returns:
        pd.DataFrame: DataFrame with evaluation scores for different languages and tasks.
    """
    results_path = Path(results_path)

    # Load lm-eval results
    j_lm_eval = json.loads((results_path / "results-lm-eval.json").read_text())
    result_lm_eval = j_lm_eval["results"]

    # Define languages and tasks
    LANGS = ["en", "de", "ru", "es", "fr", "th", "zh", "sw", "ja", "vi", "tr", "it"]
    TASKS = ["mgsm_direct", "xcopa"]

    # Create DataFrame
    df = pd.DataFrame(index=LANGS, columns=TASKS)
    df.index.name = "lang"

    # Populate DataFrame with lm-eval scores
    for key, item in result_lm_eval.items():
        task, lang = key.rsplit("_", 1)

        if task == "mgsm_direct":
            score = item["exact_match,flexible-extract"]
        elif task == "xcopa":
            score = item["acc,none"]

        df.loc[lang, task] = score

    # Load mkqa results
    j_mkqa = json.loads((results_path / "results-mkqa.json").read_text())

    # Add mkqa column
    df["mkqa"] = np.nan

    # Populate mkqa scores
    for lang, metrics in j_mkqa.items():
        if lang == "zh_cn":
            lang = "zh"
        df.loc[lang, "mkqa"] = metrics["best_f1"]

    return df


if __name__ == "__main__":
    results_path = Path("./results")
    df = read_score(results_path)
    print(df)
