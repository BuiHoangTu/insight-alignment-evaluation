import json
from pathlib import Path
import pandas as pd
import numpy as np

LANG_32_MAP = {
    "eng": "en",
    "vie": "vi",
    "zho": "zh",
    "tur": "tr",
    "jpn": "ja",
    "tha": "th",
    "ita": "it",
    "rus": "ru",
    "spa": "es",
    "deu": "de",
    "fra": "fr",
}
LANG_NAME2_MAP = {
    "thai": "th",
    "english": "en",
    "german": "de",
    "russian": "ru",
    "spanish": "es",
    "chinese_simplified": "zh",
    "swahili": "sw",
    "french": "fr",
    "japanese": "ja",
    "vietnamese": "vi",
    "turkish": "tr",
    "italian": "it",
}

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

    ### Load lm-eval results
    j_lm_eval = json.loads((results_path / "results-lm-eval.json").read_text())
    result_lm_eval = j_lm_eval["results"]

    # Define languages and tasks
    LANGS = ["th", "en", "de", "ru", "es", "zh", "sw", "fr", "ja", "vi", "tr", "it"]
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

    ### Load light-eval results (mkqa)
    lighteval_path = results_path / "light-eval"

    # search for json files including subdirectories
    json_files = list(lighteval_path.rglob("*.json"))
    # iterate files, extract timestamp, file name schema follows results_2026-03-30T11-35-31.630689.json. select the latest file based on timestamp
    latest_file = max(json_files, key=lambda x: x.stem.split("_")[1])

    lighteval_json = json.loads(latest_file.read_text())
    light_results = lighteval_json["results"]["all"]

    df["mkqa"] = np.nan
    for key, value in light_results.items():
        if key.startswith("f1_") and not key.endswith("_stderr"):
            lang3 = key.split("_")[1]  # e.g., 'eng'
            lang2 = LANG_32_MAP.get(lang3)

            if lang2 in df.index:
                df.loc[lang2, "mkqa"] = value

    ### load xlsum
    xlsum_path = results_path / "results-xlsum.json"
    xlsum_json = json.loads(xlsum_path.read_text())

    df["xlsum"] = np.nan

    for lang_name, score in xlsum_json:
        lang_code = LANG_NAME2_MAP.get(lang_name)

        if lang_code in df.index:
            df.loc[lang_code, "xlsum"] = score

    return df


if __name__ == "__main__":
    results_path = Path("./results-org-model")
    df = read_score(results_path)
    print(df)
