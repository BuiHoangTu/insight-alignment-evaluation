import json
import logging
from typing import Dict

import collections

from mkqa_eval.mkqa_eval import MKQA_LANGUAGES, MKQAPrediction, MKQAAnnotation


def read_annotations_from_hf(hf_mkqa_datasplit):
    """
    Read annotations from HuggingFace dataset split and return a dictionary of language to example_id to MKQAAnnotation.
    """
    
    all_gold_annotations = collections.defaultdict(dict)

    for example in hf_mkqa_datasplit:
        for language in MKQA_LANGUAGES:
            valid_answers, answer_types = [], []
            for answer in example["answers"][language]:
                # Binary (Yes/No) answer text is always "yes" / "no"
                # If answer['text'] is None then it `""` represents No Answer
                valid_answers.append(answer["text"] or "")
                valid_answers.extend(answer.get("aliases", []))
                answer_types.append(answer["type"])

            annotation = MKQAAnnotation(
                example_id=str(example["example_id"]),
                types=list(set(answer_types)),
                answers=list(set(valid_answers)),
            )

            all_gold_annotations[language][annotation.example_id] = annotation

    for lang, annotations in all_gold_annotations.items():
        if len(annotations) != 10000:
            logging.warning(
                f"The annotations file you've provided contains {len(all_gold_annotations)} for language {lang} examples, where 10000 are expected."
            )
    return all_gold_annotations


def save_predictions(output_path: str, predictions: Dict[str, MKQAPrediction]) -> None:
    """Write predictions to jsonl file compatible with read_predictions."""

    with open(output_path, "w+", encoding="utf-8") as f:
        for pred in predictions.values():
            record = {
                "example_id": str(pred.example_id),
                "prediction": pred.prediction or "",
                "binary_answer": pred.binary_answer if pred.binary_answer else "",
                "no_answer_prob": (
                    float(pred.no_answer_prob)
                    if pred.no_answer_prob is not None
                    else 0.0
                ),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
