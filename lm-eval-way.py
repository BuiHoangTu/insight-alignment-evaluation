import json
from pathlib import Path

import lm_eval
from lm_eval.tasks import TaskManager
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import handle_non_serializable


DEFAULT_MODEL_NAME = "../llama3_8b_lacomsa/checkpoint-94/"
DEFAULT_OUTPUT_PATH = "results"

LANGS = ["en", "de", "ru", "es", "fr", "th", "zh", "sw", "ja", "vi", "tr", "it"]
TASKS_BASE = ["mgsm_direct", "xcopa"]


def main(args):
    # Initialize model
    lm = HFLM(pretrained=args.model_name)  # type: ignore

    # Build task dictionary
    task_manager = TaskManager()

    # build tasks
    tasks = [f"{task}_{lang}" for task in TASKS_BASE for lang in args.langs]

    # verify tasks exist
    tasks = task_manager.match_tasks(tasks)

    # Run evaluation
    results = lm_eval.simple_evaluate(
        model=lm,
        task_manager=task_manager,
        tasks=tasks,
        apply_chat_template=True,
        model_args="parallelize=True",
        limit=100 if args.debug else None,
        # num_fewshot=2,
    )  # type: ignore

    (Path(args.output_path) / "results-lm-eval.json").write_text(
        json.dumps(results, default=handle_non_serializable, indent=2)
)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a model on multiple tasks.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Model name or path")
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH, help="Path to save results")
    parser.add_argument("--langs", nargs="+", default=LANGS, help="List of languages to evaluate")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with fewer examples")
    args = parser.parse_args()

    main(args)
