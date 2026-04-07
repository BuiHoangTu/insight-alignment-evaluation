DEFAULT_MODEL_NAME = "../llama3_8b_lacomsa/checkpoint-94/"
DEFAULT_MODEL_TYPE = "transformer"
DEFAULT_OUTPUT_DIR = "../results"
DEFAULT_LANGS = [
    "tha",
    "eng",
    "deu",
    "rus",
    "spa",
    "zho",
    # "swa",
    "fra",
    "jpn",
    "vie",
    "tur",
    "ita",
]
DEFAULT_TASKS = [  # subset
    f"mkqa_{lang}:short_phrase" for lang in DEFAULT_LANGS
] + [  # full dataset
    f"mkqa_{lang}" for lang in DEFAULT_LANGS
]

def override_load_dataset_for_mkqa():
    from datasets import load_dataset

    original = load_dataset

    def patched_load_dataset(*args, **kwargs):
        kwargs.setdefault("trust_remote_code", True)
        return original(*args, **kwargs)

    import datasets

    datasets.load_dataset = patched_load_dataset

if __name__ == "__main__":
    override_load_dataset_for_mkqa()


from pathlib import Path

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.imports import is_package_available

# with launcher_type=ParallelismManager.ACCELERATE below, accelerate will be configed automatically,
# these lines of code are either for reconfiguring or is legacy.
if is_package_available("accelerate"):
    from datetime import timedelta
    from accelerate import Accelerator, InitProcessGroupKwargs
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
else:
    accelerator = None


def main(args):
    output_dir = Path(args.output_path) / "light-eval"
    evaluation_tracker = EvaluationTracker(
        output_dir=str(output_dir),
        save_details=True,
        push_to_hub=False,
        # hub_results_org="your_username",  # Replace with your actual username
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        load_tasks_multilingual=True,
        custom_tasks_directory=None,  # Set to path if using custom tasks
        # Remove the parameter below once your configuration is tested
        # max_samples=10000,
    )

    if args.model_type == "transformer":
        from lighteval.models.transformers.transformers_model import TransformersModelConfig

        model_config = TransformersModelConfig(
            model_name=args.model_name,
            dtype="bfloat16",
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    pipeline = Pipeline(
        tasks=",".join(args.tasks),
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()

if __name__ == "__main__":
    # use argparse to parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a language model on a set of tasks.")
    parser.add_argument("--model_type", type=str, default=DEFAULT_MODEL_TYPE, help="Type of model to evaluate (e.g., 'transformer', 'vllm').")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Model name or path to evaluate.")
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for results.")
    parser.add_argument("--tasks", nargs="+", type=str, default=DEFAULT_TASKS, help="Tasks to evaluate.")

    args = parser.parse_args()
    
    main(args)
