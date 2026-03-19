import lighteval
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.imports import is_package_available


DEFAULT_MODEL_NAME = "../llama3_8b_lacomsa/checkpoint-94/"
DEFAULT_MODEL_TYPE = "transformer"
DEFAULT_OUTPUT_DIR = "../results"
DEFAULT_TASKS = "mgsm:fr|1"


# with launcher_type=ParallelismManager.ACCELERATE below, accelerate will be configed automatically,
# these lines of code are either for reconfiguring or is legacy.
if is_package_available("accelerate"):
    from datetime import timedelta
    from accelerate import Accelerator, InitProcessGroupKwargs
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
else:
    accelerator = None

def main(args):
    evaluation_tracker = EvaluationTracker(
        output_dir=args.output_dir,
        save_details=True,
        push_to_hub=False,
        # hub_results_org="your_username",  # Replace with your actual username
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        custom_tasks_directory=None,  # Set to path if using custom tasks
        # Remove the parameter below once your configuration is tested
        max_samples=20,
    )

    if args.model_type == "transformer":
        from lighteval.models.transformers.transformers_model import TransformersModelConfig

        model_config = TransformersModelConfig(
            model_name=args.model_name,
            dtype="float16",
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")


    pipeline = Pipeline(
        tasks=args.tasks,
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
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for results.")
    parser.add_argument("--tasks", type=str, default=DEFAULT_TASKS, help="Tasks to evaluate.")

    args = parser.parse_args()
    
    main(args)
