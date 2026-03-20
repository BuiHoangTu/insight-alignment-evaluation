import json

from alpaca_eval.evaluator import Evaluator
from alpaca_eval.model import HuggingFaceModel
from datasets import load_dataset

DEFAULT_MODEL_O_NAME = "princeton-nlp/Llama-3-Base-8B-SFT-DPO"
DEFAULT_MODEL_X_NAME = "../llama3_8b_lacomsa/checkpoint-94/"
DEFAULT_NUM_PROMPTS = 100
DEFAULT_OUTPUT = "results/results-xalpaca.json"

DATASET_NAME = "viyer98/m-AlpacaEval"
SPLIT = "test"
DEBUG_NUM_PROMPTS = 10

def main(args):
    # LOAD DATASET
    dataset = load_dataset(DATASET_NAME, split=SPLIT)
    num_prompts = args.num_prompts
    if args.debug:
        num_prompts = min(num_prompts, DEBUG_NUM_PROMPTS)
    dataset = dataset.select(range(num_prompts))  # optional subsample

    # CREATE MODEL OBJECTS
    model_O = HuggingFaceModel(model_name=args.model_o_name)
    model_X = HuggingFaceModel(model_name=args.model_x_name)

    # CREATE EVALUATOR
    evaluator = Evaluator(
        models=[model_X, model_O],  # model_X vs Original
        dataset=dataset,
        judge_type="auto",  # use a judge?
    )

    # RUN EVALUATION
    results = evaluator.evaluate()
    
    # save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print("Evaluation results:", results)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate models on AlpacaEval leaderboard.")
    parser.add_argument("--model_o_name", type=str, default=DEFAULT_MODEL_O_NAME, help="Original model name")
    parser.add_argument("--model_x_name", type=str, default=DEFAULT_MODEL_X_NAME, help="Model X name or path")
    parser.add_argument("--num_prompts", type=int, default=DEFAULT_NUM_PROMPTS, help="Number of prompts to evaluate")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with fewer prompts")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output file for results")

    args = parser.parse_args()
    main(args)
