from alpaca_eval.evaluator import Evaluator
from alpaca_eval.model import HuggingFaceModel
from datasets import load_dataset

# ----------------------------
# CONFIGURATION
# ----------------------------
model_O_name = "gpt2"
model_X_name = "your-username/model-X"

dataset_name = "viyer98/m-AlpacaEval"
split = "test"
num_prompts = 5

# ----------------------------
# LOAD DATASET
# ----------------------------
dataset = load_dataset(dataset_name, split=split)
dataset = dataset.select(range(num_prompts))  # optional subsample

# ----------------------------
# CREATE MODEL OBJECTS
# ----------------------------
model_O = HuggingFaceModel(model_name=model_O_name)
model_X = HuggingFaceModel(model_name=model_X_name)

# ----------------------------
# CREATE EVALUATOR
# ----------------------------
# AlpacaEval allows multiple models and can compare them automatically
evaluator = Evaluator(
    models=[model_X, model_O],  # model_X vs Original
    dataset=dataset,
    # You can specify a judge here, e.g., 'gpt-4' if you have API access
    judge_type="auto",
)

# ----------------------------
# RUN EVALUATION
# ----------------------------
results = evaluator.evaluate()
