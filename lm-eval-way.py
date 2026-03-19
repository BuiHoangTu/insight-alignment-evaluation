DEFAULT_MODEL_NAME = "../llama3_8b_lacomsa/checkpoint-94/"
DEFAULT_OUTPUT_DIR = "../results"
# DEFAULT_TASKS = "mgsm:fr|1"

TASKS_PATH = "./lm-eval-tasks"


import json
from pathlib import Path

import lm_eval
from lm_eval.tasks import TaskManager, get_task_dict
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import handle_non_serializable


# Initialize model
lm = HFLM(pretrained=DEFAULT_MODEL_NAME)  # type: ignore

# Build task dictionary
task_manager = TaskManager(include_path=TASKS_PATH)

# Run evaluation
results = lm_eval.simple_evaluate(
    model=lm,
    # task_manager=task_manager,
    tasks=[
        "mgsm_direct",
        "xcopa",
    ],
    num_fewshot=2,
    limit=100,
)  # type: ignore

(Path(DEFAULT_OUTPUT_DIR) / "results.json").write_text(
    json.dumps(results, default=handle_non_serializable, indent=2)
)
