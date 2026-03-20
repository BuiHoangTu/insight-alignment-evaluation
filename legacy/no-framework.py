from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate import load

# Load dataset
dataset = load_dataset("GEM/xlsum", "english", split="test", trust_remote_code=True)

# load model
path = "../llama3_8b_lacomsa/checkpoint-94"
model = AutoModelForCausalLM.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)

# metric
rouge = load("rouge")


# Inference & evaluation
def evaluate_batch(batch):
    inputs = tokenizer(
        batch["document"], return_tensors="pt", truncation=True, max_length=1024
    )
    outputs = model.generate(**inputs, max_length=256, num_beams=4)
    pred_summary = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    rouge.add_batch(predictions=pred_summary, references=[batch["summary"]])

i = 0
for sample in dataset:
    evaluate_batch(sample)
    i += 1
    if i > 3:
        break  # test with 4 samples

result = rouge.compute()
print(result)
