from datasets import load_dataset
from huggingface_hub import login

login(token="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")

dataset = load_dataset('ControlNet/LAV-DF', num_proc=10)

print(dataset)
