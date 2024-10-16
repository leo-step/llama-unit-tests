from datasets import load_dataset
import json

ds = load_dataset("codeparrot/apps", split="train")
sample = next(iter(ds))
# non-empty solutions and input_output features can be parsed from text format this way:
sample["solutions"] = json.loads(sample["solutions"])
sample["input_output"] = json.loads(sample["input_output"])
print(sample)
print(len(sample["solutions"]))
