from datasets import load_dataset
import json

ds = load_dataset("codeparrot/apps", split="train")

# some questions have no sample["input_output"]
# some questions have inputs that look weird like this: 'inputs': [['3 3 10', '4', '0', '3', '1', '6', '', '']]
# but its still read as it if its just newline separated so you should just do '\n'.join
# some have {'fn_name': 'maxScore', 'inputs': [[[1, 2, 3, 4, 5, 6, 1], 3]], 'outputs': [12]} e.g. sample 122


for i, sample in enumerate(ds):
    if i != 122:
        continue
    print(i)
    sample["solutions"] = json.loads(sample["solutions"])
    sample["input_output"] = json.loads(sample["input_output"])
    print(sample["input_output"])
    # print(sample)

    for j, solution_code in enumerate(sample["solutions"]):
        inputs = '\n'.join(sample["input_output"]["inputs"][0])
        outputs = '\n'.join(sample["input_output"]["outputs"][0])
    #     print(inputs)
    #     print(outputs)
    exit()


