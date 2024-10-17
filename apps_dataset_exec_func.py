from datasets import load_dataset
import json

ds = load_dataset("codeparrot/apps", split="train")

# some questions have no sample["input_output"]
# some questions have inputs that look weird like this: 'inputs': [['3 3 10', '4', '0', '3', '1', '6', '', '']]
# but its still read as it if its just newline separated so you should just do '\n'.join
# some have {'fn_name': 'maxScore', 'inputs': [[[1, 2, 3, 4, 5, 6, 1], 3]], 'outputs': [12]} e.g. sample 122

from typing import List, Dict, Tuple, Set, Any, Union, Optional, Callable

for i, sample in enumerate(ds):
    if i >= 1000:
        exit()
    # print(i)
    sample["solutions"] = json.loads(sample["solutions"])
    sample["input_output"] = json.loads(sample["input_output"])
    if sample["input_output"].get("fn_name"):
        print(i)
        print(sample["input_output"])
        # print(json.dumps(sample))
        def execute_code_with_scope(code, inputs):
            local_scope = {}
            exec(code, globals(), local_scope) # globals carries over global variables in current scope!!
            # Create an instance of the class
            solution = local_scope['Solution']()
            # Unpack the inputs and call the function
            result = solution.maxScore(*inputs)
            return result

        # Input example
        inputs = sample["input_output"]["inputs"][0]

        # Execute and print result
        solution_code = sample["solutions"][0]
        print(solution_code)
        result = execute_code_with_scope(solution_code, inputs)
        print(result)  # Output should be 12 based on the example
        exit()
    # print(sample)

    # for j, solution_code in enumerate(sample["solutions"]):
    #     inputs = '\n'.join(sample["input_output"]["inputs"][0])
    #     outputs = '\n'.join(sample["input_output"]["outputs"][0])
    #     print(inputs)
    #     print(outputs)
    # exit()


