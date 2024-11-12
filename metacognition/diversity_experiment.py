import json
import replicate
from dotenv import load_dotenv
import re

load_dotenv()

# 1. take one program from apps
sample_path = "./dataset/train/729.json"

with open(sample_path, "r") as fp:
    sample = json.load(fp)

question = sample["question"]
solution = sample["solutions"][0]

print(question)
print("================")
print(solution)

# using replicate llama 3
# 2. use a baseline prompt to insert 100 bugs into it and save results
prefill = "```"
stop_sequence = "```"

input = {
    "prompt": f'''Question:\n{question}\n\nSolution:\n{solution}\n\n\nTake the code above and return it having inserted a subtle bug.''',
    "max_new_tokens": 512,
    "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" + prefill,
    "stop_sequences": "<|end_of_text|>,<|eot_id|>," + stop_sequence
}

response = replicate.run(
    "meta/meta-llama-3-8b-instruct",
    input=input
)

generated_code = prefill + ''.join(response) + stop_sequence

pattern = r"```python\s*(.*?)\s*``"
matches = re.findall(pattern, generated_code, re.DOTALL)
perturbed_code = matches[0]

print("================")
print(perturbed_code)

# 3. describe the program first and embed and then use bug library to insert 100 bugs and save results
# 4. see if diversity increases (how?)
# 5. describe the program line by line, then select range of lines, use that for the description
# insert bug only in those lines, see if diversity increases
# 6. see how often the base test case passes
