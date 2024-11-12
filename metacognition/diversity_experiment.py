import json
import replicate
from dotenv import load_dotenv
import re
import difflib
from openai_utils import get_embedding
import numpy as np
import random

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
# prefill = "```"
# stop_sequence = "```"

# input = {
#     "prompt": f'''Question:\n{question}\n\nSolution:\n{solution}\n\n\nTake the code above and return it having inserted a subtle bug.''',
#     "max_new_tokens": 512,
#     "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" + prefill,
#     "stop_sequences": "<|end_of_text|>,<|eot_id|>," + stop_sequence
# }

# response = replicate.run(
#     "meta/meta-llama-3-8b-instruct",
#     input=input
# )

# generated_code = prefill + ''.join(response) + stop_sequence

# pattern = r"```python\s*(.*?)\s*``"
# matches = re.findall(pattern, generated_code, re.DOTALL)
# perturbed_code = matches[0]

# print("================")
# print(perturbed_code)

# differ = difflib.Differ() # reversed order to seem as if bug is being introduced
# diff = differ.compare(solution.splitlines(), 
#                         perturbed_code.splitlines())
# diff_output = '\n'.join(diff)
# print("================")
# print(diff_output)

# code to execute on default test case

# 3. describe the program first and embed and then use bug library to insert 100 bugs and save results

input = {
    "prompt": f'''{solution}\n\nDescribe the program above, focusing on implementation details and code concepts being used (e.g. arrays, loops, etc.). Be concise with your response.''',
    "max_new_tokens": 512,
    "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    "stop_sequences": "<|end_of_text|>,<|eot_id|>"
}

response = replicate.run(
    "meta/meta-llama-3-8b-instruct",
    input=input
)
response = ''.join(response)
embedding = get_embedding(response)

class BugLibrary:
    def __init__(self, path="metacognition/outputs/library.json"):
        with open(path, "r") as fp:
            library = json.load(fp)

        for key, data in library.items():
            data["bug_cluster"] = key
            data["embedding"] = np.array(data["embedding"])

        self.library = library

    def get_top_k_bugs(self, query_vec, k):
        return sorted(list(self.library.values()), key=lambda x: np.dot(x["embedding"], query_vec))[-k:]

library = BugLibrary()
top_k_bugs = library.get_top_k_bugs(embedding, k=5)
bugs_with_exemplar = list(map(lambda x: (x["bug_cluster"], random.choice(x["exemplars"])), top_k_bugs))

print("================")
for bug_exemplar in bugs_with_exemplar:
    print(bug_exemplar[0])
    print("---------")
    print(bug_exemplar[1]["diff"])
    print("\n")

# 4. see if diversity increases (how?)
# 5. describe the program line by line, then select range of lines, use that for the description
# insert bug only in those lines, see if diversity increases
# 6. see how often the base test case passes
