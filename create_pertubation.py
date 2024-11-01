import json
from dataset import exec_with_mocked_io
import random
import sys
import io
import difflib

sample_path = "./dataset/train/729.json"

with open(sample_path, "r") as fp:
    sample = json.load(fp)

num_test_cases = 20
input_generator = f'''{sample["input_generator"]}\n\ninput_generator({num_test_cases})'''
print(input_generator)

def capture_exec_output(code, context):
    # Create a StringIO object to capture the output
    output = io.StringIO()
    # Save the current stdout
    original_stdout = sys.stdout
    # Redirect stdout to the StringIO object
    sys.stdout = output
    try:
        exec(code, context)
    finally:
        # Restore the original stdout
        sys.stdout = original_stdout
    # Return the captured output
    return output.getvalue()

test_cases = capture_exec_output(input_generator, {})

output0 = exec_with_mocked_io(sample["solutions"][0], test_cases, timeout=10) # timeouts need to be raised as num_test_cases increases
# also maybe find some library that can execute python code safety so you dont have to write your own exec
output1 = exec_with_mocked_io(sample["solutions"][2], test_cases, timeout=10)

print(output0 == output1)
# initialize n test cases using input generator

# validate consistent outputs for test cases across all solutions

# take one solution
# ask for pertubation from llama and see that one test case fails

import replicate
from dotenv import load_dotenv
import re

load_dotenv()

prefill = "```"
stop_sequence = "```"

input = {
    "prompt": f'''{sample["solutions"][2]}\n\n\nTake the code above and return it having inserted a subtle bug. IMPORTANT: Do not add extra print statements and don't make trivial modifications to print statements.''',
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
code_p = matches[0]
# print(sample["solutions"][0])
# print(code_p)
print("==============")
output_p = exec_with_mocked_io(code_p, test_cases, timeout=5)

def print_diff(string1, string2):
    # Create a Differ object
    differ = difflib.Differ()
    # Use the compare method to compare the strings
    diff = differ.compare(string1.splitlines(), string2.splitlines())
    # Print the differences
    print('\n'.join(diff))

print_diff(sample["solutions"][2], code_p)

print(output0 == output_p)

