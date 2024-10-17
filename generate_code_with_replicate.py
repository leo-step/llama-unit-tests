import replicate
from dotenv import load_dotenv
import re

load_dotenv()

prefill = "```"
stop_sequence = "```"

input = {
    "prompt": "Generate a python function to reverse a string. Include one test case where you call the function at the end within the same code block.",
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
print(matches[0])
print("==============")
exec(matches[0])
