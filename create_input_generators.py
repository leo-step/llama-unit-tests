import os
import json
from openai_utils import system_prompt, user_prompt, openai_json_response
import random

folder_path = "./dataset"
split = "train"

data_path = os.path.join(folder_path, split)

def create_input_generator(sample):
    question = sample["question"]
    example_solution = sample["solutions"][0]
    example_input = sample["inputs"]

    @system_prompt
    def write_input_generator_function():
        return '''You will be given a competitive programming question, an
        working solutions, and an example input on which the solution can run on.
        Your job is to write an input generator function in Python. This function should 
        generate valid inputs for the question / solution given the constraints 
        of the problem. It needs to respect the input format as described and keep
        constraints within any provided ranges. This function can use the random
        package to generate the inputs (include import random before function definition). 
        The input generator must print the inputs
        in the right order with print() statements so that they can be consumed
        by the code. Your input function must be initialized as def input_generator(num_test_cases: int).
        Return only the code of the function with the function signature as a JSON
        with key "code" containing the function source code as a string.'''
    
    @user_prompt
    def give_question(question, example_solution, example_input):
        return f"Question: {question}\n\nExample Solution: {example_solution}\n\nExample Input: {example_input}"
    
    response = openai_json_response([
        write_input_generator_function(),
        give_question(question, example_solution, example_input)
    ])

    return response["code"]

# def call_code(code, num_test_cases):
#     exec(code + f"\n\ninput_generator({num_test_cases})")

# for filename in os.listdir(data_path):
#     if filename.endswith(".json"):
#         file_path = os.path.join(data_path, filename)
#         print(file_path)
#         with open(file_path, 'r', encoding='utf-8') as file:
#             sample = json.load(file)
#             code = create_input_generator(sample)


import os
import json
from concurrent.futures import ThreadPoolExecutor

def process_file(filename):
    print(filename)
    if filename.endswith(".json"):
        file_path = os.path.join(data_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            sample = json.load(file)
            # Assuming `create_input_generator` is a function you have defined
            sample['input_generator'] = create_input_generator(sample)
        
        # Write the modified content back to the same file
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(sample, file, ensure_ascii=False, indent=4)

# Execute in parallel using ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=16) as executor:
    filenames = [f for f in os.listdir(data_path) if f.endswith(".json")]
    executor.map(process_file, filenames)