import os
import shutil
from datasets import load_dataset
import pandas as pd
import json
import re
import sys
import io
import threading
from openai_utils import system_prompt, user_prompt, openai_json_response
import concurrent.futures


def exec_with_mocked_io(code, inputs, timeout=1):
    original_stdin = sys.stdin
    original_stdout = sys.stdout
    
    input_mock = io.StringIO(inputs)
    output_capture = io.StringIO()
    
    sys.stdin = input_mock
    sys.stdout = output_capture
    
    exception_in_thread = None
    
    def execute_code():
        nonlocal exception_in_thread
        try:
            context = {}
            exec(code, context)
        except Exception as e:
            exception_in_thread = e 
    
    exec_thread = threading.Thread(target=execute_code)
    exec_thread.start()

    exec_thread.join(timeout)

    sys.stdin = original_stdin
    sys.stdout = original_stdout
    
    if exception_in_thread:
        raise exception_in_thread

    if exec_thread.is_alive():
        raise Exception(f"Code execution exceeded {timeout} seconds")

    output_capture.seek(0)
    return output_capture.read().strip()


def exec_func_with_args(code, inputs):
    local_scope = {}
    exec(code, globals(), local_scope)
    solution = local_scope['Solution']()
    result = solution.maxScore(*inputs) # name needs to change
    return result


def wrap_in_function(code: str, function_name: str = 'run_code'):
    indented_code = '\n'.join(['    ' + line for line in code.splitlines()])

    import_star_lines = []
    regular_lines = []
    for line in indented_code.split('\n'):
        if "import *" in line or "import*" in line:
            import_star_lines.append(line.strip())
        else:
            regular_lines.append(line)

    import_stars = '\n'.join(import_star_lines)
    regular_code = '\n'.join(regular_lines)
    
    wrapped_code = f'{import_stars}\ndef {function_name}():\n{regular_code}\n\n{function_name}()'
    
    return wrapped_code


def format_inputs(inputs):
    if isinstance(inputs, list):
        return '\n'.join(inputs)
    return inputs


def format_outputs(outputs, allow_multiple_answers, decimals=5):
    if isinstance(outputs, list):
        outputs = '\n'.join(outputs)
    outputs = outputs.strip().replace(" \n", "\n").replace("\n\n", "\n")

    if allow_multiple_answers:
        lines = outputs.split('\n')
        sorted_lines = []
        for line in lines:
            sorted_words = sorted(line.split())
            sorted_lines.append(' '.join(sorted_words))
        
        outputs = '\n'.join(sorted_lines)

    stripped_lines = [line.strip() for line in outputs.split('\n')]
    outputs = '\n'.join(stripped_lines)

    number_pattern = re.compile(r'(-?\d+\.?\d*)')

    def format_number(match):
        return f"{float(match.group()):.{decimals}f}"

    formatted_string = number_pattern.sub(format_number, outputs)
    return formatted_string


def has_multiple_answers(sample):
    question = sample["question"]

    @system_prompt
    def identify_if_multiple_answers():
        return '''You are given a coding question. The question might state that
        multiple answers are allowed or answers can be given in any order or
        if there are many answers you can print any of them. Return a JSON with key 
        `has_multiple_answers` with boolean value true if a statement like that
        (or similar) is present in the question or false if it is not present.'''
    
    @user_prompt
    def give_question(question):
        return question
    
    response = openai_json_response([
        identify_if_multiple_answers(),
        give_question(question)
    ])

    return response["has_multiple_answers"]


def build(folder_path, split, pct_io_len_cutoff=0.99, max_workers=8):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        
    os.makedirs(folder_path)

    data_path = os.path.join(folder_path, split)
    os.makedirs(data_path)

    ds = load_dataset("codeparrot/apps", split=split)
    samples = [sample for sample in ds]
    clean_samples = []

    allow_multiple_answers_path = f"./{split}_multiple_answers.json"

    if not os.path.exists(allow_multiple_answers_path):
        print("Identifying coding questions with multiple answers...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(has_multiple_answers, sample) for sample in samples]
            multiple_answers = [future.result() for future in concurrent.futures.as_completed(futures)]
            with open(allow_multiple_answers_path, "w") as fp:
                json.dump(multiple_answers, fp)

    with open(allow_multiple_answers_path, "r") as fp:
        multiple_answers = json.load(fp)

    df = pd.DataFrame(samples)
    lengths = df["input_output"].str.len()
    stats = lengths.describe(percentiles=[pct_io_len_cutoff])
    io_len_cutoff = int(stats[f"{int(pct_io_len_cutoff*100)}%"])
    
    num_funcs = 0
    non_funcs = 0
    for i, sample in enumerate(samples):
        print("Sample", i)
        if not sample["input_output"]:
            continue
        if len(str(sample["input_output"])) > io_len_cutoff:
            continue

        solutions = json.loads(sample["solutions"])
        input_output = json.loads(sample["input_output"])
        calls_func = True if input_output.get("fn_name") else False
        allow_multiple_answers = multiple_answers[i]

        if len(input_output["inputs"]) == 0:
            continue

        if calls_func:
            num_funcs += 1
        else:
            non_funcs += 1

        inputs = input_output["inputs"][0]
        outputs = input_output["outputs"][0]
        valid_solutions = []
        
        for j, solution in enumerate(solutions):
            if calls_func:
                exec_func_with_args(solution, inputs)
            else:
                continue
                outputs = format_outputs(outputs, allow_multiple_answers)
                try:
                    solution = wrap_in_function(solution)
                    inputs = format_inputs(inputs)
                    exec_outputs = exec_with_mocked_io(solution, inputs, timeout=2)
                    # print("EXEC", exec_outputs)
                    exec_outputs = format_outputs(exec_outputs, allow_multiple_answers)
                    if outputs == exec_outputs:
                        valid_solutions.append(solution)
                    else:
                        raise Exception("the outputs do not match")
                except Exception as e:
                    # if "the outputs" in str(e):
                    #     print("EXCEPTION")
                    #     print(sample)
                    #     print(allow_multiple_answers)
                    #     print(outputs)
                    #     print("=====================")
                    #     print(exec_outputs)
                    #     input()
                    pass
                    # if i == 77 and j == 16:
                    #     continue
                    # if i == 78 and j == 10:
                    #     continue
                    # if i == 508 and j == 4:
                    #     continue
                    # if i == 508 and j == 17:
                    #     continue
                    # if i == 508 and j == 39:
                    #     continue
                    # if i == 508 and j == 41:
                    #     continue
                    # if i == 508 and j == 78:
                    #     continue
                    # if i == 509 and j == 11:
                    #     continue
                    # if i == 509 and j == 62:
                    #     continue
                    # if i == 511 and j == 9:
                    #     continue
                    # if i == 511 and j == 12:
                    #     continue
                    # if i == 511 and j == 14:
                    #     continue
                    # if ("Code execution" in str(e)):
                    #     continue
                    # if ("starred expression" in str(e)):
                    #     continue
                    # if ("gcd" in str(e)):
                    #     continue
                    # if ("strip" in str(e)):
                    #     continue
                    # if ("initial_value" in str(e)):
                    #     continue
                    # if ("fileno" in str(e)):
                    #     continue
                    # if ("invalid literal" in str(e)):
                    #     continue
                    # if ("__future__" in str(e)):
                    #     continue
                    # print(f"Sample {i} | Solution {j}:", e)
                    # print(solution)
                    # print(outputs)
                    # print(exec_outputs)
                    # exit()
        
        print("CLEAN SAMPLES", len(clean_samples), "vs", num_funcs)
        if len(valid_solutions) > 0:
            clean_samples.append({})
        elif not calls_func:
            print(i, "has no valid solutions")
            print(sample)
    
    print(len(clean_samples), num_funcs)

if __name__ == "__main__":
    folder_path = "./dataset"
    split = "train"

    build(folder_path, split)
