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
import queue
import copy

# needed to execute code
from typing import List, Dict, Tuple, Set, Any, Union, Optional, Callable
import math
from math import *
import collections
from collections import defaultdict, deque, Counter, OrderedDict
from collections import *
import heapq
from heapq import *
import functools
from functools import *
import itertools
from itertools import *
import operator
import sys
from sys import *
import queue
from queue import *
import bisect
from bisect import *


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

# TODO: make prints shut up by capturing stdout
def exec_func_with_args(code, func_name, inputs, timeout=1):
    exception_in_thread = None
    result_queue = queue.Queue()

    def execute_code():
        nonlocal exception_in_thread
        try:
            local_scope = {}
            exec(code, globals(), local_scope)
            solution = local_scope['Solution']()
            result = getattr(solution, func_name)(*inputs)
            result_queue.put(result)
        except Exception as e:
            exception_in_thread = e

    exec_thread = threading.Thread(target=execute_code)
    exec_thread.start()

    exec_thread.join(timeout)
    
    if exception_in_thread:
        raise exception_in_thread

    if exec_thread.is_alive():
        raise Exception(f"Code execution exceeded {timeout} seconds")

    if not result_queue.empty():
        return result_queue.get()

    return None


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
    outputs = str(outputs).strip().replace(" \n", "\n").replace("\n\n", "\n")

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
    
    for i, sample in enumerate(samples):
        print("Sample", i)
        if i == 160 or i == 378 or i == 379 or i == 438:
            continue
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

        inputs = input_output["inputs"][0]
        outputs = input_output["outputs"][0]
        valid_solutions = []
        
        for j, solution in enumerate(solutions):
            if calls_func:
                continue
                # try:
                #     outputs = format_outputs(outputs, allow_multiple_answers)
                #     func_name = input_output["fn_name"]
                #     exec_outputs = exec_func_with_args(solution, func_name, inputs, timeout=2)
                #     exec_outputs = format_outputs(exec_outputs, allow_multiple_answers)
                #     if outputs == exec_outputs:
                #         valid_solutions.append(solution)
                #     else:
                #         raise Exception("the outputs do not match")
                # except Exception as e:
                #     pass
                    # if "the outputs" in str(e):
                    #     print(solution)
                    #     print(outputs)
                    #     print(exec_outputs)
                    #     print(e)
                        # input()
            else:
                try:
                    outputs = format_outputs(outputs, allow_multiple_answers)
                    solution = wrap_in_function(solution)
                    inputs = format_inputs(inputs)
                    exec_outputs = exec_with_mocked_io(solution, inputs, timeout=2)
                    exec_outputs = format_outputs(exec_outputs, allow_multiple_answers)
                    if outputs == exec_outputs:
                        valid_solutions.append({
                            "solution": solution,
                            "inputs": inputs,
                            "outputs": outputs
                        })
                    else:
                        raise Exception("the outputs do not match")
                except Exception as e:
                    pass

        if len(valid_solutions) > 0:
            inputs = []
            outputs = []
            solutions = []

            for data in valid_solutions:
                inputs.append(data["inputs"])
                outputs.append(data["outputs"])
                solutions.append(data["solution"])

            with open(os.path.join(data_path, f"{i}.json"), "w") as fp:
                json.dump({
                    "question": sample["question"],
                    "solutions": solutions,
                    "inputs": inputs[0],
                    "outputs": outputs[0],
                    "has_multiple_answers": allow_multiple_answers,
                    "difficulty": sample["difficulty"]
                }, fp)

if __name__ == "__main__":
    folder_path = "./dataset"
    split = "train"

    build(folder_path, split)
