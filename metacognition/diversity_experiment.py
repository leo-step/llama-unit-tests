import json
import replicate
from dotenv import load_dotenv
import re
import difflib
from difflib import unified_diff
from openai_utils import get_embedding, user_prompt, openai_json_response
from create_bug_exemplar_library import label_bug_with_reason, provide_bug
import numpy as np
import random
import pickle
from dataset import exec_with_mocked_io, format_outputs
import autopep8
import os
import random
from tqdm import tqdm
import pandas as pd

load_dotenv()

class BugLibrary:
    def __init__(self, path="metacognition/outputs/library.json"):
        with open(path, "r") as fp:
            library = json.load(fp)

        for data in library.values():
            data["embedding"] = np.array(data["embedding"])

        self.library = library

    def __getitem__(self, key):
        return self.library[f"{key}"]
    
    def __setitem__(self, key, value):
        self.library[f"{key}"] = value

    def get_top_k_bugs(self, query_vec, k):
        return sorted(list(self.library.values()), key=lambda x: np.dot(x["embedding"], query_vec))[-k:]


class BugInsertionModel:
    def __init__(self, model: str, library: BugLibrary):
        self.model = model
        self.library = library

    def __describe_program(self, program: str):
        raise NotImplementedError

    def insert_bug(self, question: str, solution: str, use_exemplars=False, k=5, n=3):
        raise NotImplementedError


class OpenAIBugInsertion(BugInsertionModel):
    def __init__(self, model: str, library: BugLibrary):
        super().__init__(model, library)

        vectorizer_path = "metacognition/outputs/tfidf_vectorizer.pkl"
        kmeans_path = "metacognition/outputs/kmeans_model.pkl"

        with open(vectorizer_path, "rb") as tfidf_file:
            self.vectorizer = pickle.load(tfidf_file)

        with open(kmeans_path, "rb") as kmeans_file:
            self.kmeans = pickle.load(kmeans_file)

    def __describe_program(self, program: str):
        @user_prompt
        def describe_program(program):
            return f'''{program}\n\nDescribe the program above, focusing on implementation details 
            and code concepts being used (e.g. arrays, loops, etc.). Be concise with your response.
            Return the result in JSON format with key "description" having the value of the description.'''
        
        response = openai_json_response([
            describe_program(program)
        ], model=self.model, max_tokens=2048) # what model do we want to use here?

        return response["description"]
    
    def insert_bug(self, question: str, solution: str, use_exemplars=False, k=5, n=3):
        if not use_exemplars:
            @user_prompt
            def baseline_perturb(question, solution):
                return f'''Question:\n{question}\n\nSolution:\n{solution}\n\n
                Take the code above and return it having inserted a subtle bug.
                Return the result in JSON format with key "code" having the perturbed code.'''
            
            response = openai_json_response([
                baseline_perturb(question, solution)
            ], model=self.model, max_tokens=2048)

            perturbed_code = response["code"]

            differ = difflib.Differ()
            diff = differ.compare(solution.splitlines(), perturbed_code.splitlines())
            diff_output = '\n'.join(diff)

            response = openai_json_response([
                label_bug_with_reason(),
                provide_bug(perturbed_code, diff_output)
            ], model="gpt-4o")

            tfidf_vector = self.vectorizer.transform([response["label"].replace('_', ' ')])
            tfidf_df = pd.DataFrame(tfidf_vector.toarray(), columns=self.vectorizer.get_feature_names_out())
            predicted_cluster_num = self.kmeans.predict(tfidf_df)[0]
            cluster_label = self.library[predicted_cluster_num]["cluster_label"]

            return perturbed_code, cluster_label
        
        else: # using top k exemplars
            program_description = self.__describe_program(solution)
            embedding = get_embedding(program_description)

            top_k_bugs = self.library.get_top_k_bugs(embedding, k=k)
            bug_choice = random.choice(top_k_bugs)
            exemplars = random.sample(bug_choice["exemplars"], n)
            exemplars = list(map(lambda x: x["diff"], exemplars))
            
            exemplars = "\n---------\n".join(exemplars)

            @user_prompt
            def exemplar_perturb(question, exemplars, solution):
                return f'''Question:\n{question}\nBug category:\n{bug_choice["cluster_label"]}
                Examples:\n{exemplars}\nSolution:\n{solution}\n
                Take the code above and return it having the provided category of bug. Refer to the examples for 
                what this category of bug looks like and how it is inserted. Consider a variety of lines and place
                the bug into the code in an appropriate place.
                Return the result in JSON format with key "code" having the perturbed code. 
                
                ***IMPORTANT: Do not return a diff! The code must be executable. The provided diff are example only***.'''

            response = openai_json_response([
                exemplar_perturb(question, exemplars, solution)
            ], model=self.model, max_tokens=2048)

            return response["code"], bug_choice["cluster_label"]


class ReplicateBugInsertion(BugInsertionModel):
    def __init__(self, model: str, library: BugLibrary):
        super().__init__(model, library)

    def __describe_program(self, program: str):
        input = {
            "prompt": f'''{program}\n\nDescribe the program above, focusing on implementation details 
            and code concepts being used (e.g. arrays, loops, etc.). Be concise with your response.''',
            "max_new_tokens": 2048,
            "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
            "stop_sequences": "<|end_of_text|>,<|eot_id|>"
        }

        response = replicate.run(
            self.model,
            input=input
        )
        response = ''.join(response)
        return response
    
    def insert_bug(self, question: str, solution: str, use_exemplars=False, k=5):
        if not use_exemplars: # baseline
            prefill = "```"
            stop_sequence = "```"

            input = {
                "prompt": f'''Question:\n{question}\n\nSolution:\n{solution}\n\n
                Take the code above and return it having inserted a subtle bug.''',
                "max_new_tokens": 2048,
                "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" + prefill,
                "stop_sequences": "<|end_of_text|>,<|eot_id|>," + stop_sequence
            }

            response = replicate.run(
                self.model,
                input=input
            )

            generated_code = prefill + ''.join(response) + stop_sequence

            pattern = r"```python\s*(.*?)\s*``"
            matches = re.findall(pattern, generated_code, re.DOTALL)
            perturbed_code = matches[0]
            return perturbed_code
        else: # using top k exemplars
            program_description = self.__describe_program(solution)
            embedding = get_embedding(program_description)
            top_k_bugs = self.library.get_top_k_bugs(embedding, k=k)
            bugs_with_exemplar = list(map(lambda x: (x["bug_cluster"], random.choice(x["exemplars"])), top_k_bugs))
            exemplars = []
            for bug_exemplar in bugs_with_exemplar:
                exemplars.append(f'''Bug category: {bug_exemplar[0]}\nExample:\n{bug_exemplar[1]["diff"]}''')
            exemplars = "\n---------\n".join(exemplars)

            prefill = "```"
            stop_sequence = "```"

            input = {
                "prompt": f'''Question:\n{question}\nExamples of bugs:\n{exemplars}\nSolution:\n{solution}\n
                Take the code above and return it having inserted a subtle bug. Refer to the exemplars for 
                selecting and inserting a specific category of bug. Choose one type of bug and insert it.
                Look at a variety of lines and places in the code where you can insert one of the bug types.''',
                "max_new_tokens": 2048,
                "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" + prefill,
                "stop_sequences": "<|end_of_text|>,<|eot_id|>," + stop_sequence
            }

            response = replicate.run(
                self.model,
                input=input
            )

            generated_code = prefill + ''.join(response) + stop_sequence

            pattern = r"```python\s*(.*?)\s*``"
            matches = re.findall(pattern, generated_code, re.DOTALL)
            perturbed_code = matches[0]
            return perturbed_code
        

def get_modified_lines(original_code, perturbed_code):
    original_lines = original_code.splitlines()
    perturbed_lines = perturbed_code.splitlines()
    
    diff = unified_diff(original_lines, perturbed_lines, lineterm='')
    
    modified_lines = []
    current_line_number = 0

    for line in diff:
        if line.startswith('@@'):
            hunk_info = line.split(' ')[2]
            start_line, line_count = (
                map(int, hunk_info[1:].split(',')) if ',' in hunk_info[1:] else (int(hunk_info[1:]), 1)
            )
            current_line_number = start_line - 1
        elif line.startswith('+') and not line.startswith('+++'):
            current_line_number += 1
            modified_lines.append(current_line_number)
        elif line.startswith('-') and not line.startswith('---'):
            continue
        elif not line.startswith('-') and not line.startswith('---'):
            current_line_number += 1

    return modified_lines


if __name__ == "__main__":
    directory = "./dataset/train"
    file_format = ".json"
    n_samples = 2
    n_insertions = 20
    n_attempts = 3

    matching_files = [os.path.join(directory, file) 
                      for file in os.listdir(directory) 
                      if file.endswith(file_format)]
    
    results = []

    for sample_path in random.sample(matching_files, n_samples):
        print(sample_path)

        with open(sample_path, "r") as fp:
            sample = json.load(fp)    

        question = sample["question"]

        index = random.randint(0, len(sample["solutions"]) - 1)
        print(index)
        solution = sample["solutions"][index]

        solution = autopep8.fix_code(solution)

        inputs = sample["inputs"]
        outputs = sample["outputs"]
        allow_multiple_answers = sample["has_multiple_answers"]

        bug_library = BugLibrary()
        bug_insertion_model = OpenAIBugInsertion("gpt-4o", bug_library)
        # bug_insertion_model = ReplicateBugInsertion("meta/meta-llama-3-8b-instruct", bug_library)

        for i in tqdm(range(n_insertions)):
            baseline_results = None
            exemplar_results = None

            for attempt in range(n_attempts):
                try:
                    baseline_perturbed_code, baseline_bug_category = bug_insertion_model.insert_bug(question, solution, use_exemplars=False)
                    baseline_perturbed_code = autopep8.fix_code(baseline_perturbed_code)
                    baseline_outputs = exec_with_mocked_io(baseline_perturbed_code, inputs, timeout=2)
                    baseline_passes = outputs == format_outputs(baseline_outputs, allow_multiple_answers)

                    baseline_results = {
                        "perturbed_code": baseline_perturbed_code,
                        "bug_category": baseline_bug_category,
                        "modified_lines": get_modified_lines(solution, baseline_perturbed_code),
                        "passes_test_case": baseline_passes
                    }
                    break
                except Exception as e:
                    print("Baseline retrying...", attempt)
                    print(e)
                    print(baseline_bug_category)
                    with open("metacognition/outputs/solution.py", 'w') as fp:
                        fp.write(solution)
                    with open("metacognition/outputs/error.py", 'w') as fp:
                        fp.write(baseline_perturbed_code)
                    print("Enter a character to continue...", end='')
                    input()
            
            if not baseline_results:
                continue
            
            for attempt in range(n_attempts):
                try:
                    exemplar_perturbed_code, exemplar_bug_category = bug_insertion_model.insert_bug(question, solution, use_exemplars=True, n=3)
                    exemplar_perturbed_code = autopep8.fix_code(exemplar_perturbed_code)
                    exemplar_outputs = exec_with_mocked_io(exemplar_perturbed_code, inputs, timeout=2)
                    exemplar_passes = outputs == format_outputs(exemplar_outputs, allow_multiple_answers)

                    exemplar_results = {
                        "perturbed_code": exemplar_perturbed_code,
                        "bug_category": exemplar_bug_category,
                        "modified_lines": get_modified_lines(solution, exemplar_perturbed_code),
                        "passes_test_case": exemplar_passes
                    }
                    break
                except Exception as e:
                    print("Exemplar retrying...", attempt)
                    print(e)
                    print(exemplar_bug_category)
                    with open("metacognition/outputs/solution.py", 'w') as fp:
                        fp.write(solution)
                    with open("metacognition/outputs/error.py", 'w') as fp:
                        fp.write(exemplar_perturbed_code)
                    print("Enter a character to continue...", end='')
                    input()

            if not exemplar_results:
                continue

            results.append({
                "baseline": baseline_results,
                "exemplar": exemplar_results,
                "sample_path": sample_path,
                "solution_index": index
            })

            # print("Passes APPS test case:", exemplar_passes)

    with open("metacognition/outputs/diversity_results.json", "w") as fp:
        json.dump(results, fp)
