import json
import replicate
from dotenv import load_dotenv
import re
import difflib
from openai_utils import get_embedding, user_prompt, \
    system_prompt, openai_json_response
import numpy as np
import random
from difflib import unified_diff

load_dotenv()

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


class BugInsertionModel:
    def __init__(self, model: str, library: BugLibrary):
        self.model = model
        self.library = library

    def __describe_program(self, program: str):
        raise NotImplementedError

    def insert_bug(self, question: str, solution: str, use_exemplars=False, k=5):
        raise NotImplementedError


class OpenAIBugInsertion(BugInsertionModel):
    def __init__(self, model: str, library: BugLibrary):
        super().__init__(model, library)

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
    
    def insert_bug(self, question: str, solution: str, use_exemplars=False, k=5):
        if not use_exemplars:
            @user_prompt
            def baseline_perturb(question, solution):
                return f'''Question:\n{question}\n\nSolution:\n{solution}\n\n
                Take the code above and return it having inserted a subtle bug.
                Return the result in JSON format with key "code" having the perturbed code.'''
            
            response = openai_json_response([
                baseline_perturb(question, solution)
            ], model=self.model, max_tokens=2048)

            return response["code"]
        
        else: # using top k exemplars
            program_description = self.__describe_program(solution)
            embedding = get_embedding(program_description)
            top_k_bugs = self.library.get_top_k_bugs(embedding, k=k)
            bugs_with_exemplar = list(map(lambda x: (x["bug_cluster"], random.choice(x["exemplars"])), top_k_bugs))
            exemplars = []
            for bug_exemplar in bugs_with_exemplar:
                exemplars.append(f'''Bug category: {bug_exemplar[0]}\nExample:\n{bug_exemplar[1]["diff"]}''')
            exemplars = "\n---------\n".join(exemplars)

            @user_prompt
            def exemplar_perturb(question, exemplars, solution):
                return f'''Question:\n{question}\nExamples of bugs:\n{exemplars}\nSolution:\n{solution}\n
                Take the code above and return it having inserted a subtle bug. Refer to the exemplars for 
                selecting and inserting a specific category of bug. Choose one type of bug and insert it.
                Look at a variety of lines and places in the code where you can insert one of the bug types.
                Return the result in JSON format with key "code" having the perturbed code. ***Return the
                code itself such that it can be executed, do not return a diff***.'''

            response = openai_json_response([
                exemplar_perturb(question, exemplars, solution)
            ], model=self.model, max_tokens=2048)

            return response["code"]


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

            # return metadata which includes the category of bug that was inserted + reasoning
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
    sample_path = "./dataset/train/729.json"

    with open(sample_path, "r") as fp:
        sample = json.load(fp)

    question = sample["question"]
    solution = sample["solutions"][0]

    bug_library = BugLibrary()
    bug_insertion_model = OpenAIBugInsertion("gpt-4o", bug_library)
    # bug_insertion_model = ReplicateBugInsertion("meta/meta-llama-3-8b-instruct", bug_library)

    baseline_perturbed_code = bug_insertion_model.insert_bug(question, solution, use_exemplars=False)
    exemplar_perturbed_code = bug_insertion_model.insert_bug(question, solution, use_exemplars=True)

    print(baseline_perturbed_code)
    print(get_modified_lines(solution, baseline_perturbed_code))

    print(exemplar_perturbed_code)
    print(get_modified_lines(solution, exemplar_perturbed_code))

    # code to execute on default test case

    

'''
issue with openai spacing?

        else:
            ans=[]
            for i in range(n):
                ans.append([0]*m)
            for i in range(n):
                for j in range(m):
                    if l[i][j]=='1':
                        ans[i][j]=0
                    else:
                        if (d[i] or e[j]):
                            ans[i][j]=1
                        else:
                            ans[i][j]=2
            for i in range(n):
                for j in range(m):
                    print(ans[i][j],end=" ")
-                print()
+                    print()

run_code()

[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]

^^ look at all these lines modified

'''


# 4. see if diversity increases (how?)

# by line numbers modified

# by edit distance from the original
# Compute the Levenshtein distance, Hamming distance, 
# or AST-based edit distance between each pair of perturbed versions. 
# Analyze the distribution of these distances (e.g., mean, standard deviation) to measure 
# how different the perturbations are.

# by diversity of inserted tokens (trying to retroactively cluster baseline into library categories)

# % successfully evading the provided test case for APPS

# 5. # insert bug only in those lines, see if diversity increases
# basically just select a range of lines, then describe and embed like before, and then direct
# bug insertion to happen to those lines only



# 6. see how often the base test case passes
