from dotenv import load_dotenv

load_dotenv()

# first look at the dataset and find way to read problems 
# and include bug diffs in nicely formatted way to feed into prompt
import json
import os

data_path = "metacognition/data/python/jsons"
file_name = "0.json"

file_path = os.path.join(data_path, file_name)
with open(file_path, "r") as fp:
    data = json.load(fp)

# print(data[0]) # len(data) == 10000

# each data point is a pair of solutions, one with a bug and the other with
# corresponding fix we want to select data points where lang == "python", 
# verdict == "Wrong Answer" or "Accepted"
def is_valid_pair(pair):
    return pair[0]["verdict"] == "Wrong Answer" and pair[1]["verdict"] == "Accepted"

data = list(filter(lambda pair: is_valid_pair(pair), data))

# print(len(data)) # len(data) == 5132
# we just filtered out the runtime / formatting errors and kept only logic / wrong answer ones

pair = data[0]
incorrect_submission, correct_submission = pair
# print(incorrect_submission['code_tokens'])
# print("\n------------\n")
# print(correct_submission['code_tokens'])
# print("\n------------\n")

import difflib

differ = difflib.Differ()
diff = differ.compare(incorrect_submission['code_tokens'].splitlines(), 
                      correct_submission['code_tokens'].splitlines())
diff_output = '\n'.join(diff)
# print(diff_output)

# initialize prompting to name the bug
from openai_utils import system_prompt, user_prompt, openai_json_response

@system_prompt
def label_bug_with_reason():
    return f'''You will be provided with a buggy piece of code along with
    the diff of the bug fix that leads to a correct implementation. Label
    this bug with an label that precisely describes the type of bug
    that was present. You should be able to use the label as a dictionary
    key in Python. The label should be lower case letters only. The 
    label should be very descriptive and you may use multiple words to
    describe the bug that occurred. If you do use multiple words for the label,
    then join them with an underscore.
    
    Your answer should be in JSON format where key "label" corresponds to your
    naming of the bug and a second key "reason" contains a short justification
    for why you chose the label.'''

@user_prompt
def provide_bug(code_tokens, diff_output):
    return f'''Buggy solution:\n{code_tokens}\n\nBug fix:\n{diff_output}'''

provide_bug_msg = provide_bug(incorrect_submission['code_tokens'], diff_output)

response = openai_json_response([
    label_bug_with_reason(),
    provide_bug_msg
], model="gpt-4o")

print(provide_bug_msg)
print("======= RESPONSE =======")
print(response)


# loop over all problem pairs and get bug diffs and label them with gpt
from mr import MapReduce
import pandas as pd

class LabelBugs(MapReduce):
    def get_items(self):
        data_path = "metacognition/data/python/jsons"
        file_name = "0.json" # we need to do this on every file!! $12 per file

        file_path = os.path.join(data_path, file_name)
        with open(file_path, "r") as fp:
            data = json.load(fp)

        def is_valid_pair(pair):
            return pair[0]["verdict"] == "Wrong Answer" and pair[1]["verdict"] == "Accepted"

        data = list(filter(lambda pair: is_valid_pair(pair), data))
        return data
    
    def mapF(self, item):
        incorrect_submission, correct_submission = item
        differ = difflib.Differ()
        diff = differ.compare(incorrect_submission['code_tokens'].splitlines(), 
                            correct_submission['code_tokens'].splitlines())
        diff_output = '\n'.join(diff)

        response = openai_json_response([
            label_bug_with_reason(),
            provide_bug(incorrect_submission['code_tokens'], diff_output)
        ], model="gpt-4o")

        return response
    
    def reduceF(self, results):
        output_path = "metacognition/outputs/bug_labels.csv"

        df = pd.DataFrame(results)
        print(df.head())
        
        df.to_csv(output_path)


job = LabelBugs()
job.run()

# cluster skills together using prompt

# save skills library + exemplars into json library
