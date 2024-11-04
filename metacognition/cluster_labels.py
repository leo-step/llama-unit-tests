import pandas as pd

output_path = "metacognition/outputs/bug_labels.csv"

df = pd.read_csv(output_path)

print(df.head())
print(len(df["label"].unique())) # 4329 unique, some duplicates

# from openai_utils import system_prompt, user_prompt, openai_json_response

# @system_prompt
# def cluster_labels():
#     return f'''You will be provided with a buggy piece of code along with
#     the diff of the bug fix that leads to a correct implementation. Label
#     this bug with an label that precisely describes the type of bug
#     that was present. You should be able to use the label as a dictionary
#     key in Python. The label should be lower case letters only. The 
#     label should be very descriptive and you may use multiple words to
#     describe the bug that occurred. If you do use multiple words for the label,
#     then join them with an underscore.
    
#     Your answer should be in JSON format where key "label" corresponds to your
#     naming of the bug and a second key "reason" contains a short justification
#     for why you chose the label.'''

# @user_prompt
# def provide_labels(code_tokens, diff_output):
#     return f'''Buggy solution:\n{code_tokens}\n\nBug fix:\n{diff_output}'''
