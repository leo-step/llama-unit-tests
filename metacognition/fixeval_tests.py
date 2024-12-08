import json

with open("metacognition/data 2/python/processed/train.jsonl", "r") as fp:
    data = json.load(fp)

print(data[0])

'''
{'src_id': 'p00001_s631177546', 'src': ['from', 'sys', 'import', 'stdin', 'NEW_LINE', 
'x', '=', '[', 'int', '(', 'input', '(', ')', ')', 'for', 'i', 'in', 'range', '(', 
'10', ')', ']', 'NEW_LINE', 'x', '.', 'reverse', '(', ')', 'NEW_LINE', 'for', 'i', 
'in', 'range', '(', '3', ')', ':', 'NEW_LINE', 'INDENT', 'print', '(', 'i', ')', 'NEW_LINE', 
'DEDENT'], 'src_verdict': 'Wrong Answer', 'tgt': ['from', 'sys', 'import', 'stdin', 'NEW_LINE', 
'x', '=', '[', 'int', '(', 'input', '(', ')', ')', 'for', 'i', 'in', 'range', '(', '10', ')', ']', 
'NEW_LINE', 'x', '.', 'sort', '(', 'reverse', '=', 'True', ')', 'NEW_LINE', 'for', 'i', 'in', 'range', 
'(', '3', ')', ':', 'NEW_LINE', 'INDENT', 'print', '(', 'x', '[', 'i', ']', ')', 'NEW_LINE', 'DEDENT'], 
'tgt_id': 'p00001_s854661751'}
'''