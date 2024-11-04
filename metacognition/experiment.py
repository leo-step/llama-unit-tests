# create experiment to evaluate bug insertion diversity
# between usage of BugLibrary and base model when asked
# to insert a bug into code. Diversity can be evaluated as
# the line numbers which were modified, the type of bug
# inserted (can be identified via GPT-prompt), other metrics
# that I can ask ChatGPT about such as # chars changed and stuff.

# can see if diversity improves even more if I specify which line 
# numbers the bug needs to be inserted into specifically

# then make sure things aren't trivial in the sense that the
# default test cases don't always catch the inserted bug
# and the program actually runs most of the time
