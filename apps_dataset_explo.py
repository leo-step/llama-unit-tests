from datasets import load_dataset
import json
import sys
import io

ds = load_dataset("codeparrot/apps", split="train")

def exec_with_mocked_io(code, inputs):
    # Backup original stdin and stdout
    original_stdin = sys.stdin
    original_stdout = sys.stdout
    
    # Create a StringIO object to mock input and output
    input_mock = io.StringIO(inputs)
    output_capture = io.StringIO()
    
    # Replace stdin and stdout with the mocks
    sys.stdin = input_mock
    sys.stdout = output_capture
    
    try:
        # Execute the solution code
        context = {}
        exec(code, context)
        output_capture.seek(0)
        return output_capture.read().strip()
    
    finally:
        # Restore stdin and stdout
        sys.stdin = original_stdin
        sys.stdout = original_stdout


def wrap_in_function(code: str, function_name: str = 'run_code'):
    """
    Takes the existing code as a string and wraps it in a function
    to resolve scope issues for exec().
    
    Parameters:
    - code: The code to wrap as a string.
    - function_name: The name of the function to wrap the code in.
    
    Returns:
    - The wrapped code as a string.
    """
    # Split the input code into lines and ensure proper indentation
    indented_code = '\n'.join(['    ' + line for line in code.splitlines()])

    import_star_lines = []
    regular_lines = []
    for line in indented_code.split('\n'):
        if "import *" in line:
            import_star_lines.append(line.strip())
        else:
            regular_lines.append(line)

    import_stars = '\n'.join(import_star_lines)
    regular_code = '\n'.join(regular_lines)
    
    # Wrap the code in a function definition
    wrapped_code = f'{import_stars}\ndef {function_name}():\n{regular_code}\n\n{function_name}()'
    
    return wrapped_code


for i, sample in enumerate(ds):
    if i == 20:
        exit()
    print(f"========= SAMPLE {i} =========\n")
    # non-empty solutions and input_output features can be parsed from text format this way:
    sample["solutions"] = json.loads(sample["solutions"])
    sample["input_output"] = json.loads(sample["input_output"])
    # print(len(sample["solutions"]))

    for j, solution_code in enumerate(sample["solutions"]):
        print(f"--------- Solution {j} ---------")
        solution_code = wrap_in_function(solution_code)
        inputs = sample["input_output"]["inputs"][0] # all inputs across test cases stacked together
        outputs = sample["input_output"]["outputs"][0] # all outputs across test cases stacked together
        outputs = outputs.strip().replace(" \n", "\n").replace("\n\n", "\n")
        try:
            exec_outputs = exec_with_mocked_io(solution_code, inputs)
            exec_outputs = exec_outputs.strip().replace(" \n", "\n").replace("\n\n", "\n")

            if exec_outputs == outputs:
                print("The outputs match!")
            else:
                print("Expected output:", outputs)
                print("Actual output:", exec_outputs)
                raise Exception("The outputs do not match")
        except Exception as e:
            print(solution_code)
            print(e)
            print()
            continue
    print("\n")
