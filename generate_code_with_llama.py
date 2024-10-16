from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "meta-llama/Llama-3.2-1B-Instruct" 

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model.to(device)

# Function to generate code
def generate_code(prompt, max_length=1024):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs['input_ids'],       # The tokenized prompt input
        max_length=max_length,     # Generate up to max_length tokens (input + output)
        num_return_sequences=1,    # Generate n sequences
        do_sample=True,            # Enable sampling for more diversity (instead of greedy decoding)
        temperature=0.7,           # Slight randomness for more creative output
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.encode("```")[0]
    )
    
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_code

prompt = """
You are a code generation assistant. Your task is to generate Python code snippets based on given instructions. Only output the code itself, without any explanations or additional text.

Instruction: Write a Python function to calculate the factorial of a number.

Code:
```python
"""

generated_code = generate_code(prompt)
print(generated_code)
