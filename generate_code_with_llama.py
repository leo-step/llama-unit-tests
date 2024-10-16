from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "meta-llama/Meta-Llama-3-8B-Instruct" 

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
def generate_code(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs['input_ids'],       # The tokenized prompt input
        max_length=max_length,     # Generate up to max_length tokens (input + output)
        num_return_sequences=1,    # Generate n sequences
        do_sample=True,            # Enable sampling for more diversity (instead of greedy decoding)
        temperature=0.7            # Slight randomness for more creative output
    )
    
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_code

prompt = "Write a Python function to reverse a string"
generated_code = generate_code(prompt)
print(generated_code)
