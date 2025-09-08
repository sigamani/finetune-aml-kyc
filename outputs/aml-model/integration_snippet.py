# Load fine-tuned AML/KYC model
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

# Load the final trained model
model = AutoPeftModelForCausalLM.from_pretrained("./outputs/aml-model/phase_hard")
tokenizer = AutoTokenizer.from_pretrained("unsloth/tinyllama-chat")

# Set up for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Example inference function
def analyze_aml_case(instruction: str, case_input: str) -> str:
    prompt = f"### Human:\n{instruction}\n{case_input}\n\n### Assistant:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.3,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Assistant:\n")[-1].strip()

