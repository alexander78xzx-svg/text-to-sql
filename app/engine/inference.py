import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = None
tokenizer = None
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_model(weights_filename: str = "trained_weights.pt"):
    global model, tokenizer
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    weights_path = os.path.join(root_dir, weights_filename)
    
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: {weights_filename} not found at {weights_path}. Using untuned GPT-2.")
        
    model.to(device)
    model.eval()

def generate_sql(prompt_string: str, n_candidates: int = 5) -> list[str]:
    inputs = tokenizer(prompt_string, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=48,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.4,
            num_return_sequences=n_candidates
        )

    candidates = []
    for i in range(n_candidates):
        gen_tokens = output_ids[i][prompt_len:]
        decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        candidates.append(decoded)

    return candidates