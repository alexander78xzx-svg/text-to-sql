from datasets import load_dataset
from transformers import AutoTokenizer

train_data = load_dataset("xlangai/spider", split="train")
test_data = load_dataset("xlangai/spider", split="validation")

data = []
for i in range(train_data.num_rows) :
    sample = train_data[i]
    data.append( f"[SCHEMA] {sample['db_id']} [QUESTION] {sample['question']} [SQL] {sample['query']}<|endoftext|>" )


tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

tokens = tokenizer(
    data,
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors="pt"
)

