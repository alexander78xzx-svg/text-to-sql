from datasets import load_dataset
from transformers import AutoTokenizer
import json
import torch


train_data = load_dataset("xlangai/spider", split="train")
test_data = load_dataset("xlangai/spider", split="validation")
with open("tables.json", "r") as f:
    tables_data = json.load(f)

 
db_schemas = {}
for db in tables_data:
    db_id = db["db_id"]
    tables = db["table_names_original"]
    columns = db["column_names_original"]

    schema_map = {table_name: [] for table_name in tables}
    for col in columns:
        table_idx = col[0]
        col_name = col[1]
        if table_idx == -1:
            continue
        table_name = tables[table_idx]
        schema_map[table_name].append(col[1])

        pass
    db_schemas[db_id] = schema_map

def format_schema_string(schema_map):
    table_strings = []
    
    for table, cols in schema_map.items():
        columns_str = " , ".join(cols)
        table_strings.append(f"{table} : {columns_str}")
    return " | ".join(table_strings)

data = []
for i in range(train_data.num_rows) :
    sample = train_data[i]
    data.append( f"[SCHEMA] {format_schema_string(db_schemas[sample['db_id']])} [QUESTION] {sample['question']} [SQL] {sample['query']}<|endoftext|>" )



tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

tokens = tokenizer(
    data,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

input_ids = tokens["input_ids"]
attention_mask = tokens["attention_mask"]

labels = input_ids.clone()
labels[attention_mask == 0] = -100

for i, text in enumerate(data):
    prompt_str = text.split("[SQL]")[0] + "[SQL]"
    prompt_len = len(tokenizer.encode(prompt_str))
    
    labels[i, :prompt_len] = -100
    
torch.save({
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "labels": labels
}, "train_data.pt")