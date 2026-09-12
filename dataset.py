from datasets import load_dataset
from transformers import AutoTokenizer
import json
import torch


train_data = load_dataset("xlangai/spider", split="train")

with open("tables.json", "r") as f:
    tables_data = json.load(f)

db_schemas = {}
for db in tables_data:
    db_id = db["db_id"]
    tables = db["table_names_original"]
    columns = db["column_names_original"]
    primary_keys = db.get("primary_keys", [])
    foreign_keys = db.get("foreign_keys", [])

    schema_map = {table_name: [] for table_name in tables}
    
    for i, col in enumerate(columns):
        table_idx, col_name = col[0], col[1]
        if table_idx == -1:
            continue # Skip *
        
        table_name = tables[table_idx]
        if i in primary_keys:
            col_name += " (PK)"
            
        schema_map[table_name].append((i, col_name))
        
    fk_strings = []
    for fk in foreign_keys:
        col1_idx, col2_idx = fk[0], fk[1]
        t1_idx, c1_name = columns[col1_idx]
        t2_idx, c2_name = columns[col2_idx]
        fk_strings.append(f"{tables[t1_idx]}.{c1_name} = {tables[t2_idx]}.{c2_name}")

    db_schemas[db_id] = {
        "tables": schema_map,
        "fks": fk_strings
    }

def format_schema_string(schema_data):
    table_strings = []
    for table, cols in schema_data["tables"].items():
        columns_str = " , ".join([c[1] for c in cols])
        table_strings.append(f"{table} : {columns_str}")
    
    schema_str = " | ".join(table_strings)
    
    if schema_data["fks"]:
        schema_str += " | [FK] " + " , ".join(schema_data["fks"])
        
    return schema_str

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
pad_id = tokenizer.eos_token_id
max_length = 512

input_ids_list = []
attention_mask_list = []
labels_list = []

for i in range(train_data.num_rows):
    sample = train_data[i]
    schema_str = format_schema_string(db_schemas[sample['db_id']])
    
    prompt_str = f"[SCHEMA] {schema_str} [QUESTION] {sample['question']} [SQL] "
    target_str = f"{sample['query']}<|endoftext|>"
    

    prompt_ids = tokenizer.encode(prompt_str)
    target_ids = tokenizer.encode(target_str)
    
    full_ids = prompt_ids + target_ids

    full_labels = [-100] * len(prompt_ids) + target_ids
    

    pad_len = max_length - len(full_ids)
    if pad_len > 0:
        input_ids_list.append(full_ids + [pad_id] * pad_len)
        attention_mask_list.append([1] * len(full_ids) + [0] * pad_len)
        labels_list.append(full_labels + [-100] * pad_len)
    else:
        input_ids_list.append(full_ids[:max_length])
        attention_mask_list.append([1] * max_length)
        labels_list.append(full_labels[:max_length])

input_ids = torch.tensor(input_ids_list)
attention_mask = torch.tensor(attention_mask_list)
labels = torch.tensor(labels_list)


torch.save({
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "labels": labels
}, "train_data.pt")