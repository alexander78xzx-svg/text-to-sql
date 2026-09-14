import torch
import sqlite3
import json
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

val_data = load_dataset("xlangai/spider", split="validation")

with open("tables.json", "r") as f:
    tables_data = json.load(f)


db_schemas = {}
for db in tables_data:
    db_id, tables, columns = db["db_id"], db["table_names_original"], db["column_names_original"]
    primary_keys, foreign_keys = db.get("primary_keys", []), db.get("foreign_keys", [])
    
    schema_map = {table_name: [] for table_name in tables}
    for i, col in enumerate(columns):
        if col[0] == -1: continue
        table_name = tables[col[0]]
        col_name = col[1] + (" (PK)" if i in primary_keys else "")
        schema_map[table_name].append(col_name)
        
    fk_strings = [f"{tables[columns[fk[0]][0]]}.{columns[fk[0]][1]} = {tables[columns[fk[1]][0]]}.{columns[fk[1]][1]}" for fk in foreign_keys]
    
    table_strings = [f"{table} : {' , '.join(cols)}" for table, cols in schema_map.items()]
    schema_str = " | ".join(table_strings)
    if fk_strings: schema_str += " | [FK] " + " , ".join(fk_strings)
    
    db_schemas[db_id] = schema_str


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.load_state_dict(torch.load("trained_weights.pt", map_location="cpu"))
model.to(device)
model.eval()


total = 50 
valid_execution = 0
exact_match = 0
N = 5 # 5 candidates per question

print(f"\nEvaluating {total} queries...")
for i in tqdm(range(total)):
    sample = val_data[i]
    schema_str = db_schemas[sample['db_id']]
    gold_sql = sample['query']
    
    prompt = f"[SCHEMA] {schema_str} [QUESTION] {sample['question']} [SQL] "
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=48, 
            pad_token_id=tokenizer.eos_token_id, 
            do_sample=True,
            temperature=0.7,
            num_return_sequences=N
        )
        

    ddl_statements = ""
    for table in schema_str.split(" | "):
        if table.startswith("[FK]"): continue
        table_name, cols = table.split(" : ")
        col_defs = ", ".join([c.replace(" (PK)", " PRIMARY KEY") for c in cols.split(" , ")])
        ddl_statements += f"CREATE TABLE {table_name.strip()} ({col_defs});\n"
        
    conn = sqlite3.connect(":memory:")
    conn.executescript(ddl_statements)
    
    best_sql = None
    fallback_sql = None
    valid_found = False

    # check candidates
    for j in range(N):
        gen_tokens = output_ids[j][inputs.input_ids.shape[1]:]
        candidate_sql = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        
        if j == 0:
            fallback_sql = candidate_sql 
            
        try:
            
            conn.execute(f"EXPLAIN QUERY PLAN {candidate_sql}")
            best_sql = candidate_sql
            valid_found = True
            break 
        except sqlite3.OperationalError:
            continue
            
    conn.close()
    
    if valid_found:
        valid_execution += 1
    else:
        best_sql = fallback_sql #if all failed, use the first one
        
    if best_sql.lower() == gold_sql.lower():
        exact_match += 1

print("\n" + "="*50)
print("Evaluation results")
print("="*50)
print(f"Exact Match: {exact_match}/{total} ({(exact_match/total)*100:.1f}%)")
print(f"Syntactic Validity: {valid_execution}/{total} ({(valid_execution/total)*100:.1f}%)")
print("="*50)