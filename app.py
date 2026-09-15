from fastapi import FastAPI
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import sqlite3
from pydantic import BaseModel, Field


class ForeignKey(BaseModel):
    source_table: str
    source_column: str
    target_table: str
    target_column: str

class TableSchema(BaseModel):
    name: str
    columns: list[str]
    primary_keys: list[str] = Field(default_factory=list)

class Prompt(BaseModel):
    tables: list[TableSchema]
    foreign_keys: list[ForeignKey] = Field(default_factory=list)
    question: str

app = FastAPI()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


model = GPT2LMHeadModel.from_pretrained("gpt2")
state_dict = torch.load("trained_weights.pt", map_location="cpu")
model.load_state_dict(state_dict)

model.to(device)
model.eval()

def generate(prompt: str, n_candidates: int = 5) -> list[str]:
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=48,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            num_return_sequences=n_candidates
        )


    candidates = []
    for i in range(n_candidates):
        gen_tokens = output_ids[i][prompt_len:]
        decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        candidates.append(decoded)

    return candidates

import sqlite3

def select_valid(tables: list[TableSchema], candidates: list[str]) -> tuple[str, bool]:
    ddl_statements = []
    for table in tables:
        pk_set = set(table.primary_keys)
        col_defs = [
            f"{col} PRIMARY KEY" if col in pk_set else col
            for col in table.columns
        ]
        ddl_statements.append(f"CREATE TABLE {table.name} ({', '.join(col_defs)});")

    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    try:
        cursor.executescript("\n".join(ddl_statements))
        for cand in candidates:
            try:
                cursor.execute(f"EXPLAIN QUERY PLAN {cand}")
                return cand, True
            except sqlite3.OperationalError:
                continue
    finally:
        conn.close()

    return candidates[0], False

def format_schema(tables: list[TableSchema], foreign_keys: list[ForeignKey]) -> str:
    table_segments = []

    for table in tables:
        pk_set = set(table.primary_keys)
        formatted_cols = [
            f"{col} (PK)" if col in pk_set else col 
            for col in table.columns
        ]
        table_segments.append(f"{table.name} : {' , '.join(formatted_cols)}")

    schema_str = " | ".join(table_segments)

    if foreign_keys:
        fk_segments = [
            f"{fk.source_table}.{fk.source_column} = {fk.target_table}.{fk.target_column}"
            for fk in foreign_keys
        ]
        schema_str += f" | [FK] {' , '.join(fk_segments)}"

    return schema_str

@app.post("/generate")
def gen(prompt: Prompt):
    schema = format_schema(prompt.tables, prompt.foreign_keys)

    prompt_string = f"[SCHEMA] {schema} [QUESTION] {prompt.question} [SQL] "
    candidates = generate(prompt_string, n_candidates=5)

    chosen_sql, is_valid = select_valid(schema, candidates)

    return {"output": chosen_sql, "compiles": is_valid}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
