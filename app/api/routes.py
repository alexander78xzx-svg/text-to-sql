import asyncio
from fastapi import APIRouter
from app.api.schemas import Prompt
from app.engine.inference import generate_sql
from app.engine.validator import format_schema, select_valid

router = APIRouter()

@router.post("/generate")
async def gen(prompt: Prompt):
    schema_str = format_schema(prompt.tables, prompt.foreign_keys)
    prompt_string = f"[SCHEMA] {schema_str} [QUESTION] {prompt.question} [SQL] "
    
    candidates = await asyncio.to_thread(generate_sql, prompt_string, 20)
    chosen_sql, is_valid = await asyncio.to_thread(select_valid, prompt.tables, prompt.foreign_keys, candidates)

    return {"output": chosen_sql, "compiles": is_valid}