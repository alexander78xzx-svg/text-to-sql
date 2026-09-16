import sqlite3
from app.api.schemas import TableSchema, ForeignKey

def select_valid(tables: list[TableSchema], foreign_keys: list[ForeignKey], candidates: list[str]) -> tuple[str, bool]:
    ddl_statements = []
    
    for table in tables:
        pk_set = set(table.primary_keys)
        col_defs = [
            f"{col} PRIMARY KEY" if col in pk_set else col
            for col in table.columns
        ]
        
        table_fks = [fk for fk in foreign_keys if fk.source_table == table.name]
        for fk in table_fks:
            col_defs.append(f"FOREIGN KEY({fk.source_column}) REFERENCES {fk.target_table}({fk.target_column})")
            
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

    return candidates[0] if candidates else "", False

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