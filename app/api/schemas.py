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