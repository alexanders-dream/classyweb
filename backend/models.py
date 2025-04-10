# backend/models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class FileInfo(BaseModel):
    file_id: str
    filename: str
    columns: List[str]
    num_rows: int
    preview: List[Dict[str, Any]] = Field(..., description="List of dicts representing head(5) of the DataFrame")

class LLMProviderConfig(BaseModel):
    provider: str = Field(..., description="LLM Provider (e.g., 'Groq', 'Ollama')")
    endpoint: str = Field(..., description="API endpoint URL")
    model_name: str = Field(..., description="Specific model name to use")
    api_key: Optional[str] = Field(None, description="API key, if required by the provider")

class TaskStatus(BaseModel):
    task_id: str
    status: str = Field(..., description="e.g., PENDING, RUNNING, SUCCESS, FAILED")
    message: Optional[str] = None
    result_url: Optional[str] = Field(None, description="URL to download results when SUCCESS")
    # Add more fields later, like progress percentage if desired