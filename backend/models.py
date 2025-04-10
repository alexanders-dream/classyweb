# backend/models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from . import config

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

# --- LLM Config API Models ---

class ProviderListResponse(BaseModel):
    providers: List[str]

class FetchModelsRequest(BaseModel):
    provider: str
    endpoint: str
    api_key: Optional[str] = None

class ModelListResponse(BaseModel):
    models: List[str]

# --- Hierarchy Suggestion API Models ---

class HierarchySuggestRequest(BaseModel):
    sample_texts: List[str] = Field(..., min_items=1, max_items=config.MAX_LLM_SAMPLE_SIZE) # Use config value
    llm_config: LLMProviderConfig

class HierarchySuggestResponse(BaseModel):
    suggestion: Optional[Dict[str, Any]] = Field(None, description="Nested dictionary representing the suggested hierarchy")
    error: Optional[str] = Field(None, description="Error message if suggestion failed")

# --- LLM Classification API Models ---

class ClassifyLLMRequest(BaseModel):
    file_id: str = Field(..., description="ID of the uploaded file to classify")
    text_column: str = Field(..., description="Name of the column containing text data")
    hierarchy: Dict[str, Any] = Field(..., description="Nested hierarchy structure for classification")
    llm_config: LLMProviderConfig
