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
    result_data_url: Optional[str] = Field(None, description="URL to fetch result data as JSON when SUCCESS") # Renamed field
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
    original_filename: str = Field(..., description="Original name of the uploaded file")
    text_column: str = Field(..., description="Name of the column containing text data")
    hierarchy: Dict[str, Any] = Field(..., description="Nested hierarchy structure for classification")
    llm_config: LLMProviderConfig

# --- HF Training API Models ---

class HFTrainingRequest(BaseModel):
    training_file_id: str = Field(..., description="ID of the uploaded training data file")
    original_training_filename: str = Field(..., description="Original name of the training file")
    text_column: str = Field(..., description="Name of the column containing text data")
    hierarchy_columns: Dict[str, Optional[str]] = Field(..., description="Mapping of hierarchy levels (e.g., 'L1') to column names in the training data. Value can be None if level is unused.")
    base_model: str = Field(default=config.DEFAULT_HF_BASE_MODEL, description="Name of the base Hugging Face model to fine-tune")
    num_epochs: int = Field(default=config.DEFAULT_HF_EPOCHS, gt=0, le=10, description="Number of training epochs (1-10)")
    new_model_name: str = Field(..., description="Name to save the newly trained model under (e.g., 'my-custom-model-v1')")

# --- HF Rules API Models ---

class HFRule(BaseModel):
    """Represents a single rule for HF classification."""
    Label: str = Field(..., description="The label the rule applies to")
    Keywords: str = Field(..., description="Comma-separated keywords (or 'N/A')")
    Confidence_Threshold: float = Field(..., alias="Confidence Threshold", ge=0.05, le=0.95, description="Confidence threshold (0.05-0.95)")

    class Config:
        allow_population_by_field_name = True # Allow using "Confidence Threshold"

class HFRulesResponse(BaseModel):
    """Response model containing a list of HF rules."""
    rules: List[HFRule]

class HFRulesUpdateRequest(BaseModel):
    """Request model for updating HF rules."""
    rules: List[HFRule]

# --- HF Classification API Models ---

class HFClassificationRequest(BaseModel):
    """Request model for starting HF classification."""
    file_id: str = Field(..., description="ID of the uploaded file to classify")
    original_filename: str = Field(..., description="Original name of the uploaded file")
    text_column: str = Field(..., description="Name of the column containing text data")
    model_name: str = Field(..., description="Name of the saved HF model to use for classification")
