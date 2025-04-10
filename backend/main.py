# backend/main.py
import os
import logging
from fastapi import ( # Organize imports
    FastAPI, UploadFile, File, HTTPException, Depends, Body, BackgroundTasks, Path as FastApiPath
)
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path
import uuid
import shutil
import traceback # Added traceback
from fastapi.middleware.cors import CORSMiddleware
import requests # Need this for exception handling in get_llm_models
from enum import Enum # Added Enum

# Assuming config, utils, models are importable
try:
    from . import config, utils, llm_classifier # Added llm_classifier
    # Import new models
    from .models import (
        FileInfo, ProviderListResponse, FetchModelsRequest, ModelListResponse,
        HierarchySuggestRequest, HierarchySuggestResponse,
        ClassifyLLMRequest, TaskStatus # Added classification models
    )
except ImportError:
    from . import config, utils, llm_classifier # Fallback for direct script run
    # Import new models
    from models import (
        FileInfo, ProviderListResponse, FetchModelsRequest, ModelListResponse,
        HierarchySuggestRequest, HierarchySuggestResponse,
        ClassifyLLMRequest, TaskStatus # Added classification models
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Text Classifier API",
    description="API backend for the AI Text Classifier application.",
    version="0.1.0"
)

# --- CORS Configuration ---
# Define the list of origins allowed to connect (your frontend URL)
# Use '*' for development ONLY if absolutely necessary, be specific for production
origins = [
    "http://localhost:5173", # Your frontend's origin
    "http://127.0.0.1:5173", # Sometimes needed depending on how browser resolves localhost
    # Add your production frontend URL here later:
    # "https://your-frontend-domain.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # List of allowed origins
    allow_credentials=True, # Allow cookies if needed (may not be needed now)
    allow_methods=["*"],    # Allow all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],    # Allow all headers
)

# --- Configuration & Task Management ---
UPLOAD_DIR = Path("./temp_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Temporary upload directory: {UPLOAD_DIR.resolve()}")

# Simple in-memory task store (Replace with Redis/Celery for production)
tasks_db: Dict[str, Dict[str, Any]] = {}

class TaskStatusEnum(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"

# --- Background Task Function ---
def run_llm_classification_task(
    task_id: str,
    file_id: str,
    text_column: str,
    hierarchy: Dict[str, Any],
    llm_config_dict: Dict[str, Any] # Pass config as dict
):
    """Background task to perform LLM classification."""
    logger.info(f"Starting background task {task_id} for file {file_id}")
    tasks_db[task_id]["status"] = TaskStatusEnum.RUNNING
    tasks_db[task_id]["message"] = "Initializing LLM client..."

    llm_client = None
    input_file_path = UPLOAD_DIR / file_id
    result_file_path: Optional[Path] = None

    try:
        # 1. Initialize LLM Client
        llm_client = llm_classifier.initialize_llm_client(
            provider=llm_config_dict["provider"],
            endpoint=llm_config_dict["endpoint"],
            api_key=llm_config_dict.get("api_key"), # Use .get for optional key
            model_name=llm_config_dict["model_name"]
        )
        tasks_db[task_id]["message"] = "LLM client initialized. Loading data..."

        # 2. Load Data
        if not input_file_path.exists():
            raise FileNotFoundError(f"Input file not found for file_id: {file_id}")

        # Assuming original filename isn't strictly needed here, pass None or fetch if required
        df_input = utils.load_data(file_path=input_file_path, original_filename=f"{file_id}_data")
        if df_input is None:
            raise ValueError(f"Failed to load data from file: {input_file_path}")

        if text_column not in df_input.columns:
             raise ValueError(f"Text column '{text_column}' not found in the input file.")

        tasks_db[task_id]["message"] = f"Data loaded ({len(df_input)} rows). Starting classification..."

        # 3. Run Classification
        df_results = llm_classifier.classify_texts_with_llm(
            df=df_input,
            text_column=text_column,
            hierarchy_dict=hierarchy,
            llm_client=llm_client
            # Add batch_size, max_concurrency if needed/configurable
        )

        if df_results is None:
            raise RuntimeError("LLM classification function returned None.")

        tasks_db[task_id]["message"] = "Classification complete. Saving results..."

        # 4. Save Results
        result_filename = f"{task_id}_results.xlsx" # Save as Excel for easy download
        result_file_path = UPLOAD_DIR / result_filename
        excel_bytes = utils.df_to_excel_bytes(df_results)
        if not excel_bytes:
             raise RuntimeError("Failed to convert results DataFrame to Excel bytes.")

        with open(result_file_path, "wb") as f:
            f.write(excel_bytes)

        logger.info(f"Task {task_id}: Results saved to {result_file_path}")

        # 5. Update Task Status - Success
        tasks_db[task_id]["status"] = TaskStatusEnum.SUCCESS
        tasks_db[task_id]["message"] = "Classification completed successfully."
        tasks_db[task_id]["result_path"] = str(result_file_path) # Store path as string

    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}", exc_info=True)
        tasks_db[task_id]["status"] = TaskStatusEnum.FAILED
        tasks_db[task_id]["message"] = f"Error: {e}"
        tasks_db[task_id]["result_path"] = None
        # Clean up partial result file if it exists
        if result_file_path and result_file_path.exists():
            try:
                os.remove(result_file_path)
                logger.info(f"Task {task_id}: Cleaned up partial result file {result_file_path}")
            except Exception as cleanup_e:
                logger.error(f"Task {task_id}: Failed to cleanup partial result file {result_file_path}: {cleanup_e}")


# --- API Endpoints ---

@app.get("/", tags=["Status"])
async def read_root():
    """Root endpoint to check if the API is running."""
    logger.info("Root endpoint accessed.")
    return {"message": "AI Text Classifier API is running!"}

@app.post("/data/upload", response_model=FileInfo, tags=["Data Handling"])
async def upload_data(file: UploadFile = File(...)):
    """
    Uploads a CSV or Excel file, saves it temporarily, loads it into a DataFrame,
    and returns file metadata including a preview.
    """
    # Generate a unique ID for this upload
    file_id = str(uuid.uuid4())
    # Define the path where the file will be saved temporarily
    file_location = UPLOAD_DIR / file_id
    original_filename = file.filename or "unknown_file"

    logger.info(f"Received upload request for '{original_filename}'. Assigning ID: {file_id}")

    try:
        # Save the uploaded file to the temporary location
        logger.info(f"Saving uploaded file to: {file_location}")
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        logger.info(f"Successfully saved '{original_filename}' to {file_location}")

        # Process the saved file using the refactored utility function
        # Pass both the saved file path AND the original filename
        df = utils.load_data(file_path=file_location, original_filename=original_filename)
        # ============================

        if df is None:
            logger.error(f"Failed to load DataFrame from uploaded file: {file_location} (Original: '{original_filename}')")
            if file_location.exists():
                os.remove(file_location)
            raise HTTPException(status_code=400, detail=f"Could not process uploaded file '{original_filename}'. Unsupported format or file error.")
        # Prepare the preview data
        preview_records = df.head(5).to_dict('records') # Get first 5 rows as list of dicts

        # Prepare the response model
        file_info = FileInfo(
            file_id=file_id,
            filename=original_filename,
            columns=df.columns.tolist(), # Get column names as list of strings
            num_rows=len(df),
            preview=preview_records
        )
        logger.info(f"Successfully processed file ID {file_id}. Rows: {len(df)}, Cols: {len(df.columns)}")
        return file_info

    except HTTPException as http_exc:
         # Re-raise HTTPExceptions directly
         raise http_exc
    except Exception as e:
        logger.error(f"Error during file upload processing for '{original_filename}': {e}", exc_info=True)
        # Clean up the potentially corrupted file
        if file_location.exists():
            try:
                os.remove(file_location)
                logger.info(f"Cleaned up failed upload file: {file_location}")
            except Exception as cleanup_e:
                 logger.error(f"Failed to cleanup upload file {file_location}: {cleanup_e}")
        # Return a generic server error
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during file upload: {e}")
    finally:
         # Ensure the uploaded file handle is closed
         if hasattr(file, 'file') and hasattr(file.file, 'close'):
              file.file.close()

# --- LLM Configuration Endpoints ---

@app.get("/llm/providers", response_model=ProviderListResponse, tags=["LLM Configuration"])
async def get_llm_providers():
    """Returns the list of supported LLM providers."""
    logger.info("Request received for /llm/providers")
    return ProviderListResponse(providers=config.SUPPORTED_PROVIDERS)

@app.post("/llm/models", response_model=ModelListResponse, tags=["LLM Configuration"])
async def get_llm_models(request: FetchModelsRequest = Body(...)):
    """
    Fetches the list of available models for a given LLM provider, endpoint, and API key.
    """
    logger.info(f"Request received for /llm/models for provider: {request.provider}")
    try:
        models = llm_classifier.fetch_available_models(
            provider=request.provider,
            endpoint=request.endpoint,
            api_key=request.api_key
        )
        if not models:
            logger.warning(f"No models found or error fetching models for {request.provider} at {request.endpoint}")
            # Return empty list, frontend should handle this
            return ModelListResponse(models=[])
        logger.info(f"Successfully fetched {len(models)} models for {request.provider}")
        return ModelListResponse(models=models)
    except Exception as e:
        logger.error(f"Error fetching LLM models for {request.provider}: {e}", exc_info=True)
        # Return a more specific error if possible, otherwise a generic 500
        if isinstance(e, requests.exceptions.ConnectionError):
             raise HTTPException(status_code=503, detail=f"Could not connect to endpoint: {request.endpoint}")
        elif isinstance(e, requests.exceptions.HTTPError):
             status_code = e.response.status_code
             detail = f"HTTP error from provider API: {status_code} - {e.response.reason}"
             if status_code == 401:
                 detail = "Authentication failed. Check API Key."
             raise HTTPException(status_code=status_code if status_code >= 400 else 500, detail=detail)
        else:
              raise HTTPException(status_code=500, detail=f"An unexpected error occurred while fetching models: {e}")

# --- LLM Hierarchy Suggestion Endpoint ---

@app.post("/llm/hierarchy/suggest", response_model=HierarchySuggestResponse, tags=["LLM Hierarchy"])
async def suggest_hierarchy(request: HierarchySuggestRequest = Body(...)):
    """
    Generates a hierarchy suggestion based on sample texts using the configured LLM.
    """
    logger.info(f"Request received for /llm/hierarchy/suggest using {request.llm_config.provider} model {request.llm_config.model_name}")
    llm_client = None

    # 1. Initialize LLM Client
    try:
        llm_client = llm_classifier.initialize_llm_client(
            provider=request.llm_config.provider,
            endpoint=request.llm_config.endpoint,
            api_key=request.llm_config.api_key,
            model_name=request.llm_config.model_name
        )
    except ValueError as ve:
        logger.error(f"LLM Configuration Error for suggestion: {ve}")
        raise HTTPException(status_code=400, detail=f"LLM Configuration Error: {ve}")
    except requests.exceptions.ConnectionError as ce:
        logger.error(f"LLM Connection Error during initialization: {ce}")
        raise HTTPException(status_code=503, detail=f"Could not connect to LLM endpoint: {request.llm_config.endpoint}")
    except Exception as e_init:
        logger.error(f"Unexpected error initializing LLM client for suggestion: {e_init}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Failed to initialize LLM client: {e_init}")
    else:
        # 2. Generate Suggestion (only if client initialization succeeded)
        try:
            suggestion = llm_classifier.generate_hierarchy_suggestion(
                 llm_client=llm_client,
                 sample_texts=request.sample_texts
            )

            if suggestion:
                logger.info("Successfully generated hierarchy suggestion.")
                return HierarchySuggestResponse(suggestion=suggestion, error=None)
            else:
                logger.warning("LLM failed to generate a valid hierarchy suggestion.")
                # Return 200 OK but with an error message in the response body
                return HierarchySuggestResponse(suggestion=None, error="LLM failed to generate a valid suggestion. The response might have been empty or incorrectly formatted.")
        except Exception as e_suggest:
            logger.error(f"Error during hierarchy suggestion generation: {e_suggest}", exc_info=True)
            raise HTTPException(status_code=500, detail="An unexpected error occurred during suggestion generation.")

# --- LLM Classification Task Endpoints ---

@app.post("/classify/llm", response_model=TaskStatus, status_code=202, tags=["Classification Tasks"])
async def start_llm_classification(
    request: ClassifyLLMRequest,
    background_tasks: BackgroundTasks
):
    """
    Starts a background task for LLM classification.
    """
    task_id = str(uuid.uuid4())
    logger.info(f"Received LLM classification request. Assigning Task ID: {task_id}")

    # Validate input file exists (basic check)
    input_file_path = UPLOAD_DIR / request.file_id
    if not input_file_path.exists():
        logger.error(f"Input file not found for file_id: {request.file_id}")
        raise HTTPException(status_code=404, detail=f"Input file with ID '{request.file_id}' not found.")

    # Store initial task status
    tasks_db[task_id] = {
        "status": TaskStatusEnum.PENDING,
        "message": "Task received, pending execution.",
        "result_path": None
    }

    # Add the background task
    background_tasks.add_task(
        run_llm_classification_task,
        task_id=task_id,
        file_id=request.file_id,
        text_column=request.text_column,
        hierarchy=request.hierarchy,
        llm_config_dict=request.llm_config.model_dump() # Pass config as dict
    )

    # Return initial status
    return TaskStatus(
        task_id=task_id,
        status=TaskStatusEnum.PENDING,
        message=tasks_db[task_id]["message"]
    )

@app.get("/tasks/{task_id}", response_model=TaskStatus, tags=["Classification Tasks"])
async def get_task_status(task_id: str = FastApiPath(..., description="ID of the task to check")):
    """
    Retrieves the status of a background classification task.
    """
    logger.debug(f"Request received for status of task ID: {task_id}")
    task = tasks_db.get(task_id)
    if not task:
        logger.warning(f"Task ID not found: {task_id}")
        raise HTTPException(status_code=404, detail="Task not found")

    result_url = None
    if task["status"] == TaskStatusEnum.SUCCESS and task.get("result_path"):
        # Construct the download URL dynamically based on the API structure
        result_url = f"/results/{task_id}/download" # Relative URL

    return TaskStatus(
        task_id=task_id,
        status=task["status"],
        message=task.get("message"),
        result_url=result_url
    )

@app.get("/results/{task_id}/download", tags=["Classification Tasks"], response_class=FileResponse)
async def download_results(task_id: str = FastApiPath(..., description="ID of the task whose results to download")):
    """
    Downloads the result file of a completed classification task.
    """
    logger.info(f"Request received to download results for task ID: {task_id}")
    task = tasks_db.get(task_id)
    if not task:
        logger.warning(f"Download request failed: Task ID not found: {task_id}")
        raise HTTPException(status_code=404, detail="Task not found")

    if task["status"] != TaskStatusEnum.SUCCESS:
        logger.warning(f"Download request failed: Task {task_id} status is {task['status']}")
        raise HTTPException(status_code=409, detail=f"Task is not yet completed successfully (Status: {task['status']})")

    result_path_str = task.get("result_path")
    if not result_path_str:
        logger.error(f"Download request failed: Result path missing for completed task {task_id}")
        raise HTTPException(status_code=500, detail="Result file path not found for completed task.")

    result_path = Path(result_path_str)
    if not result_path.exists():
        logger.error(f"Download request failed: Result file not found at path: {result_path}")
        raise HTTPException(status_code=500, detail="Result file not found on server.")

    # Determine filename for download (e.g., original_filename_results.xlsx)
    # For simplicity, using task_id for now. Could fetch original filename if needed.
    download_filename = f"{task_id}_classification_results.xlsx"

    logger.info(f"Streaming result file {result_path} for task {task_id} as '{download_filename}'")
    return FileResponse(
        path=result_path,
        filename=download_filename,
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )


# --- Run for Development (Optional) ---
# Keep this guard for potential direct script execution, but prefer uvicorn command
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for development...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True) # Use main:app for reload
