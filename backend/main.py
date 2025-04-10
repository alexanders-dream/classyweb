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
import numpy as np # Add numpy import
from pathlib import Path
import uuid
import shutil
import traceback # Added traceback
from fastapi.middleware.cors import CORSMiddleware
import requests # Need this for exception handling in get_llm_models
from enum import Enum # Added Enum

# Assuming config, utils, models, hf_classifier are importable
try:
    from . import config, utils, llm_classifier, hf_classifier # Added hf_classifier
    # Import new models
    from .models import (
        FileInfo, ProviderListResponse, FetchModelsRequest, ModelListResponse,
        HierarchySuggestRequest, HierarchySuggestResponse,
        ClassifyLLMRequest, TaskStatus, HFTrainingRequest,
        HFRulesResponse, HFRulesUpdateRequest, HFClassificationRequest # Added HFClassificationRequest
    )
except ImportError:
    from . import config, utils, llm_classifier, hf_classifier # Fallback for direct script run
    # Import new models
    from models import (
        FileInfo, ProviderListResponse, FetchModelsRequest, ModelListResponse,
        HierarchySuggestRequest, HierarchySuggestResponse,
        ClassifyLLMRequest, TaskStatus, HFTrainingRequest,
        HFRulesResponse, HFRulesUpdateRequest, HFClassificationRequest # Added HFClassificationRequest
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
HF_MODELS_DIR = Path("./saved_hf_models") # Directory for saved HF models
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
HF_MODELS_DIR.mkdir(parents=True, exist_ok=True) # Ensure HF models dir exists
logger.info(f"Temporary upload directory: {UPLOAD_DIR.resolve()}")
logger.info(f"Saved HF models directory: {HF_MODELS_DIR.resolve()}")

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
    original_filename: str,
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
        df_input = utils.load_data(file_path=input_file_path, original_filename=original_filename)
        if df_input is None:
            raise ValueError(f"Failed to load data from file: {input_file_path} (Original: {original_filename})")

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

# --- HF Training Background Task ---
def run_hf_training_task(
    task_id: str,
    training_file_id: str,
    original_training_filename: str,
    text_column: str,
    hierarchy_columns: Dict[str, Optional[str]],
    base_model: str,
    num_epochs: int,
    new_model_name: str
):
    """Background task to perform Hugging Face model training."""
    logger.info(f"Starting HF training task {task_id} for file {training_file_id}, saving as '{new_model_name}'")
    tasks_db[task_id]["status"] = TaskStatusEnum.RUNNING
    tasks_db[task_id]["message"] = "Loading training data..."

    input_file_path = UPLOAD_DIR / training_file_id
    save_path = HF_MODELS_DIR / new_model_name # Path to save the new model

    try:
        # 1. Load Training Data
        if not input_file_path.exists():
            raise FileNotFoundError(f"Training file not found for file_id: {training_file_id}")

        df_train = utils.load_data(file_path=input_file_path, original_filename=original_training_filename)
        if df_train is None:
            raise ValueError(f"Failed to load training data from file: {input_file_path} (Original: {original_training_filename})")

        tasks_db[task_id]["message"] = f"Training data loaded ({len(df_train)} rows). Preparing data..."

        # 2. Prepare Hierarchical Data
        train_texts, train_labels = hf_classifier.prepare_hierarchical_training_data(
            df=df_train,
            text_col=text_column,
            hierarchy_cols=hierarchy_columns
        )
        if train_texts is None or train_labels is None:
            raise ValueError("Failed to prepare hierarchical training data. Check logs for details.")

        tasks_db[task_id]["message"] = f"Data prepared ({len(train_texts)} texts). Starting model training..."

        # 3. Train Model
        # Note: train_hf_model handles label processing, splitting, tokenizing, training, rule extraction
        trained_model, tokenizer, label_map, rules_df = hf_classifier.train_hf_model(
            all_train_texts=train_texts,
            all_train_labels_list=train_labels,
            model_choice=base_model,
            num_epochs=num_epochs
            # validation_split_ratio can use default from config via hf_classifier
        )

        if trained_model is None or tokenizer is None or label_map is None or rules_df is None:
             # Error should have been logged within train_hf_model
             raise RuntimeError("HF model training failed. Check logs for details.")

        tasks_db[task_id]["message"] = "Training complete. Saving model artifacts..."

        # 4. Save Model Artifacts
        save_successful = hf_classifier.save_hf_model_artifacts(
            model=trained_model,
            tokenizer=tokenizer,
            label_map=label_map,
            rules_df=rules_df,
            save_path=str(save_path) # Pass path as string
        )

        if not save_successful:
            # Error logged within save_hf_model_artifacts
            raise RuntimeError(f"Failed to save trained model artifacts to {save_path}")

        logger.info(f"Task {task_id}: HF model artifacts saved successfully to '{save_path}'")

        # 5. Update Task Status - Success
        tasks_db[task_id]["status"] = TaskStatusEnum.SUCCESS
        tasks_db[task_id]["message"] = f"HF model '{new_model_name}' trained and saved successfully."
        tasks_db[task_id]["result_path"] = str(save_path) # Store the model save path

    except Exception as e:
        logger.error(f"HF Training Task {task_id} failed: {e}", exc_info=True)
        tasks_db[task_id]["status"] = TaskStatusEnum.FAILED
        tasks_db[task_id]["message"] = f"Error during HF training: {e}"
        tasks_db[task_id]["result_path"] = None
        # Clean up potentially partially saved model directory
        if save_path.exists():
            try:
                shutil.rmtree(save_path)
                logger.info(f"Task {task_id}: Cleaned up partially saved model directory {save_path}")
            except Exception as cleanup_e:
                logger.error(f"Task {task_id}: Failed to cleanup model directory {save_path}: {cleanup_e}")

# --- HF Classification Background Task ---
def run_hf_classification_task(
    task_id: str,
    file_id: str,
    original_filename: str,
    text_column: str,
    model_name: str
):
    """Background task to perform Hugging Face classification using a saved model."""
    logger.info(f"Starting HF classification task {task_id} for file {file_id} using model '{model_name}'")
    tasks_db[task_id]["status"] = TaskStatusEnum.RUNNING
    tasks_db[task_id]["message"] = f"Loading saved model '{model_name}'..."

    input_file_path = UPLOAD_DIR / file_id
    model_path = HF_MODELS_DIR / model_name
    result_file_path: Optional[Path] = None

    try:
        # 1. Load Model Artifacts
        if not model_path.is_dir():
            raise FileNotFoundError(f"Saved model directory not found: {model_path}")

        model, tokenizer, label_map, rules_df = hf_classifier.load_hf_model_artifacts(str(model_path))
        if model is None or tokenizer is None or label_map is None:
            # Error logged within load_hf_model_artifacts
            raise RuntimeError(f"Failed to load model artifacts from {model_path}")
        # rules_df can be None if rules.csv doesn't exist, which is handled by classify_texts_with_hf

        tasks_db[task_id]["message"] = f"Model '{model_name}' loaded. Loading data..."

        # 2. Load Data to Classify
        if not input_file_path.exists():
            raise FileNotFoundError(f"Input file not found for file_id: {file_id}")

        df_input = utils.load_data(file_path=input_file_path, original_filename=original_filename)
        if df_input is None:
            raise ValueError(f"Failed to load data from file: {input_file_path} (Original: {original_filename})")

        if text_column not in df_input.columns:
             raise ValueError(f"Text column '{text_column}' not found in the input file.")

        # Extract the text column as a list
        texts_to_classify = df_input[text_column].astype(str).tolist()
        tasks_db[task_id]["message"] = f"Data loaded ({len(texts_to_classify)} texts). Starting classification..."

        # 3. Run Classification
        predicted_labels_list = hf_classifier.classify_texts_with_hf(
            texts=texts_to_classify,
            model=model,
            tokenizer=tokenizer,
            label_map=label_map,
            rules_df=rules_df # Pass rules DataFrame (can be None)
        )

        if not predicted_labels_list or len(predicted_labels_list) != len(texts_to_classify):
             raise RuntimeError("HF classification function returned unexpected results.")

        tasks_db[task_id]["message"] = "Classification complete. Processing and saving results..."

        # 4. Process and Save Results
        # Add predicted labels back to the original DataFrame
        # Use a consistent column naming convention
        df_results = df_input.copy()
        df_results['predicted_labels_hf'] = ['; '.join(labels) for labels in predicted_labels_list] # Join list into string

        # Optionally parse labels into separate columns if needed (using utils function)
        # df_results = utils.parse_predicted_labels_to_columns(df_results, 'predicted_labels_hf')

        result_filename = f"{task_id}_hf_results.xlsx"
        result_file_path = UPLOAD_DIR / result_filename
        excel_bytes = utils.df_to_excel_bytes(df_results)
        if not excel_bytes:
             raise RuntimeError("Failed to convert results DataFrame to Excel bytes.")

        with open(result_file_path, "wb") as f:
            f.write(excel_bytes)

        logger.info(f"Task {task_id}: HF classification results saved to {result_file_path}")

        # 5. Update Task Status - Success
        tasks_db[task_id]["status"] = TaskStatusEnum.SUCCESS
        tasks_db[task_id]["message"] = "HF classification completed successfully."
        tasks_db[task_id]["result_path"] = str(result_file_path) # Store result file path

    except Exception as e:
        logger.error(f"HF Classification Task {task_id} failed: {e}", exc_info=True)
        tasks_db[task_id]["status"] = TaskStatusEnum.FAILED
        tasks_db[task_id]["message"] = f"Error during HF classification: {e}"
        tasks_db[task_id]["result_path"] = None
        # Clean up partial result file if it exists
        if result_file_path and result_file_path.exists():
            try:
                os.remove(result_file_path)
                logger.info(f"Task {task_id}: Cleaned up partial HF result file {result_file_path}")
            except Exception as cleanup_e:
                logger.error(f"Task {task_id}: Failed to cleanup partial HF result file {result_file_path}: {cleanup_e}")


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

# --- HF Model Management Endpoints ---

class SavedHFModelListResponse(BaseModel):
    """Response model for listing saved Hugging Face models."""
    model_names: List[str]

@app.get("/hf/models/saved", response_model=SavedHFModelListResponse, tags=["HF Model Management"])
async def list_saved_hf_models():
    """Lists the names of saved Hugging Face model directories."""
    logger.info("Request received for /hf/models/saved")
    saved_models = []
    try:
        for item in HF_MODELS_DIR.iterdir():
            if item.is_dir():
                # Basic check: does it look like a model dir? (e.g., contains config.json)
                if (item / "config.json").exists():
                    saved_models.append(item.name)
                else:
                    logger.warning(f"Directory '{item.name}' in {HF_MODELS_DIR} does not contain config.json, skipping.")
        logger.info(f"Found saved HF models: {saved_models}")
        return SavedHFModelListResponse(model_names=sorted(saved_models))
    except Exception as e:
        logger.error(f"Error listing saved HF models in {HF_MODELS_DIR}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list saved Hugging Face models.")

# --- HF Rules Endpoints ---

@app.get("/hf/rules/{model_name}", response_model=HFRulesResponse, tags=["HF Rules Management"])
async def get_hf_rules(model_name: str = FastApiPath(..., description="Name of the saved HF model")):
    """Retrieves the classification rules for a specific saved Hugging Face model."""
    logger.info(f"Request received for rules of HF model: {model_name}")

    # Validate model name format (simple check for path traversal)
    if "/" in model_name or "\\" in model_name or ".." in model_name:
         logger.error(f"Invalid characters in model name: '{model_name}'")
         raise HTTPException(status_code=400, detail="Invalid characters in model name.")

    model_dir = HF_MODELS_DIR / model_name
    rules_file_path = model_dir / "rules.csv"

    if not model_dir.is_dir():
        logger.warning(f"Model directory not found: {model_dir}")
        raise HTTPException(status_code=404, detail=f"Saved model '{model_name}' not found.")

    if not rules_file_path.exists():
        logger.warning(f"Rules file not found for model '{model_name}' at {rules_file_path}")
        # Return empty list if rules file doesn't exist (model might exist without rules)
        return HFRulesResponse(rules=[])

    try:
        df_rules = pd.read_csv(rules_file_path)
        # Validate required columns (case-insensitive check just in case)
        required_cols = ["Label", "Keywords", "Confidence Threshold"]
        if not all(col in df_rules.columns for col in required_cols):
             logger.error(f"Rules file {rules_file_path} is missing required columns: {required_cols}")
             raise HTTPException(status_code=500, detail=f"Rules file for model '{model_name}' is malformed (missing columns).")

        # Convert to list of dicts, handling NaN/None for JSON compatibility
        # Ensure correct column names are used for pydantic model population
        df_rules.rename(columns={"Confidence Threshold": "Confidence_Threshold"}, inplace=True) # Match Pydantic alias
        rules_list = df_rules.replace({pd.NA: None, np.nan: None}).to_dict('records')

        # Pydantic will validate the structure based on HFRule model within HFRulesResponse
        logger.info(f"Successfully loaded {len(rules_list)} rules for model '{model_name}'.")
        return HFRulesResponse(rules=rules_list)

    except pd.errors.EmptyDataError:
        logger.warning(f"Rules file {rules_file_path} is empty for model '{model_name}'.")
        return HFRulesResponse(rules=[]) # Return empty list for empty file
    except Exception as e:
        logger.error(f"Error reading or processing rules file {rules_file_path} for model '{model_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to read or process rules file for model '{model_name}'.")

@app.put("/hf/rules/{model_name}", status_code=200, tags=["HF Rules Management"])
async def update_hf_rules(
    model_name: str = FastApiPath(..., description="Name of the saved HF model"),
    request: HFRulesUpdateRequest = Body(...)
):
    """Updates/overwrites the classification rules for a specific saved HF model."""
    logger.info(f"Request received to update rules for HF model: {model_name}")

    # Validate model name format
    if "/" in model_name or "\\" in model_name or ".." in model_name:
         logger.error(f"Invalid characters in model name: '{model_name}'")
         raise HTTPException(status_code=400, detail="Invalid characters in model name.")

    model_dir = HF_MODELS_DIR / model_name
    rules_file_path = model_dir / "rules.csv"

    if not model_dir.is_dir():
        logger.warning(f"Cannot update rules: Model directory not found: {model_dir}")
        raise HTTPException(status_code=404, detail=f"Saved model '{model_name}' not found.")

    try:
        # Convert Pydantic models back to a list of dicts suitable for DataFrame
        rules_data = [rule.model_dump(by_alias=True) for rule in request.rules] # Use by_alias=True to get "Confidence Threshold"
        df_new_rules = pd.DataFrame(rules_data)

        # Ensure the DataFrame has the correct columns, even if the input list was empty
        required_cols = ["Label", "Keywords", "Confidence Threshold"]
        for col in required_cols:
            if col not in df_new_rules.columns:
                df_new_rules[col] = [] # Add empty column if missing

        # Reorder columns to the standard format before saving
        df_new_rules = df_new_rules[required_cols]

        # Overwrite the existing rules file
        df_new_rules.to_csv(rules_file_path, index=False, encoding='utf-8')

        logger.info(f"Successfully updated rules file for model '{model_name}' at {rules_file_path}")
        return {"message": f"Rules for model '{model_name}' updated successfully."}

    except Exception as e:
        logger.error(f"Error updating rules file {rules_file_path} for model '{model_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update rules file for model '{model_name}'.")


@app.post("/hf/train", response_model=TaskStatus, status_code=202, tags=["HF Training"])
async def start_hf_training(
    request: HFTrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Starts a background task for training a Hugging Face model.
    """
    task_id = str(uuid.uuid4())
    logger.info(f"Received HF training request for file '{request.original_training_filename}' (ID: {request.training_file_id}). New model name: '{request.new_model_name}'. Task ID: {task_id}")

    # Validate input file exists
    input_file_path = UPLOAD_DIR / request.training_file_id
    if not input_file_path.exists():
        logger.error(f"Training file not found for file_id: {request.training_file_id}")
        raise HTTPException(status_code=404, detail=f"Training file with ID '{request.training_file_id}' not found.")

    # Validate new model name doesn't already exist
    new_model_path = HF_MODELS_DIR / request.new_model_name
    if new_model_path.exists():
        logger.error(f"Cannot train: Model directory '{request.new_model_name}' already exists at {new_model_path}")
        raise HTTPException(status_code=409, detail=f"A model named '{request.new_model_name}' already exists. Please choose a different name.")

    # Validate model name format (simple check for path traversal)
    if "/" in request.new_model_name or "\\" in request.new_model_name or ".." in request.new_model_name:
         logger.error(f"Invalid characters in new model name: '{request.new_model_name}'")
         raise HTTPException(status_code=400, detail="Invalid characters in new model name. Use alphanumeric characters, hyphens, or underscores.")


    # Store initial task status
    tasks_db[task_id] = {
        "status": TaskStatusEnum.PENDING,
        "message": "Training task received, pending execution.",
        "result_path": None # Will store the save path on success
    }

    # Add the background task
    background_tasks.add_task(
        run_hf_training_task,
        task_id=task_id,
        training_file_id=request.training_file_id,
        original_training_filename=request.original_training_filename,
        text_column=request.text_column,
        hierarchy_columns=request.hierarchy_columns,
        base_model=request.base_model,
        num_epochs=request.num_epochs,
        new_model_name=request.new_model_name
    )

    # Return initial status
    return TaskStatus(
        task_id=task_id,
        status=TaskStatusEnum.PENDING,
        message=tasks_db[task_id]["message"]
    )


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

# --- HF Classification Task Endpoint ---

@app.post("/classify/hf", response_model=TaskStatus, status_code=202, tags=["Classification Tasks"])
async def start_hf_classification(
    request: HFClassificationRequest,
    background_tasks: BackgroundTasks
):
    """
    Starts a background task for HF classification using a saved model.
    """
    task_id = str(uuid.uuid4())
    logger.info(f"Received HF classification request for file '{request.original_filename}' (ID: {request.file_id}) using model '{request.model_name}'. Task ID: {task_id}")

    # Validate input file exists
    input_file_path = UPLOAD_DIR / request.file_id
    if not input_file_path.exists():
        logger.error(f"Input file not found for file_id: {request.file_id}")
        raise HTTPException(status_code=404, detail=f"Input file with ID '{request.file_id}' not found.")

    # Validate model exists
    model_path = HF_MODELS_DIR / request.model_name
    if not model_path.is_dir() or not (model_path / "config.json").exists():
         logger.error(f"Saved HF model '{request.model_name}' not found at {model_path}")
         raise HTTPException(status_code=404, detail=f"Saved HF model '{request.model_name}' not found.")

    # Store initial task status
    tasks_db[task_id] = {
        "status": TaskStatusEnum.PENDING,
        "message": "HF classification task received, pending execution.",
        "result_path": None
    }

    # Add the background task
    background_tasks.add_task(
        run_hf_classification_task,
        task_id=task_id,
        file_id=request.file_id,
        original_filename=request.original_filename,
        text_column=request.text_column,
        model_name=request.model_name
    )

    # Return initial status
    return TaskStatus(
        task_id=task_id,
        status=TaskStatusEnum.PENDING,
        message=tasks_db[task_id]["message"]
    )


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
    logger.info(f"Received LLM classification request for file '{request.original_filename}' (ID: {request.file_id}). Assigning Task ID: {task_id}")

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
        original_filename=request.original_filename,
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

    result_data_url = None # Renamed variable
    if task["status"] == TaskStatusEnum.SUCCESS and task.get("result_path"):
        # Construct the URL for fetching data dynamically
        result_data_url = f"/results/{task_id}/data" # URL to get JSON data

    return TaskStatus(
        task_id=task_id,
        status=task["status"],
        message=task.get("message"),
        result_data_url=result_data_url # Return the data URL
    )

# Endpoint to get results as JSON data
@app.get("/results/{task_id}/data", tags=["Classification Tasks"], response_model=List[Dict[str, Any]])
async def get_result_data(task_id: str = FastApiPath(..., description="ID of the task whose results to fetch")):
    """
    Retrieves the classification results as JSON data for a completed task.
    """
    logger.info(f"Request received to fetch result data for task ID: {task_id}")
    task = tasks_db.get(task_id)
    if not task:
        logger.warning(f"Data request failed: Task ID not found: {task_id}")
        raise HTTPException(status_code=404, detail="Task not found")

    if task["status"] != TaskStatusEnum.SUCCESS:
        logger.warning(f"Data request failed: Task {task_id} status is {task['status']}")
        raise HTTPException(status_code=409, detail=f"Task is not yet completed successfully (Status: {task['status']})")

    result_path_str = task.get("result_path")
    if not result_path_str:
        logger.error(f"Data request failed: Result path missing for completed task {task_id}")
        raise HTTPException(status_code=500, detail="Result file path not found for completed task.")

    result_path = Path(result_path_str)
    if not result_path.exists():
        logger.error(f"Data request failed: Result file not found at path: {result_path}")
        raise HTTPException(status_code=500, detail="Result file not found on server.")

    try:
        # Read the Excel file into a pandas DataFrame
        df_results = pd.read_excel(result_path)
        # Convert DataFrame to list of dictionaries (JSON serializable)
        # Handle potential NaN values which are not valid JSON -> convert to None
        results_json = df_results.replace({pd.NA: None, float('nan'): None}).to_dict('records')
        logger.info(f"Successfully read and converted results for task {task_id} to JSON.")
        return results_json
    except Exception as e:
        logger.error(f"Failed to read or convert result file {result_path} for task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process result file.")


@app.get("/results/{task_id}/download", tags=["Classification Tasks"], response_class=FileResponse)
async def download_results(task_id: str = FastApiPath(..., description="ID of the task whose results to download")): # Keep download endpoint
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
