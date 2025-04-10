# backend/main.py
import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends # Added UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path
import uuid
import shutil
import traceback # Added traceback
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Assuming config, utils, models are importable
try:
    from . import config, utils
    from .models import FileInfo # Import specific model
except ImportError:
    import config, utils # Fallback for direct script run
    from models import FileInfo

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

# --- Configuration ---
# Create upload directory if it doesn't exist
UPLOAD_DIR = Path("./temp_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Temporary upload directory: {UPLOAD_DIR.resolve()}")

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


# --- Run for Development (Optional) ---
# Keep this guard for potential direct script execution, but prefer uvicorn command
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for development...")
    uvicorn.run(app, host="127.0.0.1", port=8000) # Use 127.0.0.1 for local dev