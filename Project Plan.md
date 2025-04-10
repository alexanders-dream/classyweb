Okay, here is a modular project implementation plan for converting your Streamlit application into a separate TypeScript frontend and Python (FastAPI) backend.

Overall Goal: Create a robust web application with a distinct frontend (React + TypeScript) and backend (FastAPI + Python) that replicates and enhances the functionality of the original Streamlit app.

Guiding Principles:

Modularity: Break down work into independent or semi-independent modules.

API-First: Define backend API endpoints clearly before or alongside frontend implementation.

Iterative Development: Implement core functionality first, then add features layer by layer (e.g., get data flow working, then LLM, then HF).

Clear Deliverables: Define what should be functional at the end of each phase/module.

Assumptions:

Technology Choices: FastAPI for the backend, React + TypeScript for the frontend. Zustand for state management (adaptable). A UI library like Material UI or Mantine.

Environment: Development environment with Python, Node.js, and necessary package managers (pip, npm/yarn).

Team: This plan assumes a small team or individual developer; adjust task allocation as needed.

Phase 0: Foundation & Setup (Est. Time: 1-2 days)

Goal: Establish project structures, basic configurations, and runnable shells.

Modules:

Backend: Project Setup

Task: Create backend directory, set up Python virtual environment.

Task: Install FastAPI, Uvicorn, Pydantic, python-dotenv.

Task: Create initial main.py with a basic FastAPI app instance.

Task: Create config.py by porting relevant constants from the Streamlit version. Adapt paths if necessary (e.g., for saved models relative to the backend root).

Deliverable: Runnable empty FastAPI server (uvicorn main:app --reload).

Frontend: Project Setup

Task: Create frontend directory using Vite (npm create vite@latest frontend --template react-ts).

Task: Install core dependencies: axios, react-router-dom, state manager (zustand), UI library (@mui/material @emotion/react @emotion/styled or similar).

Task: Set up basic folder structure (components, services, store, types, features/pages).

Task: Create basic App.tsx layout (e.g., Sidebar, Main Content Area) using the UI library.

Deliverable: Runnable basic React app (npm run dev).

API: Initial Definition & Tooling

Task: Define basic Pydantic models for core concepts (e.g., FileInfo, LLMConfig, basic TaskStatus) in the backend (models.py or similar).

Task: Set up basic API service file (api.ts) in the frontend for Axios instance configuration.

Deliverable: Shared understanding of initial data structures.

Phase 1: Core Data Handling (Est. Time: 2-4 days)

Goal: Implement file upload, basic processing, and preview functionality.

Modules:

Backend: Refactor Utilities

Task: Move utils.py to the backend project.

Task: Remove all Streamlit dependencies (st.*, session state logic) from utils.py.

Task: Adapt load_data to accept a file path or FastAPI UploadFile object instead of a Streamlit uploaded file object. Keep data cleaning logic.

Task: Keep df_to_excel_bytes, build_hierarchy_from_df, flatten_hierarchy, parse_predicted_labels_to_columns.

Deliverable: Streamlit-independent utils.py.

Backend: File Upload API

Task: Implement /data/upload endpoint in FastAPI.

Task: Handle UploadFile, save it temporarily (e.g., to ./temp_uploads/ with UUID).

Task: Use refactored utils.load_data to read the saved file.

Task: Return FileInfo (file ID, filename, columns, row count, head preview) as JSON. Handle potential errors.

Deliverable: Functional /data/upload endpoint.

Frontend: File Upload Component

Task: Create components for uploading prediction data and (optional) training data.

Task: Use react-dropzone or similar for the UI.

Task: On file drop/selection, call the /data/upload API using the api.ts service.

Task: Store the returned FileInfo in the state manager (Zustand).

Task: Display file info (name, rows) and preview (react-table or simple map).

Task: Implement column selection dropdowns based on FileInfo.columns, storing selection in state.

Deliverable: Users can upload files, see previews, and select columns via the UI.

Frontend: State Management (Data)

Task: Define state slices in Zustand for predictionFileInfo, trainingFileInfo, selectedColumns, etc.

Task: Implement actions/reducers to update this state.

Deliverable: Centralized state for data setup.

Phase 2: LLM Workflow Implementation (Est. Time: 5-8 days)

Goal: Implement the end-to-end LLM configuration, hierarchy definition, classification, and results viewing.

Modules:

Backend: Refactor LLM Logic

Task: Move llm_classifier.py to backend.

Task: Remove all Streamlit dependencies. Replace UI feedback with logging.

Task: Ensure initialize_llm_client, fetch_available_models, generate_hierarchy_suggestion, classify_texts_with_llm return data/errors or raise exceptions.

Deliverable: Streamlit-independent llm_classifier.py.

Backend: LLM Config & Models API

Task: Implement /llm/providers (list supported), /llm/models (fetch based on provider/endpoint/key).

Task: Define Pydantic models for request/response. Handle API key logic securely (don't return keys).

Deliverable: Endpoints for frontend to configure LLM.

Frontend: LLM Sidebar

Task: Create LLMConfigSidebar.tsx component.

Task: Fetch providers, allow selection. Add inputs for endpoint/API key (password type).

Task: Fetch available models via API based on config. Allow model selection.

Task: Store the complete LLMConfigRequest in Zustand. Indicate connection status based on client initialization success (maybe add a /llm/test endpoint).

Deliverable: UI for configuring the LLM provider and model.

Backend: Hierarchy Suggestion API

Task: Implement /llm/hierarchy/suggest endpoint. Accepts sample texts and LLM config.

Task: Calls refactored generate_hierarchy_suggestion.

Deliverable: Endpoint to get AI suggestions.

Frontend: Hierarchy Editor

Task: Create HierarchyEditor.tsx component.

Task: Use a data grid library (react-data-grid, MUI DataGrid) for editing.

Task: Implement "Generate Suggestion" button: gets sample text (from uploaded data state), calls suggestion API.

Task: Display suggestion preview, allow "Apply" (updates editor state) or "Discard".

Task: Store hierarchy data (likely as array of objects matching rows) in Zustand.

Task: Display JSON preview of the nested structure derived from the editor state (using utils.build_hierarchy_from_df logic ported to TS or via a backend call).

Deliverable: UI for defining/editing the classification hierarchy, including AI suggestions.

Backend: LLM Classification Task API

Task: Implement /classify/llm endpoint to start the task. Accepts file_id, text_column, hierarchy, llm_config.

Task: Use FastAPI BackgroundTasks to run the (refactored) classify_texts_with_llm function.

Task: Implement basic task tracking (in-memory dict: task_id -> {status, message, result_path}). Return task_id and initial PENDING status.

Task: Implement /tasks/{task_id} endpoint to check status.

Task: Implement /results/{task_id}/download endpoint. Checks task status; if SUCCESS, streams the saved result file.

Deliverable: API for initiating, tracking, and downloading LLM classification results.

Frontend: LLM Classification Trigger & Results

Task: Add "Run LLM Classification" button. Calls /classify/llm API with data from state.

Task: On receiving task_id, start polling /tasks/{task_id} using useEffect and setInterval.

Task: Display task status (Pending, Running, Success, Failed) and messages. Show loading indicators.

Task: On SUCCESS, enable download button linking to /results/{task_id}/download. Optionally fetch a preview of results to display in a table.

Deliverable: UI to run LLM classification and view/download results.

Phase 3: Hugging Face Workflow Implementation (Est. Time: 6-10 days)

Goal: Implement the end-to-end HF model training, loading, classification, and results viewing.

Modules: (Parallel structure to Phase 2)

Backend: Refactor HF Logic

Task: Move hf_classifier.py to backend.

Task: Remove Streamlit dependencies. Adapt functions for background execution (training, classification).

Task: Ensure save_hf_model_artifacts and load_hf_model_artifacts work relative to backend paths.

Deliverable: Streamlit-independent hf_classifier.py.

Backend: HF Model Management API

Task: Implement /hf/models/saved (list saved model directories).

Task: Implement /hf/models/{model_name}/load (conceptually loads model config or prepares backend state - maybe doesn't load full model until classification).

Task: Implement /hf/models/save (if allowing save separate from training).

Deliverable: API endpoints for managing saved HF models.

Backend: HF Training Task API

Task: Implement /hf/train endpoint to start background training task. Accepts training file_id, column config, model params.

Task: Track status via task system (same as LLM). On success, save artifacts using save_hf_model_artifacts and update task status.

Deliverable: API for initiating and tracking HF training.

Backend: HF Rules API

Task: Implement /hf/rules/{model_name} (GET to fetch rules CSV/JSON, PUT/POST to update rules). Requires loading rules file associated with the model name.

Deliverable: API for managing HF classification rules.

Backend: HF Classification Task API

Task: Implement /classify/hf endpoint to start background classification. Accepts file_id, text_column, model_identifier (e.g., saved model name).

Task: Background task loads the specified model (using load_hf_model_artifacts), classifies (using classify_texts_with_hf), saves results, updates task status.

Deliverable: API for initiating, tracking, and downloading HF classification results.

Frontend: HF UI Components

Task: Create UI elements in relevant tabs/features for HF workflow: select training data/columns, select base model/epochs, trigger training, list/load saved models, display/edit rules (data editor), trigger classification.

Task: Integrate with state manager for selected HF model, rules data, task statuses.

Deliverable: UI for the complete HF workflow.

Frontend: HF Task Polling & Results

Task: Implement polling logic for HF training and classification tasks.

Task: Display status, handle results download/display similarly to LLM workflow.

Deliverable: Functional HF task tracking and results handling in the UI.

Phase 4: Refinement, Testing & Deployment (Est. Time: 4-7 days)

Goal: Polish the application, add tests, and prepare for deployment.

Modules:

Backend: Task Management Improvement

Task: (Optional but recommended) Replace in-memory task dictionary with Redis or Celery for better scalability and persistence.

Deliverable: More robust background task handling.

Backend: Error Handling & Logging

Task: Implement consistent error responses from the API (e.g., standard JSON structure).

Task: Enhance logging for better debugging.

Deliverable: Improved backend robustness and observability.

Frontend: Error Handling & UX

Task: Implement better display of API errors in the UI.

Task: Improve loading indicators and user feedback.

Task: Refine UI layout and styling.

Deliverable: Polished user experience.

Testing

Task: Write basic API tests for key backend endpoints (Pytest).

Task: (Optional) Write basic unit/integration tests for critical frontend components/logic (e.g., using Vitest/React Testing Library).

Deliverable: Increased confidence in application correctness.

Deployment Preparation

Task: Dockerize the FastAPI backend application.

Task: Configure CORS settings appropriately in FastAPI.

Task: Build the static frontend assets (npm run build).

Deliverable: Deployable backend container image and frontend static files.

Deployment

Task: Deploy backend container to a cloud service (Cloud Run, ECS, App Service, etc.).

Task: Deploy frontend static files to a hosting service (Netlify, Vercel, S3/CloudFront, etc.).

Task: Configure environment variables (API keys, backend URL) in deployment environments.

Deliverable: Live, functional application.

This plan provides a modular structure. You can adjust the time estimates based on complexity and developer experience. Remember to commit frequently and potentially use feature branches for larger modules. Good luck!