// frontend/src/services/api.ts
import axios from 'axios';
// Import the new types along with existing ones
import {
  FileInfo,
  ProviderListResponse,
  FetchModelsRequest,
  ModelListResponse,
  HierarchySuggestRequest,
  HierarchySuggestResponse,
  ClassifyLLMRequest,
  TaskStatus,
  ClassificationResultRow, // Added
  // HF Types
  SavedHFModelListResponse,
  HFTrainingRequest,
  HFRulesResponse,
  HFRulesUpdateRequest,
  HFClassificationRequest,
  HFRule // Added HFRule for update request body
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API call error:', error.response?.data || error.message);
    // Optional: Transform error into a standard format
    return Promise.reject(error.response?.data || { message: error.message });
  }
);

// --- Specific API Call Functions ---

export const uploadFile = async (file: File): Promise<FileInfo> => {
  const formData = new FormData();
  formData.append('file', file); // Key must match FastAPI parameter name 'file'
  try {
    const response = await apiClient.post<FileInfo>('/data/upload', formData, {
      headers: {
        // Axios might set this automatically for FormData, but explicitly is fine
        'Content-Type': 'multipart/form-data',
      },
      // Optional: Add progress tracking here if needed
      // onUploadProgress: progressEvent => { ... }
    });
    return response.data;
  } catch (error: any) {
    // Rethrow a more specific error or the formatted error from interceptor
    throw error;
  }
};

// --- LLM Config API Functions ---

export const getLLMProviders = async (): Promise<ProviderListResponse> => {
  try {
    const response = await apiClient.get<ProviderListResponse>('/llm/providers');
    return response.data;
  } catch (error: any) {
    console.error("Error fetching LLM providers:", error);
    throw error; // Rethrow formatted error from interceptor
  }
};

export const fetchLLMModels = async (requestData: FetchModelsRequest): Promise<ModelListResponse> => {
  try {
    // Use POST method as defined in the backend
    const response = await apiClient.post<ModelListResponse>('/llm/models', requestData);
    return response.data;
  } catch (error: any) {
    console.error("Error fetching LLM models:", error);
    throw error; // Rethrow formatted error from interceptor
  }
};

// --- Hierarchy Suggestion API Function ---

export const suggestHierarchy = async (requestData: HierarchySuggestRequest): Promise<HierarchySuggestResponse> => {
  if (!requestData.llm_config) {
    // Frontend validation before sending
    console.error("LLM configuration is missing in suggestHierarchy request.");
    return { suggestion: null, error: "LLM configuration is required." };
    // Alternatively, throw new Error("LLM configuration is required.");
  }
  try {
    const response = await apiClient.post<HierarchySuggestResponse>('/llm/hierarchy/suggest', requestData);
    // Check if the backend returned an error within a 200 response
    if (response.data.error) {
        console.warn(`Hierarchy suggestion failed (backend reported): ${response.data.error}`);
    }
    return response.data;
  } catch (error: any) {
    console.error("Error fetching hierarchy suggestion:", error);
    // Return error structure consistent with backend response if possible
    // The interceptor might already format this, but being explicit can help.
    const errorMessage = error?.detail || error?.message || 'Failed to fetch suggestion due to network or server error.';
    return { suggestion: null, error: errorMessage };
    // Or rethrow: throw error;
  }
};

// --- Classification Task API Functions ---

export const startLLMClassification = async (requestData: ClassifyLLMRequest): Promise<TaskStatus> => {
  // Add frontend validation if needed (e.g., check if file_id, text_column exist)
  if (!requestData.file_id || !requestData.text_column || !requestData.hierarchy || !requestData.llm_config) {
      console.error("Missing required fields for LLM classification request:", requestData);
      throw new Error("Missing required fields to start LLM classification.");
  }
  try {
    // Backend expects 202 Accepted, response body contains initial TaskStatus
    const response = await apiClient.post<TaskStatus>('/classify/llm', requestData);
    return response.data;
  } catch (error: any) {
    console.error("Error starting LLM classification:", error);
    throw error; // Rethrow formatted error from interceptor
  }
};

export const getTaskStatus = async (taskId: string): Promise<TaskStatus> => {
  if (!taskId) {
      console.error("Task ID is required to fetch status.");
      throw new Error("Task ID is required.");
  }
  try {
    const response = await apiClient.get<TaskStatus>(`/tasks/${taskId}`);
    return response.data;
  } catch (error: any) {
    console.error(`Error fetching status for task ${taskId}:`, error);
    // Handle 404 specifically?
    if (axios.isAxiosError(error) && error.response?.status === 404) {
        throw new Error(`Task with ID ${taskId} not found.`);
    }
    throw error; // Rethrow formatted error from interceptor
  }
};

export const getResultData = async (taskId: string): Promise<ClassificationResultRow[]> => {
    if (!taskId) {
        console.error("Task ID is required to fetch result data.");
        throw new Error("Task ID is required.");
    }
    try {
        // The backend endpoint /results/{task_id}/data returns the JSON array
        const response = await apiClient.get<ClassificationResultRow[]>(`/results/${taskId}/data`);
        return response.data;
    } catch (error: any) {
        console.error(`Error fetching result data for task ${taskId}:`, error);
        if (axios.isAxiosError(error) && error.response?.status === 404) {
            throw new Error(`Result data not found for task ${taskId}.`);
        }
        if (axios.isAxiosError(error) && error.response?.status === 409) {
            throw new Error(`Task ${taskId} is not yet completed successfully.`);
        }
        throw error; // Rethrow formatted error from interceptor or other errors
    }
};


// Note: Download URL is constructed from TaskStatus.result_url, so no separate API call needed here.
// We will keep the backend download endpoint for potential direct downloads if needed later.

// --- HF Model Management API Functions ---

export const listSavedHFModels = async (): Promise<SavedHFModelListResponse> => {
  try {
    const response = await apiClient.get<SavedHFModelListResponse>('/hf/models/saved');
    return response.data;
  } catch (error: any) {
    console.error("Error fetching saved HF models:", error);
    throw error;
  }
};

// --- HF Rules API Functions ---

export const getHFRules = async (modelName: string): Promise<HFRulesResponse> => {
  if (!modelName) {
    console.error("Model name is required to fetch HF rules.");
    throw new Error("Model name is required.");
  }
  try {
    const response = await apiClient.get<HFRulesResponse>(`/hf/rules/${encodeURIComponent(modelName)}`);
    return response.data;
  } catch (error: any) {
    console.error(`Error fetching rules for HF model ${modelName}:`, error);
    if (axios.isAxiosError(error) && error.response?.status === 404) {
        // Model or rules file not found - return empty rules as per backend logic
        return { rules: [] };
    }
    throw error;
  }
};

export const updateHFRules = async (modelName: string, rules: HFRule[]): Promise<{ message: string }> => {
  if (!modelName) {
    console.error("Model name is required to update HF rules.");
    throw new Error("Model name is required.");
  }
  const requestData: HFRulesUpdateRequest = { rules };
  try {
    // Backend returns a simple message on success
    const response = await apiClient.put<{ message: string }>(`/hf/rules/${encodeURIComponent(modelName)}`, requestData);
    return response.data;
  } catch (error: any) {
    console.error(`Error updating rules for HF model ${modelName}:`, error);
    throw error;
  }
};


// --- HF Training API Function ---

export const startHFTraining = async (requestData: HFTrainingRequest): Promise<TaskStatus> => {
  // Add frontend validation if needed
  if (!requestData.training_file_id || !requestData.text_column || !requestData.hierarchy_columns || !requestData.new_model_name) {
      console.error("Missing required fields for HF training request:", requestData);
      throw new Error("Missing required fields to start HF training.");
  }
  try {
    const response = await apiClient.post<TaskStatus>('/hf/train', requestData);
    return response.data;
  } catch (error: any) {
    console.error("Error starting HF training:", error);
    throw error;
  }
};

// --- HF Classification API Function ---

export const startHFClassification = async (requestData: HFClassificationRequest): Promise<TaskStatus> => {
  if (!requestData.file_id || !requestData.text_column || !requestData.model_name) {
      console.error("Missing required fields for HF classification request:", requestData);
      throw new Error("Missing required fields to start HF classification.");
  }
  try {
    const response = await apiClient.post<TaskStatus>('/classify/hf', requestData);
    return response.data;
  } catch (error: any) {
    console.error("Error starting HF classification:", error);
    throw error;
  }
};


export default apiClient; // Keep default export if used elsewhere
