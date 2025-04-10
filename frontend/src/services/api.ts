// frontend/src/services/api.ts
import axios from 'axios';
import { FileInfo } from '../types'; // Assuming types/index.ts exists

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

export default apiClient; // Keep default export if used elsewhere