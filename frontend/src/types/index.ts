export interface FileInfo {
    file_id: string;
    filename: string;
    columns: string[];
    num_rows: number;
    preview: Record<string, any>[]; // Array of objects for preview rows
  }
  
  export interface LLMProviderConfig {
    provider: string;
    endpoint: string;
    model_name: string;
    api_key?: string | null; // Optional or null
  }
  
  export interface TaskStatus {
    task_id: string;
    status: 'PENDING' | 'RUNNING' | 'SUCCESS' | 'FAILED'; // Use string literals for status
    message?: string | null;
    result_url?: string | null;
  }
  
  // Add more types as needed for hierarchy, rules, etc. later