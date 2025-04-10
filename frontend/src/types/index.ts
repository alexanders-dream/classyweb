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
    result_data_url?: string | null; // Renamed from result_url
  }

  // --- LLM Config API Types ---

  export interface ProviderListResponse {
    providers: string[];
  }

  export interface FetchModelsRequest {
    provider: string;
    endpoint: string;
    api_key?: string | null;
  }

  export interface ModelListResponse {
    models: string[];
  }

  // --- Hierarchy Editor Types ---

  export interface HierarchyRow {
    // Using optional id for potential grid library integration
    id?: number | string;
    Theme: string;
    Category: string;
    Segment: string;
    Subsegment: string; // Standardized name
    Keywords: string; // Comma-separated string
  }

  // --- Hierarchy Suggestion API Types ---

  export interface HierarchySuggestRequest {
    sample_texts: string[];
    llm_config: LLMProviderConfig; // Reuse existing config type
  }

  // Represents the nested structure returned by the backend suggestion
  // This is intentionally kept generic (Dict/Any) as the exact structure
  // might vary slightly, and we primarily use it for flattening.
  export type NestedHierarchySuggestion = Record<string, any>;

  export interface HierarchySuggestResponse {
    suggestion?: NestedHierarchySuggestion | null;
    error?: string | null;
  }

  // --- LLM Classification API Types ---

  export interface ClassifyLLMRequest {
    file_id: string;
    original_filename: string;
    text_column: string;
    hierarchy: NestedHierarchySuggestion; // The nested hierarchy structure
    llm_config: LLMProviderConfig;
  }

  // --- Classification Results Type ---
  // Represents a single row in the results data (flexible columns)
  export type ClassificationResultRow = Record<string, any>;

  // --- HF Model Management Types ---

  export interface SavedHFModelListResponse {
    model_names: string[];
  }

  // --- HF Training Types ---

  export interface HFTrainingRequest {
    training_file_id: string;
    original_training_filename: string;
    text_column: string;
    // Mapping of hierarchy levels (e.g., 'L1') to column names. Value can be null.
    hierarchy_columns: Record<string, string | null>;
    base_model?: string; // Optional, backend has default
    num_epochs?: number; // Optional, backend has default
    new_model_name: string;
  }

  // --- HF Rules Types ---

  export interface HFRule {
    // Match backend model (using alias for Confidence Threshold)
    Label: string;
    Keywords: string;
    'Confidence Threshold': number; // Use the actual name expected by the backend/CSV
  }

  export interface HFRulesResponse {
    rules: HFRule[];
  }

  export interface HFRulesUpdateRequest {
    rules: HFRule[];
  }

  // --- HF Classification Types ---

  export interface HFClassificationRequest {
    file_id: string;
    original_filename: string;
    text_column: string;
    model_name: string; // Name of the saved HF model to use
  }

  // Add more types as needed for hierarchy, rules, etc. later
