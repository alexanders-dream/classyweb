// frontend/src/store/store.ts
import { create } from 'zustand';
// Import necessary types
import {
  FileInfo, LLMProviderConfig, HierarchyRow, NestedHierarchySuggestion, TaskStatus, ClassificationResultRow,
  // HF Types
  HFRule
} from '../types'; // Added TaskStatus, ClassificationResultRow, HFRule

// Enum for Task Types (optional but good practice)
export enum TaskType {
  LLM_CLASSIFICATION = 'LLM_CLASSIFICATION',
  HF_CLASSIFICATION = 'HF_CLASSIFICATION',
  HF_TRAINING = 'HF_TRAINING',
}

interface AppState {
  // Data & Files
  predictionFileInfo: FileInfo | null;
  setPredictionFileInfo: (info: FileInfo | null) => void;
  selectedPredictionColumn: string | null;
  setSelectedPredictionColumn: (col: string | null) => void;

  // Training Data File (for HF)
  trainingFileInfo: FileInfo | null;
  setTrainingFileInfo: (info: FileInfo | null) => void;
  // Columns selected from training data for text and hierarchy levels
  selectedTrainingTextColumn: string | null;
  setSelectedTrainingTextColumn: (col: string | null) => void;
  selectedTrainingHierarchyColumns: Record<string, string | null>; // e.g., { L1: 'ColA', L2: 'ColB', L3: null }
  setSelectedTrainingHierarchyColumns: (cols: Record<string, string | null>) => void;

  // LLM Config
  llmConfig: LLMProviderConfig | null;
  setLLMConfig: (config: LLMProviderConfig | null) => void;

  // Hierarchy Editor State (Used by both LLM and potentially HF rules)
  hierarchyRows: HierarchyRow[]; // Array of rows for the editor
  setHierarchyRows: (rows: HierarchyRow[]) => void;
  pendingSuggestion: NestedHierarchySuggestion | null; // Holds AI suggestion before applying
  setPendingSuggestion: (suggestion: NestedHierarchySuggestion | null) => void;
  hierarchyIsValid: boolean; // Flag if current rows form a valid structure
  setHierarchyIsValid: (isValid: boolean) => void;

  // --- HF Specific State ---
  savedHFModels: string[]; // List of names of saved models
  setSavedHFModels: (models: string[]) => void;
  selectedHFModel: string | null; // Name of the model selected for classification or rule editing
  setSelectedHFModel: (modelName: string | null) => void;
  hfModelRules: HFRule[]; // Rules for the selected HF model
  setHFModelRules: (rules: HFRule[]) => void;

  // --- Generic Task State (Handles LLM/HF Classification & HF Training) ---
  activeTaskId: string | null;
  activeTaskType: TaskType | null; // To know which kind of task is running
  activeTaskStatus: TaskStatus | null;
  setActiveTask: (taskId: string | null, taskType: TaskType | null) => void; // Combined setter
  setActiveTaskStatus: (status: TaskStatus | null) => void;
  // Results are kept separate as they have different structures potentially
  classificationResults: ClassificationResultRow[] | null;
  setClassificationResults: (results: ClassificationResultRow[] | null) => void;
  // Training results might just be a success/fail message in the status

}

export const useAppStore = create<AppState>((set) => ({
  // Initial values
  predictionFileInfo: null,
  selectedPredictionColumn: null,
  trainingFileInfo: null,
  selectedTrainingTextColumn: null,
  selectedTrainingHierarchyColumns: {}, // Init as empty object
  llmConfig: null,
  hierarchyRows: [], // Initialize as empty array
  pendingSuggestion: null,
  hierarchyIsValid: false, // Initially invalid until populated
  savedHFModels: [],
  selectedHFModel: null,
  hfModelRules: [],
  activeTaskId: null,
  activeTaskType: null,
  activeTaskStatus: null,
  classificationResults: null,

  // Actions/Setters
  setPredictionFileInfo: (info) => set({
     predictionFileInfo: info,
     selectedPredictionColumn: null // Reset column selection when file changes
    }),
  setSelectedPredictionColumn: (col) => set({ selectedPredictionColumn: col }),
  setTrainingFileInfo: (info) => set({
    trainingFileInfo: info,
    selectedTrainingTextColumn: null, // Reset selections
    selectedTrainingHierarchyColumns: {}
  }),
  setSelectedTrainingTextColumn: (col) => set({ selectedTrainingTextColumn: col }),
  setSelectedTrainingHierarchyColumns: (cols) => set({ selectedTrainingHierarchyColumns: cols }),

  setLLMConfig: (config) => set({ llmConfig: config }),

  // Hierarchy Setters
  setHierarchyRows: (rows) => set({ hierarchyRows: rows }),
  setPendingSuggestion: (suggestion) => set({ pendingSuggestion: suggestion }),
  setHierarchyIsValid: (isValid) => set({ hierarchyIsValid: isValid }),

  // HF State Setters
  setSavedHFModels: (models) => set({ savedHFModels: models }),
  setSelectedHFModel: (modelName) => set({
    selectedHFModel: modelName,
    hfModelRules: [] // Reset rules when model changes
  }),
  setHFModelRules: (rules) => set({ hfModelRules: rules }),

  // Generic Task Setters
  setActiveTask: (taskId, taskType) => set({
      activeTaskId: taskId,
      activeTaskType: taskType,
      activeTaskStatus: null, // Reset status when a new task starts
      classificationResults: null // Clear previous classification results
    }),
  setActiveTaskStatus: (status) => set({ activeTaskStatus: status }),
  setClassificationResults: (results) => set({ classificationResults: results }),

}));
