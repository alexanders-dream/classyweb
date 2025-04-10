// frontend/src/store/store.ts
import { create } from 'zustand';
// Import necessary types
import { FileInfo, LLMProviderConfig, HierarchyRow, NestedHierarchySuggestion, TaskStatus, ClassificationResultRow } from '../types'; // Added TaskStatus, ClassificationResultRow

interface AppState {
  // Data & Files
  predictionFileInfo: FileInfo | null;
  setPredictionFileInfo: (info: FileInfo | null) => void;
  selectedPredictionColumn: string | null; // Added
  setSelectedPredictionColumn: (col: string | null) => void; // Added

  // Optional: Add state for training file later
  // trainingFileInfo: FileInfo | null;
  // setTrainingFileInfo: (info: FileInfo | null) => void;
  // selectedTrainingColumns: Record<string, string | null>;
  // setSelectedTrainingColumns: (cols: Record<string, string | null>) => void;

  // LLM Config
  llmConfig: LLMProviderConfig | null;
  setLLMConfig: (config: LLMProviderConfig | null) => void;

  // Hierarchy Editor State
  hierarchyRows: HierarchyRow[]; // Array of rows for the editor
  setHierarchyRows: (rows: HierarchyRow[]) => void;
  pendingSuggestion: NestedHierarchySuggestion | null; // Holds AI suggestion before applying
  setPendingSuggestion: (suggestion: NestedHierarchySuggestion | null) => void;
  hierarchyIsValid: boolean; // Flag if current rows form a valid structure
  setHierarchyIsValid: (isValid: boolean) => void;

  // Classification Task State
  classificationTaskId: string | null;
  setClassificationTaskId: (taskId: string | null) => void;
  classificationTaskStatus: TaskStatus | null;
  setClassificationTaskStatus: (status: TaskStatus | null) => void;
  classificationResults: ClassificationResultRow[] | null; // Added state for results
  setClassificationResults: (results: ClassificationResultRow[] | null) => void; // Added setter


  // Add more state slices as needed in later phases...
}

export const useAppStore = create<AppState>((set) => ({
  // Initial values
  predictionFileInfo: null,
  selectedPredictionColumn: null,
  llmConfig: null,
  hierarchyRows: [], // Initialize as empty array
  pendingSuggestion: null,
  hierarchyIsValid: false, // Initially invalid until populated
  classificationTaskId: null,
  classificationTaskStatus: null,
  classificationResults: null, // Added initial value

  // Actions/Setters
  setPredictionFileInfo: (info) => set({
     predictionFileInfo: info,
     selectedPredictionColumn: null // Reset column selection when file changes
    }),
  setSelectedPredictionColumn: (col) => set({ selectedPredictionColumn: col }),
  setLLMConfig: (config) => set({ llmConfig: config }),

  // Hierarchy Setters
  setHierarchyRows: (rows) => set({ hierarchyRows: rows }),
  setPendingSuggestion: (suggestion) => set({ pendingSuggestion: suggestion }),
  setHierarchyIsValid: (isValid) => set({ hierarchyIsValid: isValid }),

  // Classification Task Setters
  setClassificationTaskId: (taskId) => set({
      classificationTaskId: taskId,
      classificationTaskStatus: null // Reset status, but DON'T clear results here
      // classificationResults: null // REMOVED: Results should persist until a new task starts
    }),
  setClassificationTaskStatus: (status) => set({ classificationTaskStatus: status }),
  setClassificationResults: (results) => set({ classificationResults: results }), // Added setter

}));
