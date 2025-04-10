// frontend/src/store/store.ts
import { create } from 'zustand';
import { FileInfo, LLMProviderConfig } from '../types'; // Removed unused TaskStatus for now

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

  // Add more state slices as needed in later phases...
}

export const useAppStore = create<AppState>((set) => ({
  // Initial values
  predictionFileInfo: null,
  selectedPredictionColumn: null, // Added
  llmConfig: null,

  // Actions/Setters
  setPredictionFileInfo: (info) => set({
     predictionFileInfo: info,
     selectedPredictionColumn: null // Reset column selection when file changes
    }),
  setSelectedPredictionColumn: (col) => set({ selectedPredictionColumn: col }), // Added
  setLLMConfig: (config) => set({ llmConfig: config }),

}));