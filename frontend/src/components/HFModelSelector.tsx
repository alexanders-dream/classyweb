import React, { useState, useEffect } from 'react';
import { Select, MenuItem, FormControl, InputLabel, Box, Typography, CircularProgress } from '@mui/material';
import { listSavedHFModels } from '../services/api'; // Corrected function name again based on error
import { useAppStore } from '../store/store'; // To interact with global state

const HFModelSelector: React.FC = () => {
  const [models, setModels] = useState<string[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Get state and actions from Zustand store
  const selectedHFModel = useAppStore((state) => state.selectedHFModel); // Corrected state name
  const setSelectedHFModel = useAppStore((state) => state.setSelectedHFModel); // Corrected setter name

  useEffect(() => {
    const loadModels = async () => {
      setLoading(true);
      setError(null);
      try {
        // Fetch models using the corrected API function name
        const response = await listSavedHFModels(); // Corrected function name
        setModels(response.model_names); // Extract the model_names array
        // Optionally select the first model by default if none is selected
        // if (!selectedHFModel && response.model_names.length > 0) {
        //   setSelectedHfModel(savedModels[0]);
        // }
      } catch (err) {
        console.error("Failed to fetch saved HF models:", err);
        setError("Failed to load models.");
        setModels([]); // Clear models on error
      } finally {
        setLoading(false);
      }
    };

    loadModels();
    // Add selectedHFModel to dependency array if auto-selection logic is uncommented
  }, [setSelectedHFModel]); // Corrected setter name

  const handleModelChange = (event: React.ChangeEvent<{ value: unknown }>) => {
    // Ensure the value is treated as a string before updating state
    const value = event.target.value as string;
    setSelectedHFModel(value); // Corrected setter name
  };


  return (
    <Box sx={{ mt: 2 }}>
      <FormControl fullWidth error={!!error}>
        <InputLabel id="hf-model-select-label">Select Saved Model</InputLabel>
        <Select
          labelId="hf-model-select-label"
          id="hf-model-select"
          value={selectedHFModel || ''} // Corrected state name
          label="Select Saved Model"
          // The type assertion for onChange is slightly different in MUI v5+ for Select
          onChange={(event) => handleModelChange(event as React.ChangeEvent<{ value: unknown }>)}
          disabled={loading || !!error || models.length === 0}
        >
          {loading && (
            <MenuItem value="" disabled>
              <CircularProgress size={20} sx={{ mr: 1 }} /> Loading...
            </MenuItem>
          )}
          {!loading && models.length === 0 && !error && (
             <MenuItem value="" disabled>No saved models found.</MenuItem>
          )}
           {!loading && error && (
             <MenuItem value="" disabled>Error loading models.</MenuItem>
          )}
          {!loading && models.map((modelName) => (
            <MenuItem key={modelName} value={modelName}>
              {modelName}
            </MenuItem>
          ))}
        </Select>
        {error && <Typography color="error" variant="caption">{error}</Typography>}
      </FormControl>
    </Box>
  );
};

export default HFModelSelector; // Ensure default export
