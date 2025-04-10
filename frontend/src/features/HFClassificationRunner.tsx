{// frontend/src/features/HFClassificationRunner.tsx
import React, { useState, useEffect, useCallback } from 'react';
import { Box, Button, Typography, Select, MenuItem, InputLabel, FormControl, CircularProgress, Alert, Paper, Grid, SelectChangeEvent } from '@mui/material'; // Added SelectChangeEvent
import { useAppStore, TaskType } from '../store/store';
import { listSavedHFModels, startHFClassification, getTaskStatus } from '../services/api';
import { TaskStatus } from '../types';

const POLLING_INTERVAL_MS = 3000;

const HFClassificationRunner: React.FC = () => {
  const {
    predictionFileInfo,
    selectedPredictionColumn,
    savedHFModels,
    setSavedHFModels,
    selectedHFModel,
    setSelectedHFModel,
    activeTaskId,
    activeTaskType,
    setActiveTask,
    activeTaskStatus,
    setActiveTaskStatus,
    setClassificationResults, // To clear previous results
  } = useAppStore();

  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pollingIntervalId, setPollingIntervalId] = useState<ReturnType<typeof setInterval> | null>(null); // Use ReturnType

  // --- Derived State ---
  const isClassificationTaskRunning = activeTaskId && activeTaskType === TaskType.HF_CLASSIFICATION;
  const canStartClassification =
    !!predictionFileInfo &&
    !!selectedPredictionColumn &&
    !!selectedHFModel &&
    !isClassificationTaskRunning; // Correct variable name

  // Fetch saved models on mount
  useEffect(() => {
    const fetchModels = async () => {
      setIsLoadingModels(true);
      setError(null);
      try {
        const response = await listSavedHFModels();
        setSavedHFModels(response.model_names);
      } catch (err: any) {
        setError(`Failed to load saved models: ${err.detail || err.message}`); // Fixed template literal
      } finally {
        setIsLoadingModels(false);
      }
    };
    fetchModels();
    // Only run on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [setSavedHFModels]);

  // --- Handlers ---
  const handleModelSelect = (modelName: string | null) => {
    setSelectedHFModel(modelName);
  };

  const handleStartClassification = async () => {
    if (!canStartClassification || !predictionFileInfo || !selectedPredictionColumn || !selectedHFModel) {
      setError('Please select prediction data, text column, and a saved HF model.');
      return;
    }

    setIsStarting(true);
    setError(null);
    setClassificationResults(null); // Clear previous results

    try {
      const initialStatus = await startHFClassification({
        file_id: predictionFileInfo.file_id,
        original_filename: predictionFileInfo.filename,
        text_column: selectedPredictionColumn,
        model_name: selectedHFModel,
      });
      setActiveTask(initialStatus.task_id, TaskType.HF_CLASSIFICATION);
      setActiveTaskStatus(initialStatus);
    } catch (err: any) {
      setError(`Error starting HF classification: ${err.detail || err.message || 'Unknown error'}`); // Fixed template literal
      setActiveTask(null, null); // Clear task on error
    } finally {
      setIsStarting(false);
    }
  };

  // --- Polling Logic (Similar to Training) ---
   const pollStatus = useCallback(async () => {
    if (!activeTaskId || activeTaskType !== TaskType.HF_CLASSIFICATION) return;

    try {
      const status = await getTaskStatus(activeTaskId);
      setActiveTaskStatus(status);

      if (status.status === 'SUCCESS' || status.status === 'FAILED') {
        if (pollingIntervalId) {
          clearInterval(pollingIntervalId);
          setPollingIntervalId(null);
        }
        // Don't clear task ID on success, wait for results fetch in App.tsx effect
         if (status.status === 'FAILED') {
            setActiveTask(null, null); // Clear task ID and type on failure
            setClassificationResults(null); // Clear results on failure
        }
      }
    } catch (err: any) {
      setError(`Error polling classification status: ${err.message || 'Unknown error'}`); // Fixed template literal
      if (pollingIntervalId) {
        clearInterval(pollingIntervalId);
        setPollingIntervalId(null);
      }
      setActiveTask(null, null); // Clear task on polling error
    }
  }, [activeTaskId, activeTaskType, pollingIntervalId, setActiveTaskStatus, setActiveTask, setClassificationResults]); // Added setClassificationResults

  useEffect(() => {
    if (isClassificationTaskRunning && !pollingIntervalId) {
      pollStatus(); // Initial poll
      const intervalId = setInterval(pollStatus, POLLING_INTERVAL_MS);
      setPollingIntervalId(intervalId);
    } else if (!isClassificationTaskRunning && pollingIntervalId) {
      clearInterval(pollingIntervalId);
      setPollingIntervalId(null);
    }

    return () => { // Cleanup
      if (pollingIntervalId) {
        clearInterval(pollingIntervalId);
      }
    };
  }, [isClassificationTaskRunning, pollingIntervalId, pollStatus]);


  // --- Render ---
  return (
    <Paper elevation={2} sx={{ p: 3, mt: 2 }}> {/* Added margin top */}
      <Typography variant="h6" gutterBottom>Run HF Classification</Typography>

       {!predictionFileInfo && (
        <Alert severity="info">Please upload prediction data in the "Data Setup" tab first.</Alert>
      )}

      {predictionFileInfo && (
        <Grid container spacing={3} alignItems="center">
           <Grid item xs={12} sm={6}>
             <FormControl fullWidth margin="normal">
                <InputLabel id="hf-classify-model-select-label">Select Saved Model</InputLabel>
                <Select
                labelId="hf-classify-model-select-label"
                value={selectedHFModel ?? ''}
                label="Select Saved Model"
                onChange={(e: SelectChangeEvent<string>) => handleModelSelect(e.target.value || null)} // Added type for e
                disabled={isLoadingModels || isClassificationTaskRunning}
                >
                <MenuItem value="" disabled={savedHFModels.length > 0}>
                    {isLoadingModels ? 'Loading models...' : (savedHFModels.length === 0 ? 'No models trained/found' : 'Select a model')}
                </MenuItem>
                {savedHFModels.map((name: string) => <MenuItem key={name} value={name}>{name}</MenuItem>)} {/* Added type for name */}
                </Select>
            </FormControl>
           </Grid>

           <Grid item xs={12} sm={6}>
             <Box sx={{ position: 'relative', display: 'inline-flex', mt: { xs: 0, sm: 2} }}> {/* Adjust margin for alignment */}
                <Button
                    variant="contained"
                    color="secondary"
                    onClick={handleStartClassification}
                    disabled={!canStartClassification || isStarting}
                >
                    Run HF Classification
                </Button>
                {(isStarting || isClassificationTaskRunning) && (
                    <CircularProgress
                    size={24}
                    color="secondary"
                    sx={{
                        position: 'absolute',
                        top: '50%',
                        left: '50%',
                        marginTop: '-12px',
                        marginLeft: '-12px',
                    }}
                    />
                )}
                </Box>
           </Grid>

            {/* Error Display */}
            {error && (
                <Grid item xs={12}>
                <Alert severity="error">{error}</Alert>
                </Grid>
            )}

            {/* Task Status Display */}
            {isClassificationTaskRunning && activeTaskStatus && (
                <Grid item xs={12}>
                <Alert severity="info" icon={<CircularProgress size={20} />}>
                    <Typography variant="body2"><strong>Classification Status:</strong> {activeTaskStatus.status}</Typography>
                    {activeTaskStatus.message && <Typography variant="caption">{activeTaskStatus.message}</Typography>}
                </Alert>
                </Grid>
            )}
             {/* Success is handled by the results display appearing */}

        </Grid>
      )}
    </Paper>
  );
};

export default HFClassificationRunner;
