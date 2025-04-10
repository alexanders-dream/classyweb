import React, { useState, useEffect, useCallback } from 'react';
import { Box, Button, TextField, Typography, Select, MenuItem, InputLabel, FormControl, CircularProgress, Alert, Grid, Paper, SelectChangeEvent } from '@mui/material';
import { useAppStore, TaskType } from '../store/store';
import { startHFTraining, getTaskStatus, listSavedHFModels } from '../services/api';
import { TaskStatus } from '../types';

const POLLING_INTERVAL_MS = 3000;

const HFTrainingForm: React.FC = () => {
  const {
    trainingFileInfo,
    selectedTrainingTextColumn,
    setSelectedTrainingTextColumn,
    selectedTrainingHierarchyColumns,
    setSelectedTrainingHierarchyColumns,
    activeTaskId,
    activeTaskType,
    setActiveTask,
    activeTaskStatus,
    setActiveTaskStatus,
    setSavedHFModels,
  } = useAppStore();

  const [baseModel, setBaseModel] = useState<string>('bert-base-uncased');
  const [numEpochs, setNumEpochs] = useState<number>(3);
  const [newModelName, setNewModelName] = useState<string>('');
  const [isStarting, setIsStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pollingIntervalId, setPollingIntervalId] = useState<NodeJS.Timeout | null>(null);

  const isTrainingTaskRunning = activeTaskId && activeTaskType === TaskType.HF_TRAINING;
  const canStartTraining = !!trainingFileInfo && !!selectedTrainingTextColumn && Object.values(selectedTrainingHierarchyColumns).some(col => col) && !!newModelName.trim() && !isTrainingTaskRunning;

  const handleHierarchyColumnChange = (level: string, columnName: string | null) => {
    setSelectedTrainingHierarchyColumns({
      ...selectedTrainingHierarchyColumns,
      [level]: columnName === '(None)' ? null : columnName,
    });
  };

  const handleStartTraining = async () => {
    if (!canStartTraining || !trainingFileInfo || !selectedTrainingTextColumn || !newModelName.trim()) {
      setError('Please select training data, text column, at least one hierarchy column, and provide a name for the new model.');
      return;
    }

    setIsStarting(true);
    setError(null);

    const validHierarchyColumns = Object.entries(selectedTrainingHierarchyColumns)
      .filter((entry): entry is [string, string] => !!entry[1])
      .reduce((acc: Record<string, string>, [key, value]: [string, string]) => {
        acc[key] = value;
        return acc;
      }, {});

    try {
      const initialStatus = await startHFTraining({
        training_file_id: trainingFileInfo.file_id,
        original_training_filename: trainingFileInfo.filename,
        text_column: selectedTrainingTextColumn,
        hierarchy_columns: validHierarchyColumns,
        base_model: baseModel,
        num_epochs: numEpochs,
        new_model_name: newModelName.trim(),
      });
      setActiveTask(initialStatus.task_id, TaskType.HF_TRAINING);
      setActiveTaskStatus(initialStatus);
    } catch (err: any) {
      setError(`Error starting training: ${err.detail || err.message || 'Unknown error'}`);
      setActiveTask(null, null);
    } finally {
      setIsStarting(false);
    }
  };

  const pollStatus = useCallback(async () => {
    if (!activeTaskId || activeTaskType !== TaskType.HF_TRAINING) return;

    try {
      const status = await getTaskStatus(activeTaskId);
      setActiveTaskStatus(status);

      if (status.status === 'SUCCESS' || status.status === 'FAILED') {
        if (pollingIntervalId) {
          clearInterval(pollingIntervalId);
          setPollingIntervalId(null);
        }
        if (status.status === 'SUCCESS') {
          try {
            const modelsResponse = await listSavedHFModels();
            setSavedHFModels(modelsResponse.model_names);
          } catch (listError) {
            console.error("Failed to refresh saved models list after training:", listError);
          }
        }
        setActiveTask(null, null);
      }
    } catch (err: any) {
      setError(`Error polling training status: ${err.message || 'Unknown error'}`);
      if (pollingIntervalId) {
        clearInterval(pollingIntervalId);
        setPollingIntervalId(null);
      }
      setActiveTask(null, null);
    }
  }, [activeTaskId, activeTaskType, pollingIntervalId, setActiveTaskStatus, setActiveTask, setSavedHFModels]);

  useEffect(() => {
    if (isTrainingTaskRunning && !pollingIntervalId) {
      pollStatus();
      const intervalId = setInterval(pollStatus, POLLING_INTERVAL_MS);
      setPollingIntervalId(intervalId);
    } else if (!isTrainingTaskRunning && pollingIntervalId) {
      clearInterval(pollingIntervalId);
      setPollingIntervalId(null);
    }

    return () => {
      if (pollingIntervalId) {
        clearInterval(pollingIntervalId);
      }
    };
  }, [isTrainingTaskRunning, pollingIntervalId, pollStatus]);

  const hierarchyLevels = ['L1', 'L2', 'L3', 'L4'];
  const availableColumns = trainingFileInfo?.columns ?? [];
  const columnOptions = ['(None)', ...availableColumns];

  return (
    <Paper elevation={2} sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>Train Hugging Face Model</Typography>

      {!trainingFileInfo && (
        <Alert severity="info">Please upload a training dataset in the "Data Setup" tab first.</Alert>
      )}

      {trainingFileInfo && (
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Typography variant="subtitle1">Training Data:</Typography>
            <Typography variant="body2" color="text.secondary">
              {trainingFileInfo.filename} ({trainingFileInfo.num_rows} rows)
            </Typography>
          </Grid>

          <Grid item xs={12} md={6}>
            <FormControl fullWidth margin="normal">
              <InputLabel id="text-column-label">Text Column</InputLabel>
              <Select
                labelId="text-column-label"
                value={selectedTrainingTextColumn ?? ''}
                label="Text Column"
                onChange={(e: SelectChangeEvent<string>) => setSelectedTrainingTextColumn(e.target.value || null)}
              >
                {availableColumns.map((col: string) => (
                  <MenuItem key={col} value={col}>{col}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12}>
            <Typography variant="subtitle1" gutterBottom>Hierarchy Label Columns</Typography>
            <Grid container spacing={2}>
              {hierarchyLevels.map(level => (
                <Grid item xs={6} sm={3} key={level}>
                  <FormControl fullWidth margin="dense">
                    <InputLabel id={`hierarchy-${level}-label`}>{level} Column</InputLabel>
                    <Select
                      labelId={`hierarchy-${level}-label`}
                      value={selectedTrainingHierarchyColumns[level] ?? '(None)'}
                      label={`${level} Column`}
                      onChange={(e: SelectChangeEvent<string>) => handleHierarchyColumnChange(level, e.target.value)}
                    >
                      {columnOptions.map((col: string) => (
                        <MenuItem key={col} value={col}>{col}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
              ))}
            </Grid>
          </Grid>

          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              margin="normal"
              label="Base Model (e.g., bert-base-uncased)"
              value={baseModel}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setBaseModel(e.target.value)}
            />
          </Grid>
          <Grid item xs={6} md={3}>
            <TextField
              fullWidth
              margin="normal"
              label="Epochs"
              type="number"
              value={numEpochs}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setNumEpochs(Math.max(1, Math.min(10, parseInt(e.target.value, 10) || 1)))}
              inputProps={{ min: 1, max: 10 }}
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              margin="normal"
              required
              label="Save New Model As"
              value={newModelName}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setNewModelName(e.target.value)}
              helperText="Use alphanumeric, hyphens, underscores (e.g., my-model-v1)"
            />
          </Grid>

          <Grid item xs={12}>
            <Box sx={{ position: 'relative', display: 'inline-flex' }}>
              <Button
                variant="contained"
                color="primary"
                onClick={handleStartTraining}
                disabled={!canStartTraining || isStarting}
              >
                Start Training
              </Button>
              {(isStarting || isTrainingTaskRunning) && (
                <CircularProgress
                  size={24}
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

          {error && (
            <Grid item xs={12}>
              <Alert severity="error">{error}</Alert>
            </Grid>
          )}

          {isTrainingTaskRunning && activeTaskStatus && (
            <Grid item xs={12}>
              <Alert severity="info" icon={<CircularProgress size={20} />}>
                <Typography variant="body2"><strong>Training Status:</strong> {activeTaskStatus.status}</Typography>
                {activeTaskStatus.message && <Typography variant="caption">{activeTaskStatus.message}</Typography>}
              </Alert>
            </Grid>
          )}
        </Grid>
      )}
    </Paper>
  );
};

export default HFTrainingForm;
