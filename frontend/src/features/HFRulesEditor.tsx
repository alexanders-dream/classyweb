// frontend/src/features/HFRulesEditor.tsx
import React, { useState, useEffect } from 'react';
import { Box, Button, Typography, Select, MenuItem, InputLabel, FormControl, CircularProgress, Alert, Paper, Grid, TextField, Divider, SelectChangeEvent } from '@mui/material'; // Added SelectChangeEvent, TextField, Divider
import { useAppStore } from '../store/store';
import { listSavedHFModels, getHFRules, updateHFRules } from '../services/api';
import { HFRule } from '../types';

// Removed RulesTablePlaceholder


const HFRulesEditor: React.FC = () => {
  const {
    savedHFModels,
    setSavedHFModels,
    selectedHFModel,
    setSelectedHFModel,
    hfModelRules,
    setHFModelRules,
  } = useAppStore();

  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [isLoadingRules, setIsLoadingRules] = useState(false);
  const [isSavingRules, setIsSavingRules] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  // Fetch saved models on component mount
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
  }, [setSavedHFModels]);

  // Fetch rules when selected model changes
  useEffect(() => {
    if (selectedHFModel) {
      const fetchRules = async () => {
        setIsLoadingRules(true);
        setError(null);
        setSuccessMessage(null);
        try {
          const response = await getHFRules(selectedHFModel);
          setHFModelRules(response.rules);
        } catch (err: any) {
          setError(`Failed to load rules for ${selectedHFModel}: ${err.detail || err.message}`); // Fixed template literal
          setHFModelRules([]); // Clear rules on error
        } finally {
          setIsLoadingRules(false);
        }
      };
      fetchRules();
    } else {
      setHFModelRules([]); // Clear rules if no model selected
    }
  }, [selectedHFModel, setHFModelRules]);

  const handleModelSelect = (modelName: string | null) => {
    setSelectedHFModel(modelName);
  };

  // Handler for individual rule field changes
  const handleRuleFieldChange = (index: number, field: keyof HFRule, value: string | number) => {
    const updatedRules = [...hfModelRules]; // Create a copy
    const ruleToUpdate = { ...updatedRules[index] }; // Create a copy of the specific rule

    if (field === 'Keywords') {
      ruleToUpdate.Keywords = value as string;
    } else if (field === 'Confidence Threshold') {
      // Validate and parse number input
      const numValue = parseFloat(value as string);
      // Use backend validation range (0.05 - 0.95), default if invalid
      ruleToUpdate['Confidence Threshold'] = isNaN(numValue) ? 0.5 : Math.max(0.05, Math.min(0.95, numValue));
    }

    updatedRules[index] = ruleToUpdate; // Update the rule in the copied array
    setHFModelRules(updatedRules); // Update the store state
    setSuccessMessage(null); // Clear success message on edit
  };


  const handleSaveChanges = async () => {
    if (!selectedHFModel) {
      setError("No model selected to save rules for.");
      return;
    }
    setIsSavingRules(true);
    setError(null);
    setSuccessMessage(null);
    try {
      // Validation is implicitly handled by the controlled inputs and handleRuleFieldChange
      const response = await updateHFRules(selectedHFModel, hfModelRules);
      setSuccessMessage(response.message || 'Rules saved successfully!');
    } catch (err: any) {
      setError(`Failed to save rules for ${selectedHFModel}: ${err.detail || err.message}`); // Fixed template literal
    } finally {
      setIsSavingRules(false);
    }
  };

  return (
    <Paper elevation={2} sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>Manage HF Model Rules</Typography>

      <Grid container spacing={2} alignItems="center"> {/* Explicitly a container */}
        <Grid item xs={12} sm={6}> {/* Added item prop back */}
          <FormControl fullWidth margin="normal">
            <InputLabel id="hf-model-select-label">Select Model</InputLabel>
            <Select
              labelId="hf-model-select-label"
              value={selectedHFModel ?? ''}
              label="Select Model"
              onChange={(e: SelectChangeEvent<string>) => handleModelSelect(e.target.value || null)}
              disabled={isLoadingModels}
            >
              <MenuItem value="" disabled={savedHFModels.length > 0}>
                {isLoadingModels ? 'Loading models...' : (savedHFModels.length === 0 ? 'No models found' : 'Select a model')}
              </MenuItem>
              {savedHFModels.map((name: string) => <MenuItem key={name} value={name}>{name}</MenuItem>)}
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} sm={6}> {/* Added item prop back */}
           {/* Maybe add a refresh button */}
        </Grid>

        {error && (
          <Grid item xs={12}> {/* Added item prop back */}
            <Alert severity="error">{error}</Alert>
          </Grid>
        )}
         {successMessage && (
          <Grid item xs={12}> {/* Added item prop back */}
            <Alert severity="success">{successMessage}</Alert>
          </Grid>
         )}

        {/* Rules Section - Conditionally Rendered */}
        {selectedHFModel &&
          <Grid item xs={12}> {/* Added item prop back */}
            <Typography variant="subtitle1" sx={{ mt: 2 }}>Rules for: {selectedHFModel}</Typography>

            {/* Loading State */}
            {isLoadingRules
              ? <Box sx={{ display: 'flex', justifyContent: 'center', my: 3 }}><CircularProgress /></Box>
              /* Rules Display */
              : <Box sx={{ mt: 2 }}>
                  {/* Header Row */}
                  <Grid container spacing={2} sx={{ mb: 1, borderBottom: 1, borderColor: 'divider', pb: 1 }}>
                    <Grid item xs={4}><Typography variant="subtitle2">Label</Typography></Grid> {/* Added item prop back */}
                    <Grid item xs={5}><Typography variant="subtitle2">Keywords (comma-separated)</Typography></Grid> {/* Added item prop back */}
                    <Grid item xs={3}><Typography variant="subtitle2">Threshold (0.05-0.95)</Typography></Grid> {/* Added item prop back */}
                  </Grid>

                  {/* No Rules Message */}
                  {hfModelRules.length === 0 &&
                    <Typography sx={{mt: 2}}>No rules found or loaded for this model.</Typography>
                  }

                  {/* Mapped Rules */}
                  {hfModelRules.map((rule: HFRule, index: number) => (
                    <Grid container spacing={2} key={rule.Label || index} sx={{ mb: 1.5, alignItems: 'center' }}>
                      <Grid item xs={4}> {/* Added item prop back */}
                        <Typography variant="body2" sx={{ wordBreak: 'break-word' }}>{rule.Label}</Typography>
                      </Grid>
                      <Grid item xs={5}> {/* Added item prop back */}
                        <TextField
                          fullWidth
                          variant="outlined"
                          size="small"
                          value={rule.Keywords}
                          onChange={(e: React.ChangeEvent<HTMLInputElement>) => handleRuleFieldChange(index, 'Keywords', e.target.value)}
                        />
                      </Grid>
                      <Grid item xs={3}> {/* Added item prop back */}
                         <TextField
                          fullWidth
                          variant="outlined"
                          size="small"
                          type="number"
                          value={rule['Confidence Threshold']}
                          onChange={(e: React.ChangeEvent<HTMLInputElement>) => handleRuleFieldChange(index, 'Confidence Threshold', e.target.value)}
                          inputProps={{ step: 0.01, min: 0.05, max: 0.95 }}
                        />
                      </Grid>
                    </Grid>
                  ))}

                  {/* Save Button - Conditionally Rendered */}
                  {hfModelRules.length > 0 &&
                    <Box sx={{ mt: 3, position: 'relative', display: 'inline-flex' }}>
                      <Button
                        variant="contained"
                        color="primary"
                        onClick={handleSaveChanges}
                        disabled={isSavingRules || isLoadingRules}
                      >
                        Save Changes
                      </Button>
                      {/* Saving Spinner - Conditionally Rendered */}
                      {isSavingRules &&
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
                      }
                    </Box>
                  }
                </Box>
            }
          </Grid>
        }
      </Grid>
    </Paper>
  );
};

// Use default export
export default HFRulesEditor;
