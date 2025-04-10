// frontend/src/features/ClassificationRunner.tsx
import React, { useState, useEffect, useCallback } from 'react';
import { useAppStore, TaskType } from '../store/store'; // Import TaskType
import { startLLMClassification, getTaskStatus, getResultData } from '../services/api'; // Added getResultData
import { TaskStatus, ClassificationResultRow } from '../types'; // Added ClassificationResultRow
// Import helper from backend utils (needs porting or separate frontend util)
import { buildHierarchyFromDf } from '../utils/hierarchyUtils'; // Assuming utils file exists

const POLLING_INTERVAL_MS = 3000; // Poll every 3 seconds

const ClassificationRunner: React.FC = () => {
  const {
    llmConfig,
    predictionFileInfo,
    selectedPredictionColumn,
    hierarchyRows,
    hierarchyIsValid,
    activeTaskId,         // Use generic task ID
    activeTaskType,       // Use generic task type
    setActiveTask,        // Use generic setter
    activeTaskStatus,     // Use generic status
    setActiveTaskStatus,  // Use generic status setter
    setClassificationResults,
  } = useAppStore();

  const [isStarting, setIsStarting] = useState(false);
  const [isFetchingResults, setIsFetchingResults] = useState(false); // Added state for fetching results
  const [error, setError] = useState<string | null>(null);
  const [pollingIntervalId, setPollingIntervalId] = useState<NodeJS.Timeout | null>(null);

  // Function to check prerequisites for starting classification
  const canStartClassification = (): boolean => {
    return (
      !!llmConfig &&
      !!predictionFileInfo &&
      !!selectedPredictionColumn &&
      hierarchyIsValid && // Use the validity flag from the store
      !activeTaskId // Don't start if a task is already running/pending
    );
  };

  // Function to start the classification task
  const handleStartClassification = async () => {
    if (!canStartClassification() || !predictionFileInfo || !selectedPredictionColumn || !llmConfig) {
      setError("Cannot start classification. Ensure LLM is configured, data is uploaded with a selected text column, and a valid hierarchy is defined.");
      return;
    }

    setIsStarting(true);
    setError(null);
    // setActiveTaskStatus(null); // Resetting status is handled by setActiveTask
    setClassificationResults(null); // Explicitly clear previous results when starting

    // Build the nested hierarchy from the rows in the store
    const nestedHierarchy = buildHierarchyFromDf(hierarchyRows);
    if (!nestedHierarchy || !nestedHierarchy.themes || nestedHierarchy.themes.length === 0) {
        setError("Failed to build a valid nested hierarchy from the editor rows.");
        setIsStarting(false);
        return;
    }

    try {
      const initialStatus = await startLLMClassification({
        file_id: predictionFileInfo.file_id,
        original_filename: predictionFileInfo.filename,
        text_column: selectedPredictionColumn,
        hierarchy: nestedHierarchy, // Send the nested structure
        llm_config: llmConfig,
      });
      // Set the active task ID and type
      setActiveTask(initialStatus.task_id, TaskType.LLM_CLASSIFICATION);
      setActiveTaskStatus(initialStatus); // Set initial status (likely PENDING)
    } catch (err: any) {
      setError(`Error starting classification: ${err.message || 'Unknown error'}`);
      setActiveTask(null, null); // Clear task ID and type on start error
      setClassificationResults(null); // Clear results on start error
    } finally {
      setIsStarting(false);
    }
  };

  // Function to poll task status - uses activeTaskId from store
  const pollStatus = useCallback(async () => {
    if (!activeTaskId) return; // Don't poll if no active task

    try {
      const status = await getTaskStatus(activeTaskId);
      setActiveTaskStatus(status);

      // Stop polling if task is completed (Success or Failed)
      if (status.status === 'SUCCESS' || status.status === 'FAILED') {
        if (pollingIntervalId) {
          clearInterval(pollingIntervalId);
          setPollingIntervalId(null);
        }
        // Don't clear task ID immediately on success if it's a classification task, wait for results fetch
        if (status.status === 'FAILED' || (status.status === 'SUCCESS' && activeTaskType === TaskType.HF_TRAINING)) {
            setActiveTask(null, null); // Clear task ID and type on failure or training success
            if (status.status === 'FAILED') {
                setClassificationResults(null); // Clear results on failure
            }
        }
      }
    } catch (err: any) {
      setError(`Error polling task status: ${err.message || 'Unknown error'}`);
      setClassificationResults(null); // Clear results on polling error
      // Optionally stop polling on error, or keep trying? Stopping for now.
      if (pollingIntervalId) {
        clearInterval(pollingIntervalId);
        setPollingIntervalId(null);
      }
       setActiveTask(null, null); // Clear task ID and type on polling error
    }
  }, [activeTaskId, activeTaskType, pollingIntervalId, setActiveTaskStatus, setActiveTask, setClassificationResults]); // Added dependencies

  // Effect to start/stop polling when activeTaskId changes
  useEffect(() => {
    if (activeTaskId && !pollingIntervalId) {
      // Start polling immediately and then set interval
      pollStatus(); // Call without taskId, it uses activeTaskId from store
      const intervalId = setInterval(pollStatus, POLLING_INTERVAL_MS);
      setPollingIntervalId(intervalId);
    } else if (!activeTaskId && pollingIntervalId) {
      // Task finished or cleared, stop polling
      clearInterval(pollingIntervalId);
      setPollingIntervalId(null);
    }

    // Cleanup function to stop polling when component unmounts or activeTaskId changes
    return () => {
      if (pollingIntervalId) {
        clearInterval(pollingIntervalId);
      }
    };
  }, [activeTaskId, pollingIntervalId, pollStatus]);

  // Effect to fetch results when status becomes SUCCESS for classification tasks
  useEffect(() => {
    // Only fetch results for classification tasks
    if (activeTaskStatus?.status === 'SUCCESS' && activeTaskId &&
        (activeTaskType === TaskType.LLM_CLASSIFICATION || activeTaskType === TaskType.HF_CLASSIFICATION))
    {
      const fetchResults = async (taskId: string) => {
        setIsFetchingResults(true);
        setError(null); // Clear previous errors
        try {
          const results = await getResultData(taskId);
          setClassificationResults(results);
          // Now clear the task ID and type as results are fetched
          setActiveTask(null, null);
        } catch (err: any) {
          setError(`Error fetching results: ${err.message || 'Unknown error'}`);
          setClassificationResults(null); // Clear results on fetch error
          setActiveTask(null, null); // Also clear task ID/type on fetch error
        } finally {
          setIsFetchingResults(false);
        }
      };
      fetchResults(activeTaskId);
    }
  }, [activeTaskStatus, activeTaskId, activeTaskType, setClassificationResults, setActiveTask]); // Dependencies


  // --- Render Logic ---
  const isClassificationReady = canStartClassification();
  const currentStatus = activeTaskStatus?.status;
  const statusMessage = activeTaskStatus?.message;


  // Determine button text and disabled state based on active task
  const isTaskRunning = !!activeTaskId;
  const buttonText = isStarting ? 'Starting...' : '🚀 Run LLM Classification'; // Keep specific for now, will generalize later
  const buttonDisabled = !isClassificationReady || isStarting || isTaskRunning;

  return (
    <div>
      <h2>Run Classification (LLM)</h2> {/* Keep specific for now */}
      <button
        onClick={handleStartClassification}
        disabled={buttonDisabled}
      >
        {buttonText}
      </button>

      {!isClassificationReady && !isTaskRunning && (
        <p><small>Requires: LLM Config Ready, Prediction Data Uploaded, Text Column Selected, Valid Hierarchy Defined.</small></p>
      )}

      {error && <p style={{ color: 'red' }}>Error: {error}</p>}

      {/* Task Status Display */}
      {activeTaskStatus && activeTaskType === TaskType.LLM_CLASSIFICATION && ( // Only show for LLM tasks for now
        <div style={{ marginTop: '1rem', padding: '1rem', border: '1px solid #eee' }}>
          <h4>Task Status ({activeTaskType})</h4>
          <p><strong>Task ID:</strong> {activeTaskId}</p>
          <p><strong>Status:</strong> {currentStatus ?? 'Idle'}</p>
          {statusMessage && <p><strong>Message:</strong> {statusMessage}</p>}
          {currentStatus === 'RUNNING' && <p>Polling for updates...</p>}
          {currentStatus === 'SUCCESS' && isFetchingResults && <p>Fetching results...</p>}
          {/* Removed redundant check */}
          {currentStatus === 'SUCCESS' && !isFetchingResults && (
              <p style={{ color: 'green' }}>✅ Classification successful! Results available.</p>
          )}
           {currentStatus === 'FAILED' && <p style={{ color: 'red' }}>Task Failed.</p>}
        </div>
      )}
    </div>
  );
};

export default ClassificationRunner;
