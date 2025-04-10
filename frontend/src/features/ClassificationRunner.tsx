// frontend/src/features/ClassificationRunner.tsx
import React, { useState, useEffect, useCallback } from 'react';
import { useAppStore } from '../store/store';
import { startLLMClassification, getTaskStatus } from '../services/api';
import { TaskStatus } from '../types';
// Import helper from backend utils (needs porting or separate frontend util)
// For now, define a basic version here
import { buildHierarchyFromDf } from '../utils/hierarchyUtils'; // Assuming utils file exists

const POLLING_INTERVAL_MS = 3000; // Poll every 3 seconds

const ClassificationRunner: React.FC = () => {
  const {
    llmConfig,
    predictionFileInfo,
    selectedPredictionColumn,
    hierarchyRows,
    hierarchyIsValid,
    classificationTaskId,
    setClassificationTaskId,
    classificationTaskStatus,
    setClassificationTaskStatus,
  } = useAppStore();

  const [isStarting, setIsStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pollingIntervalId, setPollingIntervalId] = useState<NodeJS.Timeout | null>(null);

  // Function to check prerequisites for starting classification
  const canStartClassification = (): boolean => {
    return (
      !!llmConfig &&
      !!predictionFileInfo &&
      !!selectedPredictionColumn &&
      hierarchyIsValid && // Use the validity flag from the store
      !classificationTaskId // Don't start if a task is already running/pending
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
    setClassificationTaskStatus(null); // Clear previous status

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
        text_column: selectedPredictionColumn,
        hierarchy: nestedHierarchy, // Send the nested structure
        llm_config: llmConfig,
      });
      setClassificationTaskId(initialStatus.task_id);
      setClassificationTaskStatus(initialStatus); // Set initial status (likely PENDING)
    } catch (err: any) {
      setError(`Error starting classification: ${err.message || 'Unknown error'}`);
      setClassificationTaskId(null);
    } finally {
      setIsStarting(false);
    }
  };

  // Function to poll task status
  const pollStatus = useCallback(async (taskId: string) => {
    try {
      const status = await getTaskStatus(taskId);
      setClassificationTaskStatus(status);

      // Stop polling if task is completed (Success or Failed)
      if (status.status === 'SUCCESS' || status.status === 'FAILED') {
        if (pollingIntervalId) {
          clearInterval(pollingIntervalId);
          setPollingIntervalId(null);
        }
        setClassificationTaskId(null); // Clear task ID once finished to allow starting new task
      }
    } catch (err: any) {
      setError(`Error polling task status: ${err.message || 'Unknown error'}`);
      // Optionally stop polling on error, or keep trying? Stopping for now.
      if (pollingIntervalId) {
        clearInterval(pollingIntervalId);
        setPollingIntervalId(null);
      }
       setClassificationTaskId(null); // Clear task ID on polling error
    }
  }, [pollingIntervalId, setClassificationTaskStatus, setClassificationTaskId]);

  // Effect to start/stop polling when taskId changes
  useEffect(() => {
    if (classificationTaskId && !pollingIntervalId) {
      // Start polling immediately and then set interval
      pollStatus(classificationTaskId);
      const intervalId = setInterval(() => pollStatus(classificationTaskId), POLLING_INTERVAL_MS);
      setPollingIntervalId(intervalId);
    } else if (!classificationTaskId && pollingIntervalId) {
      // Task finished or cleared, stop polling
      clearInterval(pollingIntervalId);
      setPollingIntervalId(null);
    }

    // Cleanup function to stop polling when component unmounts or taskId changes
    return () => {
      if (pollingIntervalId) {
        clearInterval(pollingIntervalId);
      }
    };
  }, [classificationTaskId, pollingIntervalId, pollStatus]);


  // --- Render Logic ---
  const isClassificationReady = canStartClassification();
  const currentStatus = classificationTaskStatus?.status;
  const statusMessage = classificationTaskStatus?.message;
  const downloadUrl = classificationTaskStatus?.result_url;
  // Construct full download URL using API base
  const fullDownloadUrl = downloadUrl ? `${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}${downloadUrl}` : null;


  return (
    <div>
      <h2>Run Classification</h2>
      <button
        onClick={handleStartClassification}
        disabled={!isClassificationReady || isStarting || !!classificationTaskId}
      >
        {isStarting ? 'Starting...' : '🚀 Run LLM Classification'}
      </button>

      {!isClassificationReady && !classificationTaskId && (
        <p><small>Requires: LLM Config Ready, Prediction Data Uploaded, Text Column Selected, Valid Hierarchy Defined.</small></p>
      )}

      {error && <p style={{ color: 'red' }}>Error: {error}</p>}

      {/* Task Status Display */}
      {classificationTaskStatus && (
        <div style={{ marginTop: '1rem', padding: '1rem', border: '1px solid #eee' }}>
          <h4>Task Status</h4>
          <p><strong>Status:</strong> {currentStatus}</p>
          {statusMessage && <p><strong>Message:</strong> {statusMessage}</p>}
          {currentStatus === 'RUNNING' && <p>Polling for updates...</p>}
          {currentStatus === 'SUCCESS' && fullDownloadUrl && (
            <a href={fullDownloadUrl} download target="_blank" rel="noopener noreferrer">
              <button>✅ Download Results (.xlsx)</button>
            </a>
          )}
           {currentStatus === 'FAILED' && <p style={{ color: 'red' }}>Classification Failed.</p>}
        </div>
      )}
    </div>
  );
};

export default ClassificationRunner;
