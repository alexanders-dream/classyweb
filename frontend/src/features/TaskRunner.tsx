import React, { useState, useEffect, useCallback } from 'react';
import { useAppStore, AppState } from '../store/store'; // Import AppState for typing selectors
import { getTaskStatus, getResultData } from '../services/api';
import { TaskStatus, ClassificationResultRow } from '../types';
import axios from 'axios'; // Needed for error checking

const POLLING_INTERVAL_MS = 3000; // Poll every 3 seconds

// Define the props the generic TaskRunner will accept
interface TaskRunnerProps {
  taskTypeLabel: string; // User-friendly label (e.g., "LLM Classification", "HF Training")
  taskIdSelector: (state: AppState) => string | null; // Selector for the specific task ID
  taskStatusSelector: (state: AppState) => TaskStatus | null; // Selector for the specific task status
  setTaskIdAction: (taskId: string | null) => void; // Action to set the task ID
  setTaskStatusAction: (status: TaskStatus | null) => void; // Action to set the task status
  clearTaskAction: () => void; // Action to clear task ID and status
  startTaskFunction: () => Promise<TaskStatus>; // Function that calls the backend API to start the task
  startButtonText: string; // Text for the start button (e.g., "🚀 Run LLM Classification")
  canStartTask: () => boolean; // Function to check if prerequisites are met
  prerequisitesMessage: string; // Message to show if prerequisites are not met
  isClassificationTask: boolean; // Flag to indicate if results need fetching/setting
  onSuccess?: (resultDetail?: string | null) => void; // Optional callback on success
}

const TaskRunner: React.FC<TaskRunnerProps> = ({
  taskTypeLabel,
  taskIdSelector,
  taskStatusSelector,
  setTaskIdAction,
  setTaskStatusAction,
  clearTaskAction,
  startTaskFunction,
  startButtonText,
  canStartTask,
  prerequisitesMessage,
  isClassificationTask, // Use this flag
  onSuccess,
}) => {
  // Select necessary state and actions from the store
  // We select setClassificationResults unconditionally, but use it conditionally later
  const { setClassificationResults, taskId, taskStatus } = useAppStore((state: AppState) => ({
      setClassificationResults: state.setClassificationResults, // Always select the action
      taskId: taskIdSelector(state), // Use the selector prop
      taskStatus: taskStatusSelector(state), // Use the selector prop
  }));

  // Component's internal state
  const [isStarting, setIsStarting] = useState(false);
  const [isFetchingResults, setIsFetchingResults] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pollingIntervalId, setPollingIntervalId] = useState<NodeJS.Timeout | null>(null);

  const isReadyToStart = canStartTask();

  const handleStartTask = async () => {
    if (!isReadyToStart) {
      setError(prerequisitesMessage);
      return;
    }

    setIsStarting(true);
    setError(null);
      clearTaskAction(); // Clear previous task info for this type
      // Only clear results if it's a classification task AND the action exists
      if (isClassificationTask && setClassificationResults) {
          setClassificationResults(null);
      }

      try {
      const initialStatus = await startTaskFunction(); // Call the passed start function
      setTaskIdAction(initialStatus.task_id);
      setTaskStatusAction(initialStatus);
    } catch (err) {
      console.error(`Error starting ${taskTypeLabel} task:`, err);
      let errorMsg = `Error starting ${taskTypeLabel} task: Unknown error`;
       if (axios.isAxiosError(err) && err.response) {
           errorMsg = err.response.data?.detail || err.message || JSON.stringify(err.response.data);
       } else if (err instanceof Error) {
           errorMsg = err.message;
       }
      setError(errorMsg);
      clearTaskAction();
      // Only clear results if it's a classification task AND the action exists
      if (isClassificationTask && setClassificationResults) {
        setClassificationResults(null);
      }
    } finally {
      setIsStarting(false);
    }
  };

  // --- Polling Logic (Mostly unchanged, uses generic actions) ---
  const pollStatus = useCallback(async (id: string) => {
    try {
      const status = await getTaskStatus(id);
      setTaskStatusAction(status); // Update the specific task status in the store

      if (status.status === 'SUCCESS' || status.status === 'FAILED') {
        if (pollingIntervalId) {
          clearInterval(pollingIntervalId);
          setPollingIntervalId(null);
        }
        if (status.status === 'FAILED') {
            clearTaskAction(); // Clear task ID on failure
            // Only clear results if it's a classification task AND the action exists
            if (isClassificationTask && setClassificationResults) {
                setClassificationResults(null);
            }
        }
        // Don't clear task ID on success yet, wait for results/details processing
      }
    } catch (err) {
      console.error(`Error polling task status for ${taskId}:`, err);
      let errorMsg = `Error polling task status: Unknown error`;
       if (axios.isAxiosError(err) && err.response) {
           errorMsg = err.response.data?.detail || err.message || JSON.stringify(err.response.data);
       } else if (err instanceof Error) {
           errorMsg = err.message;
       }
      setError(errorMsg);
      if (pollingIntervalId) {
        clearInterval(pollingIntervalId);
        setPollingIntervalId(null);
      }
      clearTaskAction(); // Clear task ID on polling error
      // Only clear results if it's a classification task AND the action exists
      if (isClassificationTask && setClassificationResults) {
        setClassificationResults(null);
      }
    }
  }, [pollingIntervalId, setTaskStatusAction, clearTaskAction, setClassificationResults, isClassificationTask, taskId]);

  // Effect to start/stop polling
  useEffect(() => {
    if (taskId && !pollingIntervalId) {
      pollStatus(taskId); // Poll immediately
      const intervalId = setInterval(() => pollStatus(taskId), POLLING_INTERVAL_MS);
      setPollingIntervalId(intervalId);
    } else if (!taskId && pollingIntervalId) {
      clearInterval(pollingIntervalId);
      setPollingIntervalId(null);
    }
    return () => { // Cleanup
      if (pollingIntervalId) clearInterval(pollingIntervalId);
    };
  }, [taskId, pollingIntervalId, pollStatus]);

  // Effect to handle SUCCESS state
  useEffect(() => {
    if (taskStatus?.status === 'SUCCESS' && taskStatus.task_id) {
      const handleSuccess = async (taskId: string, resultDetail?: string | null, resultDataUrl?: string | null) => {
        // Handle classification results fetching
        if (isClassificationTask && resultDataUrl) {
          setIsFetchingResults(true);
          setError(null);
          try {
            const results = await getResultData(taskId);
            setClassificationResults(results);
            console.log(`Fetched results for ${taskTypeLabel} task ${taskId}`);
            if (onSuccess) onSuccess(resultDetail); // Call onSuccess even for classification
          } catch (err) {
             console.error(`Error fetching results for ${taskTypeLabel} task ${taskId}:`, err);
             let errorMsg = `Error fetching results: Unknown error`;
             if (axios.isAxiosError(err) && err.response) {
                 errorMsg = err.response.data?.detail || err.message || JSON.stringify(err.response.data);
             } else if (err instanceof Error) {
                 errorMsg = err.message;
             }
              setError(errorMsg);
              // Only clear results if it's a classification task AND the action exists
              if (isClassificationTask && setClassificationResults) {
                setClassificationResults(null);
              }
            } finally {
              setIsFetchingResults(false);
              // Clear task ID after attempting to fetch results (success or fail)
               clearTaskAction();
            }
          } else if (!isClassificationTask) { // Handle success for non-classification tasks (like training)
            console.log(`${taskTypeLabel} task ${taskId} succeeded. Detail: ${resultDetail}`);
            if (onSuccess) {
                onSuccess(resultDetail); // Pass result detail (e.g., saved model name)
            }
             // Clear task ID after handling success
             clearTaskAction();
        } else {
             // Successful classification task but no data URL? Clear task ID.
             console.warn(`Task ${taskId} succeeded but no result data URL found.`);
             clearTaskAction();
        }
      };

      handleSuccess(taskStatus.task_id, taskStatus.result_detail, taskStatus.result_data_url);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [taskStatus]); // Dependency: only run when taskStatus changes


  // --- Render Logic ---
  const currentStatus = taskStatus?.status;
  const statusMessage = taskStatus?.message;
  const resultDetail = taskStatus?.result_detail; // e.g., saved model name

  // Basic Button component (replace with your UI library)
  const Button = ({ children, onClick, disabled }: { children: React.ReactNode, onClick: () => void, disabled?: boolean }) => (
      <button onClick={onClick} disabled={disabled} style={{ padding: '8px 15px', cursor: disabled ? 'not-allowed' : 'pointer' }}>
          {children}
      </button>
  );


  return (
    <div style={{ marginTop: '1rem', padding: '1rem', border: '1px solid #eee', borderRadius: '5px' }}>
      <h4>{taskTypeLabel}</h4>
      <Button
        onClick={handleStartTask}
        disabled={!isReadyToStart || isStarting || !!taskId}
      >
        {isStarting ? 'Starting...' : startButtonText}
      </Button>

      {!isReadyToStart && !taskId && (
        <p style={{ fontSize: '0.9em', color: '#666' }}><small>{prerequisitesMessage}</small></p>
      )}

      {error && <p style={{ color: 'red', marginTop: '10px' }}>Error: {error}</p>}

      {/* Task Status Display */}
      {taskStatus && (
        <div style={{ marginTop: '1rem' }}>
          <p><strong>Status:</strong> {currentStatus ?? 'Idle'}</p>
          {statusMessage && <p><strong>Message:</strong> {statusMessage}</p>}
          {currentStatus === 'RUNNING' && <p><em>Polling for updates...</em></p>}
          {currentStatus === 'SUCCESS' && isFetchingResults && <p><em>Fetching results...</em></p>}
          {currentStatus === 'SUCCESS' && !isFetchingResults && isClassificationTask && (
              <p style={{ color: 'green' }}>✅ Classification successful! Results fetched.</p>
          )}
           {currentStatus === 'SUCCESS' && !isClassificationTask && (
              <p style={{ color: 'green' }}>✅ Task successful! {resultDetail ? `(${resultDetail})` : ''}</p>
          )}
           {currentStatus === 'FAILED' && <p style={{ color: 'red' }}>Task Failed.</p>}
        </div>
      )}
    </div>
  );
};

export default TaskRunner;
