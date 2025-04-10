// frontend/src/features/DataSetup.tsx
import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Box, Typography, Paper, Button, CircularProgress, Alert,
  TableContainer, Table, TableHead, TableRow, TableCell, TableBody,
  FormControl, InputLabel, Select, MenuItem, SelectChangeEvent
} from '@mui/material';
import { useAppStore } from '../store/store'; // Import zustand store hook
import { uploadFile } from '../services/api'; // Import API function
import { FileInfo } from '../types'; // Import type

// Basic styling for the dropzone
const dropzoneStyle: React.CSSProperties = {
  flex: 1,
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  padding: '20px',
  borderWidth: 2,
  borderRadius: 2,
  borderColor: '#eeeeee',
  borderStyle: 'dashed',
  backgroundColor: '#fafafa',
  color: '#bdbdbd',
  outline: 'none',
  transition: 'border .24s ease-in-out',
  cursor: 'pointer',
  minHeight: '100px',
  justifyContent: 'center',
};

const activeStyle: React.CSSProperties = {
  borderColor: '#2196f3',
};

const acceptStyle: React.CSSProperties = {
  borderColor: '#00e676',
};

const rejectStyle: React.CSSProperties = {
  borderColor: '#ff1744',
};


function DataSetup() {
  // Component state
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Zustand state and actions
  const predictionFileInfo = useAppStore((state) => state.predictionFileInfo);
  const setPredictionFileInfo = useAppStore((state) => state.setPredictionFileInfo);
  const selectedColumn = useAppStore((state) => state.selectedPredictionColumn);
  const setSelectedColumn = useAppStore((state) => state.setSelectedPredictionColumn);

  // react-dropzone callback
  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) {
      return;
    }
    const file = acceptedFiles[0]; // Handle only the first file for now
    setIsUploading(true);
    setUploadError(null);
    setPredictionFileInfo(null); // Clear previous info

    try {
      console.log("Uploading file:", file.name);
      const fileInfo = await uploadFile(file);
      console.log("Upload successful:", fileInfo);
      setPredictionFileInfo(fileInfo);
    } 
    
    catch (error: any) {
      console.error("Upload failed:", error);
      const backendErrorDetail = error?.detail; // FastAPI often puts messages here
        const axiosErrorMessage = error?.message; // Axios's own message (like Network Error)
        let displayError = "An unknown error occurred during upload.";

        if (typeof backendErrorDetail === 'string') {
            displayError = backendErrorDetail; // Prefer backend detail message
        } else if (axiosErrorMessage) {
            displayError = axiosErrorMessage; // Fallback to Axios message
        }
        // ================================
        setUploadError(`Upload Failed: ${displayError}`); // Use the extracted message
        setPredictionFileInfo(null);
    } 
    
    finally {
        setIsUploading(false);
    }
    }, [setPredictionFileInfo]);
    
  // react-dropzone hook
  const { getRootProps, getInputProps, isDragActive, isDragAccept, isDragReject } = useDropzone({
    onDrop,
    accept: { // Define accepted file types
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.xls'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
    },
    multiple: false, // Allow only single file upload
  });

  // Determine dynamic dropzone style
  const style = {
      ...dropzoneStyle,
      ...(isDragActive ? activeStyle : {}),
      ...(isDragAccept ? acceptStyle : {}),
      ...(isDragReject ? rejectStyle : {}),
  };

  // Handle column selection change
  const handleColumnChange = (event: SelectChangeEvent<string>) => {
     setSelectedColumn(event.target.value === "" ? null : event.target.value);
  };

  return (
    <Box sx={{ mb: 4 }}> {/* Add margin bottom */}
      <Typography variant="h5" gutterBottom>
        1. Data Upload and Column Selection
      </Typography>
      <Typography paragraph color="text.secondary">
        Upload the CSV or Excel file containing the text data you want to classify.
      </Typography>

      {/* --- Prediction Data Upload --- */}
      <Paper elevation={1} sx={{ p: 2, mb: 2 }}>
        <Typography variant="h6" gutterBottom>Data to Classify</Typography>
        <Box {...getRootProps({ style })}>
          <input {...getInputProps()} />
          {isUploading ? (
            <CircularProgress size={24} />
          ) : predictionFileInfo ? (
            <Typography>File Uploaded: {predictionFileInfo.filename} ({predictionFileInfo.num_rows} rows)</Typography>
          ) : isDragActive ? (
            <Typography>Drop the file here ...</Typography>
          ) : (
            <Typography>Drag 'n' drop file here, or click to select (CSV, XLS, XLSX)</Typography>
          )}
        </Box>
        {uploadError && <Alert severity="error" sx={{ mt: 2 }}>{uploadError}</Alert>}
      </Paper>

      {/* --- File Info & Column Selection --- */}
      {predictionFileInfo && !isUploading && (
        <Paper elevation={1} sx={{ p: 2 }}>
           <Typography variant="h6" gutterBottom>Select Text Column</Typography>
           <FormControl fullWidth sx={{mb: 2}}>
             <InputLabel id="select-text-column-label">Text Column for Classification</InputLabel>
             <Select
               labelId="select-text-column-label"
               id="select-text-column"
               value={selectedColumn ?? ''} // Use empty string if null for Select
               label="Text Column for Classification"
               onChange={handleColumnChange}
             >
                <MenuItem value=""><em>-- Select a Column --</em></MenuItem>
               {predictionFileInfo.columns.map((colName) => (
                 <MenuItem key={colName} value={colName}>
                   {colName}
                 </MenuItem>
               ))}
             </Select>
           </FormControl>
           {selectedColumn && <Typography variant="caption" display="block" gutterBottom>Selected: <strong>{selectedColumn}</strong></Typography> }

          {/* --- Data Preview Table --- */}
          <Typography variant="h6" gutterBottom sx={{ mt: 1 }}>Preview Data (First 5 Rows)</Typography>
          <TableContainer component={Paper} elevation={0} variant="outlined">
            <Table size="small" aria-label="data preview table">
              <TableHead>
                <TableRow>
                  {predictionFileInfo.columns.map((colName) => (
                    <TableCell key={colName} sx={{ fontWeight: 'bold' }}>{colName}</TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {predictionFileInfo.preview.map((row, rowIndex) => (
                  <TableRow key={rowIndex}>
                    {predictionFileInfo.columns.map((colName) => (
                       // Render cell content, handle potential non-string values safely
                      <TableCell key={`${rowIndex}-${colName}`}>
                        {String(row[colName] ?? '')}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      )}

        {/* --- Optional: Training Data Upload (Placeholder) --- */}
        {/* Add a similar Dropzone section here if implementing HF training data upload */}
        {/* <Paper elevation={1} sx={{ p: 2, mt: 2 }}>
            <Typography variant="h6" gutterBottom>Training Data (Optional for HF)</Typography>
            {/* Dropzone for training data... */}
        {/* </Paper> */}

    </Box>
  );
}

export default DataSetup;