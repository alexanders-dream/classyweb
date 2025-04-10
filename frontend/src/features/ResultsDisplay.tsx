// frontend/src/features/ResultsDisplay.tsx
import React from 'react';
import { useAppStore } from '../store/store';
import { ClassificationResultRow } from '../types';
import Papa from 'papaparse'; // Import papaparse

// Helper function to trigger CSV download
const downloadCSV = (data: ClassificationResultRow[], filename: string = 'classification_results.csv') => {
    if (!data || data.length === 0) {
        console.error("No data available to download.");
        alert("No results data to download.");
        return;
    }
    try {
        const csv = Papa.unparse(data);
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', filename);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url); // Clean up blob URL
    } catch (error) {
        console.error("Error generating or downloading CSV:", error);
        alert("Failed to download CSV.");
    }
};


const ResultsDisplay: React.FC = () => {
    const { classificationResults } = useAppStore();

    if (!classificationResults) {
        return <div>No classification results to display yet. Run a classification first.</div>;
    }

    if (classificationResults.length === 0) {
        return <div>Classification ran, but produced no result rows.</div>;
    }

    // Dynamically get headers from the first result row
    const headers = Object.keys(classificationResults[0]);

    return (
        <div>
            <h2>Classification Results</h2>
            <button
                onClick={() => downloadCSV(classificationResults)}
                style={{ marginBottom: '1rem' }}
            >
                📥 Download as CSV
            </button>

            <div style={{ maxHeight: '500px', overflow: 'auto', border: '1px solid #ccc' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                        <tr>
                            {headers.map(header => (
                                <th key={header} style={tableHeaderStyle}>{header}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {classificationResults.map((row, rowIndex) => (
                            <tr key={`row-${rowIndex}`}>
                                {headers.map(header => (
                                    <td key={`${header}-${rowIndex}`} style={tableCellStyle}>
                                        {/* Display value, handle null/undefined */}
                                        {row[header] !== null && row[header] !== undefined ? String(row[header]) : ''}
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

// Basic styles (reuse or move to CSS)
const tableHeaderStyle: React.CSSProperties = {
  border: '1px solid #ddd',
  padding: '8px',
  textAlign: 'left',
  backgroundColor: '#f2f2f2',
  position: 'sticky', // Make header sticky
  top: 0, // Stick to the top of the scrollable container
  zIndex: 1 // Ensure header is above table content
};

const tableCellStyle: React.CSSProperties = {
  border: '1px solid #ddd',
  padding: '8px',
  textAlign: 'left',
  whiteSpace: 'nowrap', // Prevent text wrapping initially
  overflow: 'hidden',
  textOverflow: 'ellipsis', // Add ellipsis for overflow
  maxWidth: '200px' // Limit column width
};

export default ResultsDisplay;
