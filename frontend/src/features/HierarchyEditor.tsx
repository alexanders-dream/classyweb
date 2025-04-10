// frontend/src/features/HierarchyEditor.tsx
import React, { useState, useEffect, useCallback } from 'react';
import { useAppStore } from '../store/store';
import { suggestHierarchy } from '../services/api';
import { HierarchyRow, NestedHierarchySuggestion } from '../types';
import { produce, WritableDraft } from 'immer'; // Using Immer for easier state updates, import WritableDraft

// --- Constants ---
const MAX_SAMPLE_TEXTS = 200; // Limit sample size for suggestion

// --- Helper Functions (Consider moving to a utils file later) ---

// Basic validation: Check if all required fields in a row are filled
const isRowComplete = (row: HierarchyRow): boolean => {
  return !!row.Theme && !!row.Category && !!row.Segment && !!row.Subsegment;
};

// Flatten nested suggestion (similar to backend utils.flatten_hierarchy)
const flattenNestedSuggestion = (nested: NestedHierarchySuggestion): HierarchyRow[] => {
  const rows: HierarchyRow[] = [];
  try {
    nested?.themes?.forEach((theme: any, themeIndex: number) => {
      const themeName = theme?.name || '';
      theme?.categories?.forEach((category: any, catIndex: number) => {
        const catName = category?.name || '';
        category?.segments?.forEach((segment: any, segIndex: number) => {
          const segName = segment?.name || '';
          segment?.subsegments?.forEach((subSegment: any, subSegIndex: number) => {
            const subSegName = subSegment?.name || '';
            const keywords = (subSegment?.keywords || []).join(', ');
            // Add a simple unique-ish id for React keys
            const id = `sugg-${themeIndex}-${catIndex}-${segIndex}-${subSegIndex}`;
            if (themeName && catName && segName && subSegName) {
              rows.push({ id, Theme: themeName, Category: catName, Segment: segName, Subsegment: subSegName, Keywords: keywords });
            }
          });
        });
      });
    });
  } catch (error) {
    console.error("Error flattening suggestion:", error);
    return []; // Return empty on error
  }
  return rows;
};

// Basic check if the hierarchy structure (list of rows) is valid
// A more robust check might ensure no duplicate paths exist.
const validateHierarchyRows = (rows: HierarchyRow[]): boolean => {
    if (!rows || rows.length === 0) {
        return false; // Empty is not valid for classification
    }
    // Check if at least one row has all required fields filled
    return rows.some(isRowComplete);
};

// --- Component ---

const HierarchyEditor: React.FC = () => {
  const {
    hierarchyRows,
    setHierarchyRows,
    pendingSuggestion,
    setPendingSuggestion,
    llmConfig,
    predictionFileInfo,
    selectedPredictionColumn,
    hierarchyIsValid,
    setHierarchyIsValid,
  } = useAppStore();

  const [isLoadingSuggestion, setIsLoadingSuggestion] = useState(false);
  const [suggestionError, setSuggestionError] = useState<string | null>(null);
  const [flattenedSuggestionPreview, setFlattenedSuggestionPreview] = useState<HierarchyRow[]>([]);

  // Update flattened preview when pendingSuggestion changes
  useEffect(() => {
    if (pendingSuggestion) {
      setFlattenedSuggestionPreview(flattenNestedSuggestion(pendingSuggestion));
    } else {
      setFlattenedSuggestionPreview([]);
    }
  }, [pendingSuggestion]);

  // Validate hierarchy whenever rows change
  useEffect(() => {
    const isValid = validateHierarchyRows(hierarchyRows);
    if (isValid !== hierarchyIsValid) {
      setHierarchyIsValid(isValid);
    }
  }, [hierarchyRows, hierarchyIsValid, setHierarchyIsValid]);


  const handleInputChange = (index: number, field: keyof HierarchyRow, value: string) => {
    const nextState = produce(hierarchyRows, (draft: WritableDraft<HierarchyRow>[]) => {
        // Ensure the row exists before trying to modify it
        if (draft[index]) {
            // Type assertion needed because field is a keyof HierarchyRow
            (draft[index] as any)[field] = value;
        }
    });
    setHierarchyRows(nextState);
  };

  const addRow = () => {
    const newRow: HierarchyRow = { id: Date.now(), Theme: '', Category: '', Segment: '', Subsegment: '', Keywords: '' };
    setHierarchyRows([...hierarchyRows, newRow]);
  };

  const deleteRow = (index: number) => {
    const nextState = hierarchyRows.filter((_, i) => i !== index);
    setHierarchyRows(nextState);
  };

  const handleGenerateSuggestion = useCallback(async () => {
    if (!llmConfig) {
      setSuggestionError("LLM is not configured. Please configure it in the sidebar.");
      return;
    }
    if (!predictionFileInfo || !selectedPredictionColumn) {
      setSuggestionError("Prediction data and text column must be selected first.");
      return;
    }

    setIsLoadingSuggestion(true);
    setSuggestionError(null);
    setPendingSuggestion(null); // Clear previous suggestion

    try {
        // Extract sample texts (up to MAX_SAMPLE_TEXTS)
        const sampleTexts = predictionFileInfo.preview
            .map(row => row[selectedPredictionColumn])
            .filter(text => typeof text === 'string' && text.trim() !== '') // Filter out non-strings/empty
            .slice(0, MAX_SAMPLE_TEXTS); // Limit sample size

        if (sampleTexts.length === 0) {
             setSuggestionError("No valid text found in the preview data for suggestions.");
             setIsLoadingSuggestion(false);
             return;
        }

      const response = await suggestHierarchy({ sample_texts: sampleTexts, llm_config: llmConfig });

      if (response.error) {
        setSuggestionError(`Suggestion failed: ${response.error}`);
      } else if (response.suggestion) {
        setPendingSuggestion(response.suggestion);
      } else {
        setSuggestionError("Suggestion returned no data and no error.");
      }
    } catch (err: any) {
      setSuggestionError(`Error generating suggestion: ${err.message || 'Unknown error'}`);
    } finally {
      setIsLoadingSuggestion(false);
    }
  }, [llmConfig, predictionFileInfo, selectedPredictionColumn, setPendingSuggestion]);

  const applySuggestion = () => {
    if (pendingSuggestion) {
      const flattened = flattenNestedSuggestion(pendingSuggestion);
      setHierarchyRows(flattened);
      setPendingSuggestion(null); // Clear after applying
      setSuggestionError(null);
    }
  };

  const discardSuggestion = () => {
    setPendingSuggestion(null);
    setSuggestionError(null);
    setFlattenedSuggestionPreview([]);
  };

  // --- Render Logic ---
  const canGenerateSuggestion = !!llmConfig && !!predictionFileInfo && !!selectedPredictionColumn;

  return (
    <div>
      <h2>Hierarchy Definition</h2>

      {/* AI Suggestion Section */}
      <div style={{ marginBottom: '1rem', padding: '1rem', border: '1px solid #eee' }}>
        <h4>AI Hierarchy Suggestion</h4>
        <button
          onClick={handleGenerateSuggestion}
          disabled={!canGenerateSuggestion || isLoadingSuggestion}
        >
          {isLoadingSuggestion ? 'Generating...' : '🚀 Generate Suggestion'}
        </button>
        {!canGenerateSuggestion && <p><small>Requires LLM config, prediction data, and selected text column.</small></p>}
        {suggestionError && <p style={{ color: 'red' }}>Error: {suggestionError}</p>}

        {pendingSuggestion && (
          <div style={{ marginTop: '1rem' }}>
            <p><strong>Suggestion Ready!</strong> Preview:</p>
            {/* Simple preview of flattened suggestion */}
            <div style={{ maxHeight: '150px', overflowY: 'auto', border: '1px solid #ddd', padding: '0.5rem', background: '#f9f9f9' }}>
              <pre style={{ margin: 0, fontSize: '0.8em' }}>
                {/* Escape > characters */}
                {flattenedSuggestionPreview.map(row =>
                    `T: ${row.Theme} > C: ${row.Category} > S: ${row.Segment} > Sub: ${row.Subsegment} [${row.Keywords || 'No Keywords'}]`
                ).join('\n')}
              </pre>
            </div>
            <button onClick={applySuggestion} style={{ marginRight: '0.5rem', marginTop: '0.5rem' }}>✅ Apply (Replaces Editor)</button>
            <button onClick={discardSuggestion} style={{ marginTop: '0.5rem' }}>❌ Discard</button>
          </div>
        )}
      </div>

      {/* Hierarchy Editor Table */}
      <h4>Hierarchy Editor</h4>
      <p><small>Define the Theme -&gt Category -&gt Segment -&gt Subsegment structure. Keywords should be comma-separated.</small></p>
      <button onClick={addRow} style={{ marginBottom: '0.5rem' }}>+ Add Row</button>
      <div style={{ maxHeight: '400px', overflowY: 'auto' }}> {/* Scrollable container */}
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={tableHeaderStyle}>Theme</th>
              <th style={tableHeaderStyle}>Category</th>
              <th style={tableHeaderStyle}>Segment</th>
              <th style={tableHeaderStyle}>Subsegment</th>
              <th style={tableHeaderStyle}>Keywords</th>
              <th style={tableHeaderStyle}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {hierarchyRows.map((row, index) => (
              <tr key={row.id || index}>
                <td style={tableCellStyle}><input type="text" value={row.Theme} onChange={(e) => handleInputChange(index, 'Theme', e.target.value)} style={inputStyle} /></td>
                <td style={tableCellStyle}><input type="text" value={row.Category} onChange={(e) => handleInputChange(index, 'Category', e.target.value)} style={inputStyle} /></td>
                <td style={tableCellStyle}><input type="text" value={row.Segment} onChange={(e) => handleInputChange(index, 'Segment', e.target.value)} style={inputStyle} /></td>
                <td style={tableCellStyle}><input type="text" value={row.Subsegment} onChange={(e) => handleInputChange(index, 'Subsegment', e.target.value)} style={inputStyle} /></td>
                <td style={tableCellStyle}><input type="text" value={row.Keywords} onChange={(e) => handleInputChange(index, 'Keywords', e.target.value)} style={inputStyle} placeholder="comma, separated" /></td>
                <td style={tableCellStyle}>
                  <button onClick={() => deleteRow(index)} style={{ color: 'red', border: 'none', background: 'none', cursor: 'pointer' }}>🗑️</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!hierarchyIsValid && hierarchyRows.length > 0 && (
          <p style={{ color: 'orange', marginTop: '0.5rem' }}>⚠️ Hierarchy is currently invalid. Ensure at least one row has Theme, Category, Segment, and Subsegment defined.</p>
      )}
      {hierarchyRows.length === 0 && (
          <p style={{ color: 'grey', marginTop: '0.5rem' }}>Hierarchy is empty. Add rows or generate a suggestion.</p>
      )}

      {/* TODO: Add JSON Preview of nested structure */}

    </div>
  );
};

// Basic styles (consider moving to CSS file)
const tableHeaderStyle: React.CSSProperties = {
  border: '1px solid #ddd',
  padding: '8px',
  textAlign: 'left',
  backgroundColor: '#f2f2f2',
};

const tableCellStyle: React.CSSProperties = {
  border: '1px solid #ddd',
  padding: '4px', // Reduced padding for inputs
  textAlign: 'left',
};

const inputStyle: React.CSSProperties = {
    width: '95%', // Adjust width to fit cell padding
    border: '1px solid #ccc',
    padding: '4px',
};


export default HierarchyEditor;
