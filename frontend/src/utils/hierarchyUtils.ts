// frontend/src/utils/hierarchyUtils.ts
import { HierarchyRow } from '../types';
import { produce } from 'immer'; // May not be needed here, but good practice if modifying state

// This function mirrors the backend logic for converting flat rows to nested structure
// Needed by ClassificationRunner to send the correct format to the backend.

interface SubSegment {
    name: string;
    keywords: string[];
}

interface Segment {
    name: string;
    subsegments: SubSegment[];
}

interface Category {
    name: string;
    segments: Segment[];
}

interface Theme {
    name: string;
    categories: Category[];
}

interface NestedHierarchy {
    themes: Theme[];
}

export const buildHierarchyFromDf = (rows: HierarchyRow[]): NestedHierarchy | null => {
    if (!rows || rows.length === 0) {
        return { themes: [] }; // Return empty structure if no rows
    }

    const themesDict: Record<string, { name: string; categories: Record<string, { name: string; segments: Record<string, { name: string; subsegments: SubSegment[] }> }> }> = {};

    let hasValidRow = false;

    rows.forEach(row => {
        const themeName = row.Theme?.trim();
        const catName = row.Category?.trim();
        const segName = row.Segment?.trim();
        const subSegName = row.Subsegment?.trim(); // Use standardized name

        // Basic validation: skip rows missing essential parts
        if (!themeName || !catName || !segName || !subSegName) {
            return;
        }
        hasValidRow = true; // Mark that we found at least one processable row

        const keywords = (row.Keywords || '')
            .split(',')
            .map(k => k.trim())
            .filter(k => k !== '');

        // Ensure theme exists
        if (!themesDict[themeName]) {
            themesDict[themeName] = { name: themeName, categories: {} };
        }
        const categoriesDict = themesDict[themeName].categories;

        // Ensure category exists
        if (!categoriesDict[catName]) {
            categoriesDict[catName] = { name: catName, segments: {} };
        }
        const segmentsDict = categoriesDict[catName].segments;

        // Ensure segment exists
        if (!segmentsDict[segName]) {
            segmentsDict[segName] = { name: segName, subsegments: [] };
        }
        const subsegmentsList = segmentsDict[segName].subsegments;

        // Add subsegment if it doesn't already exist for this path
        if (!subsegmentsList.some(ss => ss.name === subSegName)) {
            subsegmentsList.push({ name: subSegName, keywords: keywords });
        }
    });

    if (!hasValidRow) {
        console.warn("buildHierarchyFromDf: No valid rows found to build hierarchy.");
        return null; // Indicate invalid structure if no rows were complete
    }

    // Convert the dictionaries back to nested lists
    const finalThemes: Theme[] = Object.values(themesDict).map(themeData => ({
        name: themeData.name,
        categories: Object.values(themeData.categories).map(catData => ({
            name: catData.name,
            segments: Object.values(catData.segments).map(segData => ({
                name: segData.name,
                subsegments: segData.subsegments,
            })).filter(seg => seg.subsegments.length > 0), // Filter empty segments
        })).filter(cat => cat.segments.length > 0), // Filter empty categories
    })).filter(theme => theme.categories.length > 0); // Filter empty themes

    return { themes: finalThemes };
};
