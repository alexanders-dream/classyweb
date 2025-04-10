// frontend/src/App.tsx
import React, { useState, useEffect } from 'react'; // Import useState and useEffect
import { ThemeProvider, createTheme, CssBaseline, Box, Drawer, List, ListItem, ListItemButton, ListItemText, Toolbar, Typography, Divider, AppBar, Tabs, Tab, Select, MenuItem, FormControl, InputLabel, CircularProgress } from '@mui/material'; // Added Tabs, Tab, Select, MenuItem, FormControl, InputLabel, CircularProgress
import DataSetup from './features/DataSetup';
import { useAppStore } from './store/store';
// Import the components
import LLMConfigSidebar from './components/LLMConfigSidebar';
import HFModelSelector from './components/HFModelSelector'; // Import the new component
import HierarchyEditor from './features/HierarchyEditor';
import ClassificationRunner from './features/ClassificationRunner'; // Will likely need modification later
import ResultsDisplay from './features/ResultsDisplay';
// Import HF components matching their exports
import { HFTrainingForm } from './features/HFTrainingForm'; // Use named import as confirmed
import HFRulesEditor from './features/HFRulesEditor'; // Assuming this has a default export
import HFClassificationRunner from './features/HFClassificationRunner'; // Use default import as confirmed


const drawerWidth = 280;

// Helper component for Tab Panels
interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`workflow-tabpanel-${index}`}
      aria-labelledby={`workflow-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ pt: 3 }}> {/* Add padding top to content */}
          {children}
        </Box>
      )}
    </div>
  );
}

function a11yProps(index: number) {
  return {
    id: `workflow-tab-${index}`,
    'aria-controls': `workflow-tabpanel-${index}`,
  };
}


function App() {
  const theme = createTheme({
    palette: {
      mode: 'light', // Keep light mode for now
    },
  });

  const [activeWorkflow, setActiveWorkflow] = useState<'LLM' | 'HF'>("LLM"); // Workflow state
  const [activeTab, setActiveTab] = useState(0); // Tab state (index)
  const classificationResults = useAppStore((state) => state.classificationResults);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleWorkflowChange = (workflow: 'LLM' | 'HF') => {
    setActiveWorkflow(workflow);
    setActiveTab(0); // Reset tab when workflow changes
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: 'flex' }}>
         {/* Optional: Add an AppBar for Title */}
         <AppBar
            position="fixed"
            sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}
         >
            <Toolbar>
            <Typography variant="h6" noWrap component="div">
                🏷️ AI Text Classifier
            </Typography>
            </Toolbar>
        </AppBar>

        <Drawer
          variant="permanent"
          sx={{
            width: drawerWidth,
            flexShrink: 0,
            [`& .MuiDrawer-paper`]: { width: drawerWidth, boxSizing: 'border-box' },
          }}
        >
          <Toolbar /> {/* Spacer for AppBar */}
          <Box sx={{ overflow: 'auto', padding: 2 }}>
            <Typography variant="h6" gutterBottom>🛠️ Workflow</Typography>
            <List dense>
              <ListItem disablePadding>
                <ListItemButton selected={activeWorkflow === "LLM"} onClick={() => handleWorkflowChange("LLM")}>
                  <ListItemText primary="LLM Categorization" />
                </ListItemButton>
              </ListItem>
              <ListItem disablePadding>
                <ListItemButton selected={activeWorkflow === "HF"} onClick={() => handleWorkflowChange("HF")}>
                  <ListItemText primary="Hugging Face Model" />
                </ListItemButton>
              </ListItem>
            </List>

            <Divider sx={{ my: 2 }} />

            {/* Conditional Sidebar Content */}
            {activeWorkflow === "LLM" && (
              // Render the LLM Config Sidebar when LLM workflow is active
              <LLMConfigSidebar />
            )}
             {activeWorkflow === "HF" && (
              <>
                 <Typography variant="h6" gutterBottom>🤗 HF Model</Typography>
                 {/* HF Model Selection - Phase 3 */}
                 <HFModelSelector />
               </>
             )}

            {/* Add End Session button later */}

          </Box>
        </Drawer>

        {/* Main Content Area */}
        <Box
          component="main"
          sx={{ flexGrow: 1, bgcolor: 'background.default', p: 3 }}
        >
          <Toolbar /> {/* Spacer for AppBar */}

          {/* Tabs for Workflow Steps */}
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            {activeWorkflow === 'LLM' && (
              <Tabs value={activeTab} onChange={handleTabChange} aria-label="LLM workflow tabs">
                <Tab label="1. Data Setup" {...a11yProps(0)} />
                <Tab label="2. Hierarchy" {...a11yProps(1)} />
                <Tab label="3. Run & Results" {...a11yProps(2)} />
              </Tabs>
            )}
            {activeWorkflow === 'HF' && (
              <Tabs value={activeTab} onChange={handleTabChange} aria-label="HF workflow tabs">
                <Tab label="1. Data Setup" {...a11yProps(0)} />
                <Tab label="2. Train Model" {...a11yProps(1)} />
                <Tab label="3. Manage Rules" {...a11yProps(2)} />
                <Tab label="4. Run & Results" {...a11yProps(3)} />
              </Tabs>
            )}
          </Box>

          {/* Tab Content Panels */}
          {activeWorkflow === 'LLM' && (
            <>
              <TabPanel value={activeTab} index={0}>
                <DataSetup /> {/* Use for prediction data */}
              </TabPanel>
              <TabPanel value={activeTab} index={1}>
                <HierarchyEditor />
              </TabPanel>
              <TabPanel value={activeTab} index={2}>
                <ClassificationRunner /> {/* Handles LLM run trigger */}
                {/* Results Display will show below if results exist */}
                {classificationResults && <ResultsDisplay />}
                {!classificationResults && <Typography sx={{mt: 2}}>Run classification to see results.</Typography>}
              </TabPanel>
            </>
          )}

          {activeWorkflow === 'HF' && (
            <>
              <TabPanel value={activeTab} index={0}>
                 {/* Reuse DataSetup, maybe add prop later to distinguish training/prediction */}
                <DataSetup />
                <Typography variant="caption" display="block" gutterBottom sx={{mt: 1}}>
                    Upload training data here. Ensure it includes text and label columns.
                </Typography>
              </TabPanel>
              <TabPanel value={activeTab} index={1}>
                <HFTrainingForm /> {/* Placeholder */}
              </TabPanel>
              <TabPanel value={activeTab} index={2}>
                <HFRulesEditor /> {/* Use the imported component */}
              </TabPanel>
              <TabPanel value={activeTab} index={3}>
                <HFClassificationRunner /> {/* Use the imported component */}
                 {/* Results Display will show below if results exist */}
                {classificationResults && <ResultsDisplay />}
                {!classificationResults && <Typography sx={{mt: 2}}>Run classification to see results.</Typography>}
              </TabPanel>
            </>
          )}


        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
