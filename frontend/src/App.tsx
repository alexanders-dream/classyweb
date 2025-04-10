// frontend/src/App.tsx
import React from 'react';
import { ThemeProvider, createTheme, CssBaseline, Box, Drawer, List, ListItem, ListItemButton, ListItemText, Toolbar, Typography, Divider, AppBar } from '@mui/material';
import DataSetup from './features/DataSetup';
import { useAppStore } from './store/store'; // Added store import
// Import the new components
import LLMConfigSidebar from './components/LLMConfigSidebar';
import HierarchyEditor from './features/HierarchyEditor';
import ClassificationRunner from './features/ClassificationRunner';
import ResultsDisplay from './features/ResultsDisplay'; // Added ResultsDisplay


const drawerWidth = 280; // Slightly wider maybe

function App() {
  const theme = createTheme({
    palette: {
      mode: 'light',
    },
  });

  // Placeholder for active workflow/tab state
  const [activeWorkflow, setActiveWorkflow] = React.useState("LLM"); // Example state
  // Get results from store to conditionally render ResultsDisplay
  const classificationResults = useAppStore((state) => state.classificationResults);

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
            <List dense> {/* Use dense for sidebar */}
              <ListItem disablePadding>
                <ListItemButton selected={activeWorkflow === "LLM"} onClick={() => setActiveWorkflow("LLM")}>
                  <ListItemText primary="LLM Categorization" />
                </ListItemButton>
              </ListItem>
              <ListItem disablePadding>
                <ListItemButton selected={activeWorkflow === "HF"} onClick={() => setActiveWorkflow("HF")}>
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
                <Typography variant="h6" gutterBottom>🤗 HF Configuration</Typography>
                {/* Placeholder for HF Sidebar - Phase 3 */}
                 <Typography variant="body2" color="text.secondary">HF Config options will appear here...</Typography>
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

          {/* Render content based on workflow or future tabs */}
          {/* For now, just show DataSetup */}
          <DataSetup />

          {/* Render Hierarchy Editor only for LLM workflow */}
          {activeWorkflow === "LLM" && (
            <>
              <Divider sx={{ my: 4 }} />
              <HierarchyEditor />
            </>
          )}

          {/* Placeholder for other tabs/sections */}
          {/* <Divider sx={{ my: 4 }} />
          <Typography variant="h5">2. Hierarchy (Placeholder)</Typography> */}

          {/* Render Classification Runner only for LLM workflow */}
          {activeWorkflow === "LLM" && (
            <>
              <Divider sx={{ my: 4 }} />
              <ClassificationRunner />
            </>
          )}

           <Divider sx={{ my: 4 }} />
           {/* Conditionally render Results Display */}
           {classificationResults && (
                <ResultsDisplay />
           )}
           {!classificationResults && (
                <Typography variant="h5">4. Results (Run classification to see results)</Typography>
           )}


        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
