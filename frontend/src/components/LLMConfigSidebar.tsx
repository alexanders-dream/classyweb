// frontend/src/components/LLMConfigSidebar.tsx
import React, { useState, useEffect, useCallback } from 'react';
import { useAppStore } from '../store/store';
import { getLLMProviders, fetchLLMModels } from '../services/api';
import { LLMProviderConfig, FetchModelsRequest } from '../types';

// Helper function to check if a provider needs an API key
const providerNeedsApiKey = (provider: string): boolean => {
  // Extend this list if other providers requiring keys are added
  return provider === 'Groq';
};

const LLMConfigSidebar: React.FC = () => {
  // Global state
  const { llmConfig, setLLMConfig } = useAppStore();

  // Local component state
  const [providers, setProviders] = useState<string[]>([]);
  const [selectedProvider, setSelectedProvider] = useState<string>(llmConfig?.provider || '');
  const [endpoint, setEndpoint] = useState<string>(llmConfig?.endpoint || '');
  const [apiKey, setApiKey] = useState<string>(llmConfig?.api_key || '');
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>(llmConfig?.model_name || '');
  const [isLoadingProviders, setIsLoadingProviders] = useState<boolean>(false);
  const [isLoadingModels, setIsLoadingModels] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [isKeyNeeded, setIsKeyNeeded] = useState<boolean>(providerNeedsApiKey(selectedProvider));

  // Fetch providers on mount
  useEffect(() => {
    const loadProviders = async () => {
      setIsLoadingProviders(true);
      setError(null);
      try {
        const response = await getLLMProviders();
        setProviders(response.providers);
        // Set default provider if none is selected and providers list is not empty
        if (!selectedProvider && response.providers.length > 0) {
          const defaultProvider = response.providers[0];
          setSelectedProvider(defaultProvider);
          setIsKeyNeeded(providerNeedsApiKey(defaultProvider));
          // Set default endpoint based on provider (assuming defaults are known or fetched)
          // For simplicity, we'll rely on user input or existing config for now.
          // A more robust approach might fetch default endpoints.
        }
      } catch (err: any) {
        setError(`Error fetching providers: ${err.message || 'Unknown error'}`);
        setProviders([]);
      } finally {
        setIsLoadingProviders(false);
      }
    };
    loadProviders();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Run only on mount

  // Fetch models when provider, endpoint, or API key validity changes
  const loadModels = useCallback(async () => {
    if (!selectedProvider || !endpoint || (isKeyNeeded && !apiKey)) {
      setModels([]);
      setSelectedModel('');
      return; // Don't fetch if required info is missing
    }

    setIsLoadingModels(true);
    setError(null);
    setModels([]); // Clear previous models
    setSelectedModel(''); // Clear previous selection

    const requestData: FetchModelsRequest = {
      provider: selectedProvider,
      endpoint: endpoint,
      api_key: isKeyNeeded ? apiKey : null,
    };

    try {
      const response = await fetchLLMModels(requestData);
      setModels(response.models || []);
      // Select first model as default if available
      if (response.models && response.models.length > 0) {
        setSelectedModel(response.models[0]);
      } else {
         setError(`No models found for ${selectedProvider} at ${endpoint}.`);
      }
    } catch (err: any) {
      setError(`Error fetching models: ${err.message || 'Unknown error'}`);
      setModels([]);
    } finally {
      setIsLoadingModels(false);
    }
  }, [selectedProvider, endpoint, apiKey, isKeyNeeded]);

  // Effect to trigger model loading when relevant inputs change
  // We use a separate effect to avoid calling loadModels directly in onChange handlers
  useEffect(() => {
    // Only load models if provider and endpoint are set
    // And API key is set if needed
    if (selectedProvider && endpoint && (!isKeyNeeded || apiKey)) {
      loadModels();
    } else {
      // Clear models if config becomes incomplete
      setModels([]);
      setSelectedModel('');
    }
  }, [selectedProvider, endpoint, apiKey, isKeyNeeded, loadModels]);


  // Update global state when a valid configuration is selected
  useEffect(() => {
    if (selectedProvider && endpoint && selectedModel && (!isKeyNeeded || apiKey)) {
      const newConfig: LLMProviderConfig = {
        provider: selectedProvider,
        endpoint: endpoint,
        model_name: selectedModel,
        api_key: isKeyNeeded ? apiKey : null,
      };
      // Only update if the config has actually changed
      if (JSON.stringify(newConfig) !== JSON.stringify(llmConfig)) {
        setLLMConfig(newConfig);
      }
    } else {
      // If config is incomplete, clear global state
      if (llmConfig !== null) {
        setLLMConfig(null);
      }
    }
  }, [selectedProvider, endpoint, apiKey, selectedModel, isKeyNeeded, setLLMConfig, llmConfig]);

  // Handlers
  const handleProviderChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const newProvider = event.target.value;
    setSelectedProvider(newProvider);
    setIsKeyNeeded(providerNeedsApiKey(newProvider));
    // Reset dependent fields
    setEndpoint(''); // Or set a default endpoint for the provider
    setApiKey('');
    setModels([]);
    setSelectedModel('');
    setError(null);
  };

  const handleEndpointChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setEndpoint(event.target.value);
    // Models will refetch via useEffect
  };

  const handleApiKeyChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setApiKey(event.target.value);
    // Models will refetch via useEffect
  };

   const handleModelChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedModel(event.target.value);
    // Global state updates via useEffect
  };

  // Determine connection status (basic check)
  const isReady = !!llmConfig;

  return (
    <div style={{ padding: '1rem', borderRight: '1px solid #ccc', height: '100%' }}>
      <h3>LLM Configuration</h3>
      {isLoadingProviders && <p>Loading providers...</p>}
      {error && <p style={{ color: 'red' }}>Error: {error}</p>}

      <div>
        <label htmlFor="llm-provider">Provider:</label>
        <select
          id="llm-provider"
          value={selectedProvider}
          onChange={handleProviderChange}
          disabled={isLoadingProviders}
          style={{ width: '100%', marginBottom: '0.5rem' }}
        >
          <option value="" disabled>-- Select Provider --</option>
          {providers.map((p) => (
            <option key={p} value={p}>{p}</option>
          ))}
        </select>
      </div>

      <div>
        <label htmlFor="llm-endpoint">API Endpoint:</label>
        <input
          type="text"
          id="llm-endpoint"
          value={endpoint}
          onChange={handleEndpointChange}
          placeholder="e.g., http://localhost:11434 or https://api.groq.com/openai/v1"
          disabled={!selectedProvider}
          style={{ width: '100%', marginBottom: '0.5rem' }}
        />
      </div>

      {isKeyNeeded && (
        <div>
          <label htmlFor="llm-api-key">API Key:</label>
          <input
            type="password"
            id="llm-api-key"
            value={apiKey}
            onChange={handleApiKeyChange}
            placeholder={`Enter ${selectedProvider} API Key`}
            disabled={!selectedProvider}
            style={{ width: '100%', marginBottom: '0.5rem' }}
          />
        </div>
      )}

      <div>
        <label htmlFor="llm-model">Model:</label>
        <select
          id="llm-model"
          value={selectedModel}
          onChange={handleModelChange}
          disabled={isLoadingModels || models.length === 0 || !selectedProvider || !endpoint || (isKeyNeeded && !apiKey)}
          style={{ width: '100%', marginBottom: '0.5rem' }}
        >
          <option value="" disabled>-- Select Model --</option>
          {isLoadingModels && <option value="" disabled>Loading models...</option>}
          {models.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
         {/* Optional: Refresh button */}
         <button onClick={loadModels} disabled={isLoadingModels || !selectedProvider || !endpoint || (isKeyNeeded && !apiKey)} style={{marginLeft: '0.5rem'}}>
            🔄
         </button>
      </div>

      <hr style={{ margin: '1rem 0' }} />

      <div>
        <strong>Status:</strong> {isReady ? <span style={{color: 'green'}}>Ready ✅</span> : <span style={{color: 'orange'}}>Not Ready ⚠️</span>}
        {isReady && llmConfig && (
            <p style={{fontSize: '0.8em', color: '#555'}}>
                Using: {llmConfig.provider} - {llmConfig.model_name}
            </p>
        )}
      </div>

      {/* TODO: Add End Session button? Or handle globally */}

    </div>
  );
};

export default LLMConfigSidebar;
