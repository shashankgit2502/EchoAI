/**
 * Backend Integration Module for Workflow Builder IDE & Agent Builder
 *
 * This module provides all API integration functions for:
 * - Workflow generation (Chat tab)
 * - Canvas management (Nodes tab)
 * - Agent templates (Agents tab)
 * - HITL operations (Approve/Reject/Modify/Defer)
 * - Workflow execution and testing
 * - Agent builder operations
 */

// ============================================================================
// CONFIGURATION
// ============================================================================

const API_CONFIG = {
    baseURL: 'http://localhost:8000',
    timeout: 30000,
    headers: {
        'Content-Type': 'application/json'
    }
};

// ============================================================================
// HTTP CLIENT
// ============================================================================

class APIClient {
    constructor(config) {
        this.baseURL = config.baseURL;
        this.timeout = config.timeout;
        this.headers = config.headers;
    }

    async request(method, endpoint, data = null, customHeaders = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const headers = { ...this.headers, ...customHeaders };

        const options = {
            method,
            headers,
            signal: AbortSignal.timeout(this.timeout)
        };

        if (data && (method === 'POST' || method === 'PUT' || method === 'PATCH')) {
            options.body = JSON.stringify(data);
        }

        try {
            const response = await fetch(url, options);

            if (!response.ok) {
                const error = await response.json().catch(() => ({ detail: response.statusText }));
                throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            if (error.name === 'AbortError' || error.name === 'TimeoutError') {
                throw new Error('Request timeout - server took too long to respond');
            }
            throw error;
        }
    }

    get(endpoint) {
        return this.request('GET', endpoint);
    }

    post(endpoint, data) {
        return this.request('POST', endpoint, data);
    }

    put(endpoint, data) {
        return this.request('PUT', endpoint, data);
    }

    delete(endpoint) {
        return this.request('DELETE', endpoint);
    }
}

const apiClient = new APIClient(API_CONFIG);

// ============================================================================
// WORKFLOW APIs
// ============================================================================

const WorkflowAPI = {
    /**
     * Generate workflow from natural language prompt (Chat tab)
     */
    async generateFromPrompt(prompt, defaultLLM = null) {
        return await apiClient.post('/workflows/design/prompt', {
            prompt,
            default_llm: defaultLLM
        });
    },

    /**
     * Convert canvas nodes to backend workflow format (Nodes tab)
     */
    async canvasToBackend(canvasNodes, connections, workflowName = null) {
        return await apiClient.post('/workflows/canvas/to-backend', {
            canvas_nodes: canvasNodes,
            connections,
            workflow_name: workflowName
        });
    },

    /**
     * Convert backend workflow to canvas format (Load workflow)
     */
    async backendToCanvas(workflow, agents) {
        return await apiClient.post('/workflows/backend/to-canvas', {
            workflow,
            agents
        });
    },

    /**
     * Save canvas workflow (converts, validates, and saves)
     */
    async saveCanvas(canvasNodes, connections, workflowName, saveAs = 'draft') {
        return await apiClient.post('/workflows/canvas/save', {
            canvas_nodes: canvasNodes,
            connections,
            workflow_name: workflowName,
            save_as: saveAs
        });
    },

    /**
     * Validate workflow before saving
     */
    async validateDraft(workflow, agents) {
        return await apiClient.post('/workflows/validate/draft', {
            workflow,
            agents
        });
    },

    /**
     * Validate workflow after HITL (full validation)
     */
    async validateFinal(workflow, agents) {
        return await apiClient.post('/workflows/validate/final', {
            workflow,
            agents
        });
    },

    /**
     * Save workflow as temp for testing
     */
    async saveTemp(workflow) {
        return await apiClient.post('/workflows/temp/save', workflow);
    },

    /**
     * Load temp workflow
     */
    async loadTemp(workflowId) {
        return await apiClient.get(`/workflows/${workflowId}/temp`);
    },

    /**
     * Execute workflow (test or final mode)
     */
    async execute(workflowId, mode = 'test', version = null, inputPayload = {}) {
        return await apiClient.post('/workflows/execute', {
            workflow_id: workflowId,
            mode,
            version,
            input_payload: inputPayload
        });
    },

    /**
     * Save tested workflow as final
     */
    async saveFinal(workflowId, version = '1.0', notes = '') {
        return await apiClient.post('/workflows/save-final', {
            workflow_id: workflowId,
            version,
            notes
        });
    },

    /**
     * List workflow versions
     */
    async listVersions(workflowId) {
        return await apiClient.get(`/workflows/${workflowId}/versions`);
    },

    /**
     * Load specific final version
     */
    async loadFinal(workflowId, version) {
        return await apiClient.get(`/workflows/${workflowId}/final/${version}`);
    },

    /**
     * Clone final workflow to draft
     */
    async cloneFinal(workflowId, fromVersion) {
        return await apiClient.post('/workflows/clone', {
            workflow_id: workflowId,
            from_version: fromVersion
        });
    }
};

// ============================================================================
// CHAT SESSION APIs (Runtime Testing)
// ============================================================================

const ChatAPI = {
    /**
     * Start new chat session for workflow testing
     */
    async startSession(workflowId, mode = 'test', version = null, initialContext = {}) {
        return await apiClient.post('/workflows/chat/start', {
            workflow_id: workflowId,
            mode,
            version,
            initial_context: initialContext
        });
    },

    /**
     * Send message and execute workflow
     */
    async sendMessage(sessionId, message, executeWorkflow = true) {
        return await apiClient.post('/workflows/chat/send', {
            session_id: sessionId,
            message,
            execute_workflow: executeWorkflow
        });
    },

    /**
     * Get chat history
     */
    async getHistory(sessionId) {
        return await apiClient.get(`/workflows/chat/history/${sessionId}`);
    },

    /**
     * List all chat sessions
     */
    async listSessions(workflowId = null, limit = 50) {
        const params = new URLSearchParams();
        if (workflowId) params.append('workflow_id', workflowId);
        if (limit) params.append('limit', limit);

        return await apiClient.get(`/workflows/chat/sessions?${params}`);
    },

    /**
     * Delete chat session
     */
    async deleteSession(sessionId) {
        return await apiClient.delete(`/workflows/chat/${sessionId}`);
    }
};

// ============================================================================
// HITL (Human-in-the-Loop) APIs
// ============================================================================

const HITLAPI = {
    /**
     * Get HITL status for a run
     */
    async getStatus(runId) {
        return await apiClient.get(`/workflows/hitl/status/${runId}`);
    },

    /**
     * Get full HITL context
     */
    async getContext(runId) {
        return await apiClient.get(`/workflows/hitl/context/${runId}`);
    },

    /**
     * Approve workflow execution at HITL checkpoint
     */
    async approve(runId, actor, rationale = null) {
        return await apiClient.post('/workflows/hitl/approve', {
            run_id: runId,
            actor,
            rationale
        });
    },

    /**
     * Reject workflow execution
     */
    async reject(runId, actor, rationale = null) {
        return await apiClient.post('/workflows/hitl/reject', {
            run_id: runId,
            actor,
            rationale
        });
    },

    /**
     * Modify workflow/agent at HITL checkpoint
     */
    async modify(runId, actor, changes, rationale = null) {
        return await apiClient.post('/workflows/hitl/modify', {
            run_id: runId,
            actor,
            changes,
            rationale
        });
    },

    /**
     * Defer HITL decision
     */
    async defer(runId, actor, deferUntil = null, rationale = null) {
        return await apiClient.post('/workflows/hitl/defer', {
            run_id: runId,
            actor,
            defer_until: deferUntil,
            rationale
        });
    },

    /**
     * List all pending HITL reviews
     */
    async listPending() {
        return await apiClient.get('/workflows/hitl/pending');
    },

    /**
     * Get decision audit trail
     */
    async getDecisions(runId) {
        return await apiClient.get(`/workflows/hitl/decisions/${runId}`);
    }
};

// ============================================================================
// AGENT APIs
// ============================================================================

const AgentAPI = {
    /**
     * Get all agent templates (static + created)
     */
    async getAllTemplates() {
        return await apiClient.get('/agents/templates/all');
    },

    /**
     * Get only static templates
     */
    async getStaticTemplates() {
        return await apiClient.get('/agents/templates/static');
    },

    /**
     * Design agent from prompt
     */
    async designFromPrompt(prompt, defaultLLM = null) {
        return await apiClient.post('/agents/design/prompt', {
            prompt,
            default_llm: defaultLLM
        });
    },

    /**
     * Register new agent
     */
    async register(agent) {
        return await apiClient.post('/agents/register', agent);
    },

    /**
     * Get agent by ID
     */
    async get(agentId) {
        return await apiClient.get(`/agents/${agentId}`);
    },

    /**
     * Update agent
     */
    async update(agentId, updates) {
        return await apiClient.put(`/agents/${agentId}`, updates);
    },

    /**
     * Delete agent
     */
    async delete(agentId) {
        return await apiClient.delete(`/agents/${agentId}`);
    },

    /**
     * List all agents
     */
    async list() {
        return await apiClient.get('/agents');
    }
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Handle API errors gracefully
 */
function handleAPIError(error, context = '') {
    console.error(`API Error ${context}:`, error);

    let userMessage = 'An error occurred';

    if (error.message.includes('timeout')) {
        userMessage = 'Request timeout - please try again';
    } else if (error.message.includes('Network')) {
        userMessage = 'Network error - check your connection';
    } else if (error.message) {
        userMessage = error.message;
    }

    return {
        success: false,
        error: userMessage,
        details: error
    };
}

/**
 * Show notification (can be replaced with your UI notification system)
 */
function showNotification(message, type = 'info') {
    console.log(`[${type.toUpperCase()}]`, message);
    // TODO: Integrate with your UI notification system
    // Example: toast.show(message, { type });
}

/**
 * Format workflow name from prompt
 */
function formatWorkflowName(prompt) {
    return prompt
        .trim()
        .slice(0, 50)
        .replace(/[^a-zA-Z0-9\s-]/g, '')
        .replace(/\s+/g, '_')
        .toLowerCase() || 'untitled_workflow';
}

/**
 * Convert frontend node to backend agent format
 * (This is handled by backend node_mapper, but useful for preview)
 */
function previewNodeAsAgent(node) {
    return {
        id: node.id,
        name: node.name,
        type: node.type,
        config: node.config,
        position: { x: node.x, y: node.y }
    };
}

// ============================================================================
// EXPORT
// ============================================================================

// Export for use in HTML files
if (typeof window !== 'undefined') {
    window.BackendAPI = {
        Workflow: WorkflowAPI,
        Chat: ChatAPI,
        HITL: HITLAPI,
        Agent: AgentAPI,
        handleError: handleAPIError,
        showNotification,
        formatWorkflowName,
        previewNodeAsAgent
    };
}

// Export for use in module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        WorkflowAPI,
        ChatAPI,
        HITLAPI,
        AgentAPI,
        handleAPIError,
        showNotification,
        formatWorkflowName,
        previewNodeAsAgent
    };
}
