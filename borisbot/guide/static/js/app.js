/**
 * BorisBot Guide - App Logic (v2)
 */

const API = {
    async get(endpoint) {
        const res = await fetch(endpoint);
        if (!res.ok) throw new Error(`API Error: ${res.status}`);
        return res.json();
    },
    async post(endpoint, body) {
        const res = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        if (!res.ok) throw new Error(`API Error: ${res.status}`);
        return res.json();
    }
};

const Wizard = {
    currentStep: 1,
    totalSteps: 4,

    async init() {
        try {
            const state = await API.get('/api/wizard-state');
            // customized logic: if step-4 is done, or if profile exists?
            // simpler: check if locally we think we are done.
            // but for now, let's just check if the last step is marked complete.
            // The server state might not have 'completed' flag for the whole wizard, 
            // but let's check if the user has visited step 4.

            // Actually, let's just load the dashboard if the URL has ?dashboard=1 
            // OR if we can detect completion. 
            // For now, to unblock the user who just finished, let's modify finish() to set a flag in localStorage 
            // and check it here.

            if (localStorage.getItem('borisbot_setup_complete') === 'true') {
                this.finish();
                return;
            }
        } catch (e) { console.error(e); }

        this.renderStep(1);
        this.startSystemChecks();
        this.loadRecommendedModels();
        this.checkProviderStatus();
    },

    next() {
        if (this.currentStep < this.totalSteps) {
            this.currentStep++;
            this.renderStep(this.currentStep);
        } else {
            this.finish();
        }
    },

    prev() {
        if (this.currentStep > 1) {
            this.currentStep--;
            this.renderStep(this.currentStep);
        }
    },

    finish() {
        localStorage.setItem('borisbot_setup_complete', 'true');
        document.getElementById('wizard-container').classList.add('hidden');
        document.getElementById('dashboard-container').classList.remove('hidden');
        // Initial dashboard load
        Dashboard.init();
    },

    async applyPolicyAndFinish() {
        try {
            const selected = document.querySelector('input[name="policy"]:checked');
            const policyName = selected ? selected.value : 'safe-local';
            await API.post('/api/run', {
                action: 'policy_apply',
                params: { policy_name: policyName, agent_id: 'default' },
                approve_permission: true
            });
        } catch (e) {
            console.error('Policy apply failed', e);
        }
        this.finish();
    },

    renderStep(step) {
        // Toggle step
        document.querySelectorAll('.wizard-step').forEach(el => el.classList.add('hidden'));
        document.getElementById(`step-${step}`).classList.remove('hidden');

        // Update dots
        document.querySelectorAll('.step-dot').forEach(el => {
            const dotStep = parseInt(el.dataset.step);
            el.className = 'step-dot';
            if (dotStep === step) el.classList.add('active');
            if (dotStep < step) el.classList.add('completed');
        });

        // Dynamic checks
        if (step === 2) this.checkProviderStatus();
    },

    // Step 1: System Checks
    async startSystemChecks() {
        const container = document.getElementById('system-checks');
        container.innerHTML = `<div class="text-sm text-gray-500">Running diagnostics...</div>`;

        try {
            const checks = await API.get('/api/system-check');
            container.innerHTML = '';

            let allPassed = true;
            checks.items.forEach(check => {
                const el = document.createElement('div');
                el.className = 'check-item';
                el.innerHTML = `
                    <div class="check-icon ${check.passed ? 'ok' : 'err'}">
                        ${check.passed ? '✓' : '✕'}
                    </div>
                    <div>
                        <div class="font-medium text-sm">${check.label}</div>
                        <div class="text-xs text-gray-500">${check.message}</div>
                    </div>
                `;
                container.appendChild(el);
                if (!check.passed && check.label !== 'Docker Engine') allPassed = false; // Docker opt
            });

            if (allPassed) {
                document.getElementById('btn-step-1-next').disabled = false;
                document.getElementById('btn-step-1-next').onclick = () => this.next();
            }

        } catch (e) {
            container.innerHTML = `<div class="text-red-500 text-sm">Failed to run checks: ${e.message}</div>`;
        }
    },

    // Step 2: Models & Providers
    async loadRecommendedModels() {
        try {
            const res = await API.get('/api/recommended-models');
            const grid = document.getElementById('model-grid');
            grid.innerHTML = '';

            // Also check what's installed
            const status = await API.get('/api/provider-status/ollama');
            const currentModel = status.model || '';

            res.items.forEach(model => {
                const isInstalled = currentModel.includes(model.id.split(':')[0]);
                const card = document.createElement('div');
                card.className = `model-card ${model.recommended ? 'recommended' : ''}`;
                card.innerHTML = `
                    ${model.recommended ? '<div class="model-badge">Recommended</div>' : ''}
                    <div class="model-name">${model.name}</div>
                    <div class="model-meta">
                        <span>${model.size}</span> • <span>${model.id}</span>
                    </div>
                    <div class="model-desc">${model.description}</div>
                    <div class="model-actions">
                         ${isInstalled
                        ? `<button class="btn-secondary btn-small" disabled>Installed</button>`
                        : `<button class="btn btn-small" onclick="Wizard.pullModel('${model.id}', this)">Get</button>`
                    }
                    </div>
                    <div class="progress-container hidden">
                        <div class="progress-bar" style="width:0%"></div>
                    </div>
                `;
                grid.appendChild(card);
            });
        } catch (e) {
            console.error(e);
        }
    },

    async pullModel(modelId, btn) {
        btn.disabled = true;
        btn.textContent = "Pulling...";
        const card = btn.closest('.model-card');
        const progressContainer = card.querySelector('.progress-container');
        const progressBar = card.querySelector('.progress-bar');
        progressContainer.classList.remove('hidden');

        try {
            const res = await API.post('/api/ollama/pull', { model_name: modelId });
            const jobId = res.job_id;

            // Poll
            const poll = setInterval(async () => {
                const job = await API.get(`/api/jobs/${jobId}`);
                if (job.status === 'completed') {
                    clearInterval(poll);
                    progressBar.style.width = '100%';
                    btn.textContent = "Installed";
                    btn.className = "btn-secondary btn-small";
                    setTimeout(() => progressContainer.classList.add('hidden'), 1000);
                    // Enable next
                    document.getElementById('btn-step-2-next').disabled = false;
                } else if (job.status === 'failed') {
                    clearInterval(poll);
                    btn.textContent = "Retry";
                    btn.disabled = false;
                    alert('Pull failed: ' + job.error);
                } else {
                    // Fake progress slightly based on time if no real progress in job output
                    // For now, simple indeterminate animation handles visuals
                    progressBar.style.width = '50%';
                }
            }, 1000);

        } catch (e) {
            btn.disabled = false;
            btn.textContent = "Get";
            alert('Failed to start pull: ' + e.message);
        }
    },

    async checkProviderStatus() {
        // ... (Ollama check logic) ...
        try {
            const status = await API.get('/api/provider-status/ollama');
            const el = document.getElementById('ollama-connection-status');
            if (status.running) {
                el.innerHTML = `<span style="color:#137333">● Ollama is running</span>`;
                if (status.model) {
                    document.getElementById('btn-step-2-next').disabled = false;
                    document.getElementById('btn-step-2-next').onclick = () => this.next();
                }
            } else {
                el.innerHTML = `<span style="color:#c5221f">● Ollama disconnected</span>`;
            }
        } catch (e) { }
    },

    async saveVerifyAgent() {
        const log = document.getElementById('verification-log');
        log.innerHTML = 'Starting verification...\n';
        try {
            const res = await API.post('/api/run', {
                action: 'verify_agent_logic',
                params: {},
                approve_permission: true
            });
            const poll = setInterval(async () => {
                const job = await API.get(`/api/jobs/${res.job_id}`);
                log.textContent = job.output || 'Running...';
                log.scrollTop = log.scrollHeight;
                if (job.status === 'completed') {
                    clearInterval(poll);
                    document.getElementById('btn-step-3-next').classList.remove('hidden');
                }
            }, 1000);
        } catch (e) {
            log.textContent += `Error: ${e.message}`;
        }
    },

    async saveCloudKeys() {
        const openai = (document.getElementById('key-openai').value || '').trim();
        const anthropic = (document.getElementById('key-anthropic').value || '').trim();
        try {
            if (openai) {
                await API.post('/api/provider-secrets', { provider: 'openai', api_key: openai });
            }
            if (anthropic) {
                await API.post('/api/provider-secrets', { provider: 'anthropic', api_key: anthropic });
            }
            alert('Provider keys saved.');
            document.getElementById('key-openai').value = '';
            document.getElementById('key-anthropic').value = '';
        } catch (e) {
            alert('Failed to save keys: ' + e.message);
        }
    }
};

const Dashboard = {
    currentJobId: null,
    latestPlanTraceId: null,

    async init() {
        this.renderInbox();
        this.renderChatHistory();
        this.loadWorkflows();
        this.loadPermissions();
        this.loadTraces();
        this.refreshRuntime();
        // Poll for updates
        setInterval(() => this.renderInbox(), 5000);
        setInterval(() => this.pollJob(), 1000);
        setInterval(() => this.refreshRuntime(), 5000);

        // Navigation bindings
        document.querySelectorAll('.nav-item').forEach(el => {
            el.onclick = () => {
                document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
                el.classList.add('active');

                // Hide all views
                document.getElementById('view-inbox').classList.add('hidden');
                document.getElementById('view-automation').classList.add('hidden');
                document.getElementById('view-chat').classList.add('hidden');
                document.getElementById('view-settings').classList.add('hidden');

                const text = el.textContent.trim();
                if (text === 'Automation') {
                    document.getElementById('view-automation').classList.remove('hidden');
                } else if (text === 'Chat') {
                    document.getElementById('view-chat').classList.remove('hidden');
                } else if (text === 'Settings') {
                    document.getElementById('view-settings').classList.remove('hidden');
                    Dashboard.loadSettings();
                } else {
                    document.getElementById('view-inbox').classList.remove('hidden');
                }
            };
        });
    },

    actionParams() {
        return {
            workflow_path: (document.getElementById('auto-workflow')?.value || '').trim(),
            agent_id: (document.getElementById('auto-agent-id')?.value || 'default').trim(),
            task_id: (document.getElementById('auto-task-id')?.value || 'wf_demo').trim(),
            start_url: (document.getElementById('auto-start-url')?.value || 'https://example.com').trim(),
            model_name: (document.getElementById('auto-model')?.value || 'llama3.2:3b').trim()
        };
    },

    async loadWorkflows() {
        const select = document.getElementById('auto-workflow');
        if (!select) return;
        try {
            const res = await API.get('/api/workflows');
            const items = Array.isArray(res.items) ? res.items : [];
            select.innerHTML = items.map(p => `<option value="${p}">${p}</option>`).join('');
        } catch (e) {
            select.innerHTML = '<option value="">No workflows found</option>';
        }
    },

    async refreshRuntime() {
        const out = document.getElementById('auto-runtime');
        if (!out) return;
        try {
            const data = await API.get('/api/runtime-status');
            out.textContent = JSON.stringify(data, null, 2);
        } catch (e) {
            out.textContent = 'Runtime status failed: ' + e.message;
        }
    },

    async runAction(action) {
        try {
            const res = await API.post('/api/run', {
                action,
                params: this.actionParams(),
                approve_permission: true
            });
            this.currentJobId = res.job_id;
            document.getElementById('auto-output').textContent = 'Started: ' + action + ' (job=' + res.job_id + ')';
        } catch (e) {
            document.getElementById('auto-output').textContent = 'Action failed: ' + e.message;
        }
    },

    async pollJob() {
        if (!this.currentJobId) return;
        try {
            const job = await API.get(`/api/jobs/${this.currentJobId}`);
            document.getElementById('auto-output').textContent = job.output || ('Status: ' + job.status);
            const link = document.getElementById('auto-browser-link');
            if (job.browser_ui_url) {
                link.classList.remove('hidden');
                link.href = job.browser_ui_url;
                link.textContent = 'Open Live Browser: ' + job.browser_ui_url;
            } else {
                link.classList.add('hidden');
            }
            if (job.status !== 'running') {
                this.currentJobId = null;
            }
        } catch (e) {
            document.getElementById('auto-output').textContent = 'Job poll failed: ' + e.message;
            this.currentJobId = null;
        }
    },

    async loadPermissions() {
        const holder = document.getElementById('auto-permissions');
        if (!holder) return;
        const agentId = this.actionParams().agent_id || 'default';
        try {
            const data = await API.get(`/api/permissions?agent_id=${encodeURIComponent(agentId)}`);
            const perms = data.permissions || {};
            const tools = Object.keys(perms).sort();
            holder.innerHTML = tools.map(tool => `
                <label>${tool}</label>
                <select onchange="Dashboard.setPermission('${tool}', this.value)">
                    <option value="prompt" ${perms[tool] === 'prompt' ? 'selected' : ''}>prompt</option>
                    <option value="allow" ${perms[tool] === 'allow' ? 'selected' : ''}>allow</option>
                    <option value="deny" ${perms[tool] === 'deny' ? 'selected' : ''}>deny</option>
                </select>
            `).join('');
        } catch (e) {
            holder.innerHTML = '<div class="text-secondary">Failed to load permissions.</div>';
        }
    },

    async setPermission(toolName, decision) {
        try {
            await API.post('/api/permissions', {
                agent_id: this.actionParams().agent_id || 'default',
                tool_name: toolName,
                decision
            });
        } catch (e) {
            alert('Permission update failed: ' + e.message);
        }
    },

    async loadTraces() {
        const select = document.getElementById('auto-trace-select');
        if (!select) return;
        try {
            const data = await API.get('/api/traces');
            const items = Array.isArray(data.items) ? data.items : [];
            select.innerHTML = items.map(t => `<option value="${t.trace_id}">${t.trace_id} | ${t.type || 'trace'}</option>`).join('');
        } catch (e) {
            select.innerHTML = '<option value="">No traces</option>';
        }
    },

    async loadSelectedTrace() {
        const select = document.getElementById('auto-trace-select');
        const out = document.getElementById('auto-trace-output');
        if (!select || !out || !select.value) return;
        try {
            const trace = await API.get(`/api/traces/${encodeURIComponent(select.value)}`);
            out.textContent = JSON.stringify(trace, null, 2);
        } catch (e) {
            out.textContent = 'Trace load failed: ' + e.message;
        }
    },

    async runPlanPreview() {
        const out = document.getElementById('auto-output');
        try {
            const payload = await API.post('/api/plan-preview', {
                intent: (document.getElementById('auto-prompt').value || '').trim(),
                agent_id: this.actionParams().agent_id || 'default',
                model_name: this.actionParams().model_name,
                provider_name: 'ollama'
            });
            this.latestPlanTraceId = payload.trace_id || null;
            out.textContent = JSON.stringify(payload, null, 2);
            this.loadTraces();
        } catch (e) {
            out.textContent = 'Plan preview failed: ' + e.message;
        }
    },

    async executeLatestPlan() {
        const out = document.getElementById('auto-output');
        if (!this.latestPlanTraceId) {
            out.textContent = 'No plan trace. Run Dry-Run Planner first.';
            return;
        }
        try {
            const payload = await API.post('/api/execute-plan', {
                trace_id: this.latestPlanTraceId,
                agent_id: this.actionParams().agent_id || 'default',
                approve_permission: true,
                force: false
            });
            if (payload.job_id) {
                this.currentJobId = payload.job_id;
            }
            out.textContent = JSON.stringify(payload, null, 2);
        } catch (e) {
            out.textContent = 'Execute plan failed: ' + e.message;
        }
    },

    async renderInbox() {
        try {
            const res = await API.get('/api/task-inbox');
            console.log("Inbox items:", res.items);

            const list = document.getElementById('inbox-list');
            if (res.items.length === 0) {
                list.innerHTML = '<div class="text-center text-secondary" style="padding: 40px;">No tasks yet. Add one above!</div>';
                return;
            }

            // Simple diffing or just replace for now
            list.innerHTML = res.items.map(item => {
                let statusColor = '#666';
                let statusText = item.status;
                if (item.status === 'in_progress') {
                    statusColor = '#1a73e8'; // Blue
                    statusText = 'Running...';
                } else if (item.status === 'done') {
                    statusColor = '#137333'; // Green
                    statusText = '✓ Done';
                } else if (item.status === 'failed') {
                    statusColor = '#c5221f'; // Red
                    statusText = '✕ Failed';
                }

                return `
                <div class="card" style="display: flex; align-items: center; justify-content: space-between; padding: 16px;">
                    <div>
                        <div class="font-medium">${item.intent}</div>
                        <div class="text-xs" style="color:${statusColor}">${statusText} • ID: ${item.item_id}</div>
                    </div>
                     <button class="btn-text" onclick="Dashboard.deleteItem('${item.item_id}')">Dismiss</button>
                </div>
            `}).join('');

        } catch (e) { console.error("Inbox fetch failed", e); }
    },

    async addToInbox(intent) {
        if (!intent.trim()) return;
        try {
            await API.post('/api/task-inbox', { action: 'add', intent: intent });
            this.renderInbox();
        } catch (e) {
            alert(e.message);
        }
    },

    async deleteItem(id) {
        try {
            await API.post('/api/task-inbox', { action: 'delete', item_id: id });
            this.renderInbox();
        } catch (e) {
            alert(e.message);
        }
    },

    async renderChatHistory() {
        // Placeholder
    },

    async sendChat(text) {
        if (!text.trim()) return;
        try {
            // Optimistic UI update
            this.appendMessage('user', text);

            // Send to backend
            const res = await API.post('/api/assistant-chat', {
                prompt: text,
                agent_id: 'default',
                approve_permission: true // Auto-approve for guide interactions
            });

            if (res.status === 'ok') {
                this.appendMessage('assistant', res.message); // or res.response.message depends on API
            } else {
                this.appendMessage('system', `Error: ${res.error}`);
            }
        } catch (e) {
            this.appendMessage('system', `Communication failed: ${e.message}`);
        }
    },

    appendMessage(role, text) {
        const container = document.getElementById('chat-messages');
        if (!container) return;

        const div = document.createElement('div');
        div.className = `chat-msg ${role}`;
        div.innerText = typeof text === 'string' ? text : JSON.stringify(text);
        container.appendChild(div);
        container.scrollTop = container.scrollHeight;
    },

    async loadSettings() {
        try {
            const profile = await API.get('/api/profile');
            const settings = profile.provider_settings || {};

            // Set Model
            const ollamaConfig = settings.ollama || {};
            document.getElementById('settings-model').value = ollamaConfig.model_name || 'llama3.2:3b';

            // Keys are not returned for security, so we leave them blank
        } catch (e) {
            console.error("Failed to load settings", e);
        }
    },

    async saveSettings() {
        const model = document.getElementById('settings-model').value.trim();
        const keyOpenAI = document.getElementById('settings-key-openai').value.trim();
        const keyAnthropic = document.getElementById('settings-key-anthropic').value.trim();
        const btn = document.querySelector('#view-settings button[type="submit"]');

        btn.disabled = true;
        btn.textContent = "Saving...";

        try {
            const profile = await API.get('/api/profile');
            const providerSettings = profile.provider_settings || {};
            providerSettings.ollama = providerSettings.ollama || { enabled: true, model_name: '' };
            providerSettings.ollama.model_name = model || 'llama3.2:3b';
            const payload = {
                schema_version: profile.schema_version || 'profile.v2',
                agent_name: profile.agent_name || 'default',
                primary_provider: profile.primary_provider || 'ollama',
                provider_chain: profile.provider_chain || ['ollama'],
                model_name: model || profile.model_name || 'llama3.2:3b',
                provider_settings: providerSettings
            };
            await API.post('/api/profile', payload);
            if (keyOpenAI) {
                await API.post('/api/provider-secrets', { provider: 'openai', api_key: keyOpenAI });
            }
            if (keyAnthropic) {
                await API.post('/api/provider-secrets', { provider: 'anthropic', api_key: keyAnthropic });
            }
            btn.textContent = "Saved!";
            setTimeout(() => {
                btn.disabled = false;
                btn.textContent = "Save Changes";
            }, 2000);
        } catch (e) {
            alert('Failed to save settings: ' + e.message);
            btn.disabled = false;
            btn.textContent = "Save Changes";
        }
    }
};

window.Wizard = Wizard;
window.Dashboard = Dashboard;
Wizard.init();
