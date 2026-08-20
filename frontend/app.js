/**
 * Darwix AI Voice Agent & Intelligence Platform
 * Frontend Interactive Controller with Ultra-Fast Speech Synthesis & WebSockets
 */

class VoiceAgentApp {
    constructor() {
        this.ws = null;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.recordingInterval = null;
        this.recordingSeconds = 0;
        this.isRecording = false;
        this.autoPlayAudio = true;
        this.voiceMode = 'fast'; // 'fast' (WebSpeech API 0ms) or 'cloud' (Server EdgeTTS .mp3)
        this.turnCount = 0;
        
        // Web Audio Visualizer
        this.audioCtx = null;
        this.analyser = null;
        this.sourceNode = null;
        this.visualizerCanvas = document.getElementById('visualizerCanvas');
        this.canvasCtx = this.visualizerCanvas.getContext('2d');
        this.animationFrameId = null;

        // Elements
        this.recordBtn = document.getElementById('voiceRecordBtn');
        this.agentStatePill = document.getElementById('agentStatePill');
        this.agentStateText = document.getElementById('agentStateText');
        this.recordingTimer = document.getElementById('recordingTimer');
        this.messagesStream = document.getElementById('messagesStream');
        this.textChatForm = document.getElementById('textChatForm');
        this.textInput = document.getElementById('textInput');
        this.resetSessionBtn = document.getElementById('resetSessionBtn');
        this.audioToggleBtn = document.getElementById('audioToggleBtn');
        this.voiceModeToggleBtn = document.getElementById('voiceModeToggleBtn');
        this.voiceModeText = document.getElementById('voiceModeText');
        this.audioIconOn = document.getElementById('audioIconOn');
        this.audioIconOff = document.getElementById('audioIconOff');
        this.connectionBadge = document.getElementById('connectionBadge');
        this.connectionStatusText = document.getElementById('connectionStatusText');
        this.turnCounter = document.getElementById('turnCounter');
        this.lastLatencyVal = document.getElementById('lastLatencyVal');
        this.escalationStatusText = document.getElementById('escalationStatusText');
        this.globalAudioPlayer = document.getElementById('globalAudioPlayer');

        // Qualification Elements
        this.qualProgressBadge = document.getElementById('qualProgressBadge');
        this.decisionBanner = document.getElementById('decisionBanner');
        this.decisionIcon = document.getElementById('decisionIcon');
        this.decisionTitle = document.getElementById('decisionTitle');
        this.decisionDesc = document.getElementById('decisionDesc');

        this.init();
    }

    init() {
        this.setupWebSocket();
        this.setupEventListeners();
        this.setupVisualizer();
        this.fetchSystemStatus();
        this.fetchSessionState();
        this.preloadVoices();
    }

    fetchSessionState() {
        fetch('/api/session')
            .then(res => res.json())
            .then(data => {
                if (data && data.conversation_history && data.conversation_history.length > 0) {
                    this.messagesStream.innerHTML = '';
                    data.conversation_history.forEach(item => {
                        if (item.speaker === 'customer' || item.role === 'user') {
                            this.appendUserMessage(item.text);
                        } else if (item.speaker === 'agent' || item.role === 'assistant') {
                            this.appendAgentMessage({ response_text: item.text, citations: [] });
                        }
                    });
                    this.turnCount = data.conversation_history.length;
                    if (this.turnCounter) {
                        this.turnCounter.innerText = `${Math.floor(this.turnCount / 2)} turns`;
                    }
                    this.updateScorecard(data.qualification || {}, data.is_qualified, data.qualification_message);
                }
            })
            .catch(err => console.log("Session restore notice:", err));
    }

    preloadVoices() {
        if ('speechSynthesis' in window) {
            window.speechSynthesis.onvoiceschanged = () => {
                window.speechSynthesis.getVoices();
            };
            window.speechSynthesis.getVoices();
        }
    }

    /* --------------------------------------------------------------------------
       WebSocket Management
       -------------------------------------------------------------------------- */
    setupWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/voice`;
        
        this.updateConnectionStatus('connecting');

        try {
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                this.updateConnectionStatus('connected');
                console.log("WebSocket connected to Darwix Voice Engine");
            };

            this.ws.onmessage = (event) => {
                if (typeof event.data === 'string') {
                    try {
                        const data = JSON.parse(event.data);
                        this.handleServerMessage(data);
                    } catch (e) {
                        console.log("Raw WS message:", event.data);
                    }
                } else if (event.data instanceof Blob) {
                    if (this.voiceMode === 'cloud') {
                        this.playAudioBlob(event.data);
                    }
                }
            };

            this.ws.onclose = () => {
                this.updateConnectionStatus('disconnected');
                setTimeout(() => this.setupWebSocket(), 3000);
            };

            this.ws.onerror = (err) => {
                console.error("WebSocket error:", err);
                this.updateConnectionStatus('disconnected');
            };
        } catch (e) {
            console.error("Failed to initialize WebSocket:", e);
            this.updateConnectionStatus('disconnected');
        }
    }

    updateConnectionStatus(status) {
        const dot = this.connectionBadge.querySelector('.status-dot');
        dot.className = 'status-dot ' + status;
        
        if (status === 'connected') {
            this.connectionStatusText.textContent = 'Live Connected';
        } else if (status === 'connecting') {
            this.connectionStatusText.textContent = 'Connecting...';
        } else {
            this.connectionStatusText.textContent = 'Offline (Reconnecting)';
        }
    }

    async fetchSystemStatus() {
        try {
            const res = await fetch('/api/status');
            if (res.ok) {
                const data = await res.json();
                const modelEl = document.getElementById('activeModelText');
                if (modelEl) {
                    const primary = data.primary_provider === 'groq' ? `Groq • ${data.groq_model}` : `Gemini • ${data.gemini_model}`;
                    modelEl.textContent = primary;
                }
            }
        } catch (e) {
            console.log("Status check skipped");
        }
    }

    /* --------------------------------------------------------------------------
       Event Listeners
       -------------------------------------------------------------------------- */
    setupEventListeners() {
        // Voice Record Button Toggle
        this.recordBtn.addEventListener('click', () => {
            if (this.isRecording) {
                this.stopRecording();
            } else {
                this.startRecording();
            }
        });

        // Text Form Submit
        this.textChatForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const text = this.textInput.value.trim();
            if (!text) return;
            
            this.sendTextMessage(text);
            this.textInput.value = '';
        });

        // Reset Session Button
        this.resetSessionBtn.addEventListener('click', () => this.resetSession());

        // Voice Mode Selector Toggle (Fast 0ms WebSpeech vs Cloud Neural)
        if (this.voiceModeToggleBtn) {
            this.voiceModeToggleBtn.addEventListener('click', () => {
                if (this.voiceMode === 'fast') {
                    this.voiceMode = 'cloud';
                    this.voiceModeToggleBtn.className = 'btn-voice-mode cloud-mode';
                    this.voiceModeToggleBtn.querySelector('.mode-icon').textContent = '🎧';
                    this.voiceModeText.textContent = 'Cloud Voice (.mp3)';
                } else {
                    this.voiceMode = 'fast';
                    this.voiceModeToggleBtn.className = 'btn-voice-mode';
                    this.voiceModeToggleBtn.querySelector('.mode-icon').textContent = '⚡';
                    this.voiceModeText.textContent = 'Fast Voice (0ms)';
                }
            });
        }

        // Audio Mute / Toggle
        this.audioToggleBtn.addEventListener('click', () => {
            this.autoPlayAudio = !this.autoPlayAudio;
            if (this.autoPlayAudio) {
                this.audioIconOn.classList.remove('hidden');
                this.audioIconOff.classList.add('hidden');
            } else {
                this.audioIconOn.classList.add('hidden');
                this.audioIconOff.classList.remove('hidden');
                this.globalAudioPlayer.pause();
                if ('speechSynthesis' in window) {
                    window.speechSynthesis.cancel();
                }
            }
        });

        // Scenario Quick Chips
        document.querySelectorAll('.chip').forEach(chip => {
            chip.addEventListener('click', () => {
                const prompt = chip.getAttribute('data-prompt');
                if (prompt) {
                    this.sendTextMessage(prompt);
                }
            });
        });
    }

    /* --------------------------------------------------------------------------
       Audio Recording (Microphone)
       -------------------------------------------------------------------------- */
    async startRecording() {
        try {
            if ('speechSynthesis' in window) {
                window.speechSynthesis.cancel();
            }
            this.globalAudioPlayer.pause();

            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            this.initAudioVisualizer(stream);
            
            this.mediaRecorder = new MediaRecorder(stream);
            this.audioChunks = [];

            this.mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    this.audioChunks.push(e.data);
                }
            };

            this.mediaRecorder.onstop = () => {
                const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
                this.sendAudioData(audioBlob);
                
                stream.getTracks().forEach(track => track.stop());
                this.audioChunks = [];
            };

            this.mediaRecorder.start(250);
            this.isRecording = true;

            // Update UI State
            this.recordBtn.className = 'voice-btn recording';
            this.recordBtn.querySelector('.mic-icon').classList.add('hidden');
            this.recordBtn.querySelector('.stop-icon').classList.remove('hidden');
            
            this.setAgentState('recording', 'Listening... (Click to Send)');
            this.startTimer();

        } catch (err) {
            console.error("Microphone access error:", err);
            alert("Could not access microphone. Please allow microphone permissions or type your prompt in the text box.");
        }
    }

    stopRecording() {
        if (!this.isRecording) return;
        this.isRecording = false;

        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
        }

        this.stopTimer();

        this.recordBtn.className = 'voice-btn idle';
        this.recordBtn.querySelector('.mic-icon').classList.remove('hidden');
        this.recordBtn.querySelector('.stop-icon').classList.add('hidden');
        this.setAgentState('processing', 'Transcribing and thinking...');
    }

    startTimer() {
        this.recordingSeconds = 0;
        this.recordingTimer.textContent = '00:00';
        this.recordingTimer.classList.remove('hidden');
        
        clearInterval(this.recordingInterval);
        this.recordingInterval = setInterval(() => {
            this.recordingSeconds++;
            const mins = String(Math.floor(this.recordingSeconds / 60)).padStart(2, '0');
            const secs = String(this.recordingSeconds % 60).padStart(2, '0');
            this.recordingTimer.textContent = `${mins}:${secs}`;
        }, 1000);
    }

    stopTimer() {
        clearInterval(this.recordingInterval);
        this.recordingTimer.classList.add('hidden');
    }

    /* --------------------------------------------------------------------------
       Dispatch Messages (Voice & Text)
       -------------------------------------------------------------------------- */
    sendAudioData(audioBlob) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(audioBlob);
        } else {
            const formData = new FormData();
            formData.append('file', audioBlob, 'voice_query.webm');
            formData.append('session_id', 'web_session');
            
            fetch('/api/audio', { method: 'POST', body: formData })
                .then(res => res.json())
                .then(data => this.handleServerMessage(data))
                .catch(err => {
                    console.error("Audio upload error:", err);
                    this.setAgentState('ready', 'Click to Speak or Type Below');
                });
        }
    }

    sendTextMessage(text) {
        this.appendUserMessage(text);
        this.setAgentState('processing', 'Analyzing & retrieving policy...');

        const skipTTS = (this.voiceMode === 'fast');

        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'text', text: text, skip_tts: skipTTS }));
        } else {
            fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text, session_id: 'web_session', skip_tts: skipTTS })
            })
            .then(res => res.json())
            .then(data => this.handleServerMessage(data))
            .catch(err => {
                console.error("Chat error:", err);
                this.setAgentState('ready', 'Click to Speak or Type Below');
            });
        }
    }

    /* --------------------------------------------------------------------------
       Server Message Handling
       -------------------------------------------------------------------------- */
    handleServerMessage(payload) {
        if (payload.type === 'status') {
            if (payload.state === 'processing') {
                this.setAgentState('processing', payload.message || 'Processing turn...');
            }
            return;
        }

        if (payload.type === 'turn_result') {
            this.turnCount++;
            this.turnCounter.textContent = `${this.turnCount} turn${this.turnCount > 1 ? 's' : ''}`;
            
            if (payload.user_transcription && !this.wasLastMessageUser(payload.user_transcription)) {
                this.appendUserMessage(payload.user_transcription);
            }

            // Append Agent Bubble
            this.appendAgentMessage(payload);

            // Update Qualification Scorecard
            this.updateQualificationScorecard(payload);

            // Update Diagnostics
            if (payload.latency_ms) {
                this.lastLatencyVal.textContent = `${payload.latency_ms} ms`;
            }
            
            if (payload.requires_escalation) {
                this.escalationStatusText.textContent = '🚨 Human Escalation Triggered';
                this.escalationStatusText.style.color = '#f59e0b';
            }

            // Audio Playback Strategy
            if (this.autoPlayAudio) {
                if (this.voiceMode === 'fast' || !payload.audio_base64) {
                    // Instant Browser WebSpeech
                    this.speakWithWebSpeech(payload.response_text);
                } else if (payload.audio_base64) {
                    // Server Neural Audio
                    this.playBase64Audio(payload.audio_base64);
                } else {
                    this.setAgentState('ready', 'Click to Speak or Type Below');
                }
            } else {
                this.setAgentState('ready', 'Click to Speak or Type Below');
            }
        }
    }

    wasLastMessageUser(text) {
        const bubbles = this.messagesStream.querySelectorAll('.user-bubble .message-content');
        if (bubbles.length > 0) {
            const lastText = bubbles[bubbles.length - 1].textContent.trim();
            return lastText === text.trim();
        }
        return false;
    }

    /* --------------------------------------------------------------------------
       Chat UI Message Appending
       -------------------------------------------------------------------------- */
    appendUserMessage(text) {
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble user-bubble';
        bubble.innerHTML = `
            <div class="message-avatar">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
                    <circle cx="12" cy="7" r="4"/>
                </svg>
            </div>
            <div class="message-body">
                <div class="message-author">You</div>
                <div class="message-content">${this.escapeHtml(text)}</div>
            </div>
        `;
        this.messagesStream.appendChild(bubble);
        this.scrollToBottom();
    }

    appendAgentMessage(data) {
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble agent-bubble';
        
        let citationsHtml = '';
        if (data.citations && data.citations.length > 0) {
            citationsHtml = data.citations.map(c => `<span class="citation-pill">📚 ${this.escapeHtml(c)}</span>`).join('');
        }

        const audioReplayBtn = `
            <button class="btn-replay-audio" data-audio="${data.audio_base64 || ''}" data-text="${this.escapeHtml(data.response_text)}">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                    <polygon points="5 3 19 12 5 21 5 3"/>
                </svg>
                Replay Audio
            </button>
        `;

        const latencyTag = data.latency_ms ? `<span class="latency-pill">⚡ ${data.latency_ms}ms</span>` : '';

        bubble.innerHTML = `
            <div class="message-avatar">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                    <line x1="12" x2="12" y1="19" y2="22"/>
                </svg>
            </div>
            <div class="message-body">
                <div class="message-author">Darwix Assistant <span class="author-tag">AI Voice</span></div>
                <div class="message-content">${this.escapeHtml(data.response_text)}</div>
                <div class="message-footer">
                    ${citationsHtml}
                    ${latencyTag}
                    ${audioReplayBtn}
                </div>
            </div>
        `;

        this.messagesStream.appendChild(bubble);

        // Bind replay button
        const replayBtn = bubble.querySelector('.btn-replay-audio');
        if (replayBtn) {
            replayBtn.addEventListener('click', () => {
                const base64 = replayBtn.getAttribute('data-audio');
                const text = replayBtn.getAttribute('data-text');
                if (this.voiceMode === 'cloud' && base64) {
                    this.playBase64Audio(base64);
                } else {
                    this.speakWithWebSpeech(text);
                }
            });
        }

        this.scrollToBottom();
    }

    scrollToBottom() {
        this.messagesStream.scrollTop = this.messagesStream.scrollHeight;
    }

    /* --------------------------------------------------------------------------
       Instant Web Speech Synthesis (0ms latency)
       -------------------------------------------------------------------------- */
    speakWithWebSpeech(text) {
        if (!('speechSynthesis' in window) || !text) {
            this.setAgentState('ready', 'Click to Speak or Type Below');
            return;
        }

        window.speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.05;
        utterance.pitch = 1.0;

        const voices = window.speechSynthesis.getVoices();
        const preferredVoice = voices.find(v => (v.lang.includes('en') && (v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('Aria') || v.name.includes('Samantha')))) || voices.find(v => v.lang.startsWith('en'));
        if (preferredVoice) {
            utterance.voice = preferredVoice;
        }

        this.setAgentState('speaking', 'Speaking (0ms)...');
        this.recordBtn.className = 'voice-btn speaking';

        utterance.onend = () => {
            this.setAgentState('ready', 'Click to Speak or Type Below');
            this.recordBtn.className = 'voice-btn idle';
        };

        utterance.onerror = () => {
            this.setAgentState('ready', 'Click to Speak or Type Below');
            this.recordBtn.className = 'voice-btn idle';
        };

        window.speechSynthesis.speak(utterance);
    }

    /* --------------------------------------------------------------------------
       Audio Playback
       -------------------------------------------------------------------------- */
    playBase64Audio(base64) {
        try {
            this.setAgentState('speaking', 'Speaking...');
            this.recordBtn.className = 'voice-btn speaking';
            
            this.globalAudioPlayer.src = `data:audio/mp3;base64,${base64}`;
            this.globalAudioPlayer.play().catch(e => {
                console.log("Audio autoplay fallback to WebSpeech:", e);
                this.speakWithWebSpeech(this.lastSpokenText || '');
            });

            this.globalAudioPlayer.onended = () => {
                this.setAgentState('ready', 'Click to Speak or Type Below');
                this.recordBtn.className = 'voice-btn idle';
            };
        } catch (e) {
            console.error("Audio playback error:", e);
            this.setAgentState('ready', 'Click to Speak or Type Below');
        }
    }

    playAudioBlob(blob) {
        try {
            this.setAgentState('speaking', 'Speaking...');
            this.recordBtn.className = 'voice-btn speaking';

            const url = URL.createObjectURL(blob);
            this.globalAudioPlayer.src = url;
            this.globalAudioPlayer.play().catch(e => console.log("Autoplay error:", e));

            this.globalAudioPlayer.onended = () => {
                URL.revokeObjectURL(url);
                this.setAgentState('ready', 'Click to Speak or Type Below');
                this.recordBtn.className = 'voice-btn idle';
            };
        } catch (e) {
            console.error("Audio blob error:", e);
        }
    }

    /* --------------------------------------------------------------------------
       Dynamic Qualification Underwriting Scorecard
       -------------------------------------------------------------------------- */
    renderChecklist(requirements, qualification = {}) {
        this.requirements = requirements || this.requirements || [];
        const container = document.getElementById('qualificationChecklist');
        if (!container) return;

        container.innerHTML = '';
        this.requirements.forEach(req => {
            const item = document.createElement('div');
            item.className = 'check-item';
            item.id = `item_${req.key}`;

            const icon = req.icon || '📋';
            const title = req.title || req.key;
            const reqDisplay = req.req ? `Requirement: ${req.req}` : '';

            item.innerHTML = `
                <div class="check-left">
                    <div class="check-icon">${icon}</div>
                    <div class="check-details">
                        <div class="check-title">${this.escapeHtml(title)}</div>
                        <div class="check-req">${this.escapeHtml(reqDisplay)}</div>
                    </div>
                </div>
                <div class="check-right">
                    <span class="extracted-val" id="val_${req.key}">—</span>
                    <span class="status-pill-small pending" id="badge_${req.key}">Pending</span>
                </div>
            `;
            container.appendChild(item);
        });
    }

    updateQualificationScorecard(payload) {
        if (payload.requirements && (!this.requirements || this.requirements.length === 0 || JSON.stringify(payload.requirements) !== JSON.stringify(this.requirements))) {
            this.requirements = payload.requirements;
            this.renderChecklist(this.requirements, payload.qualification || {});
        }

        const q = payload.qualification || {};
        let metCount = 0;
        let failCount = 0;
        const totalCount = (this.requirements && this.requirements.length > 0) ? this.requirements.length : 4;

        if (this.requirements && this.requirements.length > 0) {
            this.requirements.forEach(req => {
                const valEl = document.getElementById(`val_${req.key}`);
                const badgeEl = document.getElementById(`badge_${req.key}`);
                const val = q[req.key];

                if (!valEl || !badgeEl) return;

                if (val !== undefined && val !== null && String(val).trim() !== '' && String(val).trim() !== '—') {
                    let passed = false;
                    let displayVal = String(val);

                    if (req.type === 'min_number') {
                        const num = parseFloat(String(val).replace(/[^0-9.]/g, ''));
                        if (!isNaN(num)) {
                            if (req.key === 'monthly_revenue') displayVal = `$${num.toLocaleString()}`;
                            else if (req.key === 'time_in_business_months') displayVal = `${Math.floor(num)} mos`;
                            else displayVal = `${Math.floor(num)}`;

                            if (req.min_value !== undefined && req.min_value !== null) {
                                passed = (num >= req.min_value);
                            } else {
                                passed = true;
                            }
                        }
                    } else if (req.type === 'location') {
                        displayVal = String(val);
                        const vLower = displayVal.toLowerCase();
                        const usKeywords = ['us', 'usa', 'united states', 'california', 'texas', 'new york', 'florida', 'illinois', 'ohio', 'georgia', 'ca', 'tx', 'ny', 'fl', 'nc', 'sc', 'wa', 'or', 'az', 'co', 'pa', 'mi', 'nj', 'va'];
                        passed = usKeywords.some(kw => vLower.includes(kw));
                    } else {
                        displayVal = String(val);
                        const vLower = displayVal.toLowerCase();
                        passed = !['no', 'false', 'none', 'denied', '0'].some(neg => vLower.includes(neg));
                    }

                    valEl.textContent = displayVal;
                    if (passed) {
                        badgeEl.className = 'status-pill-small passed';
                        badgeEl.textContent = 'Passed';
                        metCount++;
                    } else {
                        badgeEl.className = 'status-pill-small failed';
                        badgeEl.textContent = 'Below Req';
                        failCount++;
                    }
                } else {
                    valEl.textContent = '—';
                    badgeEl.className = 'status-pill-small pending';
                    badgeEl.textContent = 'Pending';
                }
            });
        }

        // Update overall badge
        this.qualProgressBadge.textContent = `${metCount} / ${totalCount} Met`;

        // Update Decision Banner
        if (payload.requires_escalation) {
            this.decisionBanner.className = 'decision-banner state-escalated';
            this.decisionIcon.textContent = '🚨';
            this.decisionTitle.textContent = 'Escalation Required';
            this.decisionDesc.textContent = 'A human loan officer has been notified to assist further.';
        } else if (payload.is_qualified || metCount === totalCount) {
            this.decisionBanner.className = 'decision-banner state-qualified';
            this.decisionIcon.textContent = '🎉';
            this.decisionTitle.textContent = 'Preliminary Qualified!';
            this.decisionDesc.textContent = payload.qualification_message || 'Applicant satisfies all business loan criteria.';
        } else if (failCount > 0) {
            this.decisionBanner.className = 'decision-banner state-unqualified';
            this.decisionIcon.textContent = '⚠️';
            this.decisionTitle.textContent = 'Below Criteria';
            this.decisionDesc.textContent = payload.qualification_message || 'One or more requirements are below policy minimums.';
        } else if (metCount > 0) {
            this.decisionBanner.className = 'decision-banner state-pending';
            this.decisionIcon.textContent = '📋';
            this.decisionTitle.textContent = 'Qualification In Progress';
            this.decisionDesc.textContent = `${totalCount - metCount} more criteria needed for evaluation.`;
        } else {
            this.decisionBanner.className = 'decision-banner state-pending';
            this.decisionIcon.textContent = '⏳';
            this.decisionTitle.textContent = 'Evaluating Eligibility';
            this.decisionDesc.textContent = 'Provide business operating details to qualify.';
        }
    }

    /* --------------------------------------------------------------------------
       Session Reset
       -------------------------------------------------------------------------- */
    async resetSession() {
        try {
            if ('speechSynthesis' in window) {
                window.speechSynthesis.cancel();
            }
            this.globalAudioPlayer.pause();

            await fetch('/api/reset', { method: 'POST' });
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'reset' }));
            }
        } catch (e) {
            console.error("Reset error:", e);
        }

        // Reset Dynamic Checklist
        if (this.requirements && this.requirements.length > 0) {
            this.renderChecklist(this.requirements, {});
        }

        const totalCount = (this.requirements && this.requirements.length > 0) ? this.requirements.length : 4;
        this.qualProgressBadge.textContent = `0 / ${totalCount} Met`;
        this.decisionBanner.className = 'decision-banner state-pending';
        this.decisionIcon.textContent = '⏳';
        this.decisionTitle.textContent = 'Evaluating Eligibility';
        this.decisionDesc.textContent = 'Provide business operating details to qualify.';
        
        this.escalationStatusText.textContent = 'Not Requested';
        this.escalationStatusText.style.color = 'inherit';

        // Clear Messages except welcome
        this.messagesStream.innerHTML = `
            <div class="message-bubble agent-bubble welcome-message">
                <div class="message-avatar">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
                        <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                        <line x1="12" x2="12" y1="19" y2="22"/>
                    </svg>
                </div>
                <div class="message-body">
                    <div class="message-author">Darwix Assistant <span class="author-tag">AI Voice</span></div>
                    <div class="message-content">
                        Session refreshed! How long has your business been operating, and what is your average monthly revenue?
                    </div>
                </div>
            </div>
        `;
        this.turnCount = 0;
        this.turnCounter.textContent = '0 turns';
        this.setAgentState('ready', 'Click to Speak or Type Below');
    }

    setAgentState(state, text) {
        this.agentStatePill.className = 'status-pill ' + state;
        this.agentStateText.textContent = text;
    }

    /* --------------------------------------------------------------------------
       Canvas Audio Visualizer
       -------------------------------------------------------------------------- */
    setupVisualizer() {
        const resizeCanvas = () => {
            const rect = this.visualizerCanvas.getBoundingClientRect();
            this.visualizerCanvas.width = rect.width * window.devicePixelRatio || 400;
            this.visualizerCanvas.height = rect.height * window.devicePixelRatio || 140;
            this.canvasCtx.scale(window.devicePixelRatio, window.devicePixelRatio);
        };
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);
        this.drawIdleVisualizer();
    }

    initAudioVisualizer(stream) {
        if (!this.audioCtx) {
            this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (this.audioCtx.state === 'suspended') {
            this.audioCtx.resume();
        }
        this.analyser = this.audioCtx.createAnalyser();
        this.analyser.fftSize = 64;
        this.sourceNode = this.audioCtx.createMediaStreamSource(stream);
        this.sourceNode.connect(this.analyser);
        this.drawActiveWave();
    }

    drawActiveWave() {
        if (!this.isRecording) {
            this.drawIdleVisualizer();
            return;
        }

        const width = this.visualizerCanvas.width / window.devicePixelRatio;
        const height = this.visualizerCanvas.height / window.devicePixelRatio;
        const bufferLength = this.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        this.analyser.getByteFrequencyData(dataArray);

        this.canvasCtx.clearRect(0, 0, width, height);

        const barWidth = (width / bufferLength) * 1.5;
        let x = width / 2 - (bufferLength * barWidth) / 2;

        for (let i = 0; i < bufferLength; i++) {
            const barHeight = (dataArray[i] / 255) * (height * 0.75);
            
            const gradient = this.canvasCtx.createLinearGradient(0, height / 2 - barHeight, 0, height / 2 + barHeight);
            gradient.addColorStop(0, '#6366f1');
            gradient.addColorStop(0.5, '#06b6d4');
            gradient.addColorStop(1, '#38bdf8');

            this.canvasCtx.fillStyle = gradient;
            this.canvasCtx.beginPath();
            this.canvasCtx.roundRect(x, (height - barHeight) / 2, barWidth - 3, barHeight, 4);
            this.canvasCtx.fill();

            x += barWidth;
        }

        this.animationFrameId = requestAnimationFrame(() => this.drawActiveWave());
    }

    drawIdleVisualizer() {
        if (this.isRecording) return;
        
        const width = this.visualizerCanvas.width / window.devicePixelRatio;
        const height = this.visualizerCanvas.height / window.devicePixelRatio;
        const time = Date.now() * 0.002;

        this.canvasCtx.clearRect(0, 0, width, height);

        this.canvasCtx.beginPath();
        this.canvasCtx.lineWidth = 2;
        
        const gradient = this.canvasCtx.createLinearGradient(0, 0, width, 0);
        gradient.addColorStop(0, 'rgba(99, 102, 241, 0.05)');
        gradient.addColorStop(0.5, 'rgba(6, 182, 212, 0.4)');
        gradient.addColorStop(1, 'rgba(99, 102, 241, 0.05)');
        
        this.canvasCtx.strokeStyle = gradient;

        for (let x = 0; x < width; x += 4) {
            const y = height / 2 + Math.sin(x * 0.02 + time) * 10 * Math.sin(time * 0.5);
            if (x === 0) {
                this.canvasCtx.moveTo(x, y);
            } else {
                this.canvasCtx.lineTo(x, y);
            }
        }
        this.canvasCtx.stroke();

        this.animationFrameId = requestAnimationFrame(() => this.drawIdleVisualizer());
    }

    escapeHtml(str) {
        if (!str) return '';
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.app = new VoiceAgentApp();
});
