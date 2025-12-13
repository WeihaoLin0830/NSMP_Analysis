/**
 * NSMP Cancer Assistant - Chatbox Component
 * RAG-powered chatbot for patients and medical professionals
 */

class ChatBox {
    constructor(options = {}) {
        this.role = options.role || localStorage.getItem('userRole') || 'patient';
        this.sessionId = options.sessionId || this.generateSessionId();
        this.isOpen = false;
        this.isLoading = false;
        this.messages = [];
        this.suggestedQuestions = [];
        this.systemAvailable = false;
        
        this.init();
    }
    
    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    init() {
        this.createChatElements();
        this.attachEventListeners();
        this.checkSystemStatus();
        this.loadSuggestions();
    }
    
    createChatElements() {
        // Create toggle button
        const toggleBtn = document.createElement('button');
        toggleBtn.className = 'chat-toggle';
        toggleBtn.id = 'chatToggle';
        toggleBtn.innerHTML = `
            <span class="chat-toggle-icon">💬</span>
            <span class="notification-badge" id="chatNotification">1</span>
        `;
        
        // Create chat container
        const chatContainer = document.createElement('div');
        chatContainer.className = 'chat-container';
        chatContainer.id = 'chatContainer';
        chatContainer.innerHTML = `
            <div class="chat-header">
                <div class="chat-header-icon">🤖</div>
                <div class="chat-header-info">
                    <h4 class="chat-header-title">Asistente NSMP</h4>
                    <div class="chat-header-status">
                        <span class="status-dot" id="statusDot"></span>
                        <span id="statusText">Conectando...</span>
                    </div>
                </div>
                <button class="chat-minimize" id="chatMinimize">✕</button>
            </div>
            
            <div class="chat-messages" id="chatMessages">
                <div class="welcome-message">
                    <h3>¡Hola! Soy tu asistente virtual</h3>
                    <p>Puedo ayudarte con información sobre cáncer de endometrio y el perfil NSMP. 
                    ${this.role === 'patient' 
                        ? 'Estoy aquí para resolver tus dudas de forma clara y comprensible.' 
                        : 'Tengo acceso a las guías clínicas y literatura médica actualizada.'}
                    </p>
                </div>
            </div>
            
            <div class="suggested-questions" id="suggestionsContainer">
                <div class="suggestions-title">Preguntas sugeridas:</div>
                <div class="suggestions-list" id="suggestionsList"></div>
            </div>
            
            <div class="chat-input-area">
                <div class="chat-input-wrapper">
                    <textarea 
                        class="chat-input" 
                        id="chatInput" 
                        placeholder="Escribe tu pregunta..." 
                        rows="1"
                    ></textarea>
                    <button class="chat-send" id="chatSend">
                        <span class="chat-send-icon">➤</span>
                    </button>
                </div>
            </div>
        `;
        
        // Add to document
        document.body.appendChild(toggleBtn);
        document.body.appendChild(chatContainer);
        
        // Store references
        this.elements = {
            toggle: toggleBtn,
            container: chatContainer,
            messages: document.getElementById('chatMessages'),
            input: document.getElementById('chatInput'),
            sendBtn: document.getElementById('chatSend'),
            minimize: document.getElementById('chatMinimize'),
            statusDot: document.getElementById('statusDot'),
            statusText: document.getElementById('statusText'),
            suggestionsList: document.getElementById('suggestionsList'),
            suggestionsContainer: document.getElementById('suggestionsContainer'),
            notification: document.getElementById('chatNotification')
        };
    }
    
    attachEventListeners() {
        // Toggle chat
        this.elements.toggle.addEventListener('click', () => this.toggle());
        this.elements.minimize.addEventListener('click', () => this.close());
        
        // Send message
        this.elements.sendBtn.addEventListener('click', () => this.sendMessage());
        
        // Enter to send (Shift+Enter for new line)
        this.elements.input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Auto-resize textarea
        this.elements.input.addEventListener('input', () => {
            this.elements.input.style.height = 'auto';
            this.elements.input.style.height = Math.min(this.elements.input.scrollHeight, 100) + 'px';
        });
    }
    
    async checkSystemStatus() {
        try {
            const response = await fetch('/api/chat/status');
            const data = await response.json();
            
            this.systemAvailable = data.available;
            this.updateStatus(data.available);
        } catch (error) {
            console.error('Error checking system status:', error);
            this.updateStatus(false);
        }
    }
    
    updateStatus(available) {
        if (available) {
            this.elements.statusDot.classList.remove('offline');
            this.elements.statusText.textContent = 'En línea';
        } else {
            this.elements.statusDot.classList.add('offline');
            this.elements.statusText.textContent = 'Modo limitado';
        }
    }
    
    async loadSuggestions() {
        try {
            const response = await fetch(`/api/chat/suggestions/${this.role}`);
            const data = await response.json();
            
            this.suggestedQuestions = data.questions || [];
            this.renderSuggestions();
        } catch (error) {
            console.error('Error loading suggestions:', error);
            // Use default suggestions
            this.suggestedQuestions = [
                "¿Qué es el perfil NSMP?",
                "¿Cuáles son los tratamientos disponibles?",
                "¿Qué seguimiento se recomienda?"
            ];
            this.renderSuggestions();
        }
    }
    
    renderSuggestions() {
        const container = this.elements.suggestionsList;
        container.innerHTML = '';
        
        this.suggestedQuestions.slice(0, 4).forEach(question => {
            const chip = document.createElement('button');
            chip.className = 'suggestion-chip';
            chip.textContent = question.length > 40 ? question.substring(0, 40) + '...' : question;
            chip.title = question;
            chip.addEventListener('click', () => {
                this.elements.input.value = question;
                this.sendMessage();
            });
            container.appendChild(chip);
        });
    }
    
    toggle() {
        if (this.isOpen) {
            this.close();
        } else {
            this.open();
        }
    }
    
    open() {
        this.isOpen = true;
        this.elements.container.classList.add('open');
        this.elements.toggle.classList.add('active');
        this.elements.notification.style.display = 'none';
        this.elements.input.focus();
    }
    
    close() {
        this.isOpen = false;
        this.elements.container.classList.remove('open');
        this.elements.toggle.classList.remove('active');
    }
    
    async sendMessage() {
        const message = this.elements.input.value.trim();
        if (!message || this.isLoading) return;
        
        // Clear input
        this.elements.input.value = '';
        this.elements.input.style.height = 'auto';
        
        // Hide suggestions after first message
        this.elements.suggestionsContainer.style.display = 'none';
        
        // Add user message
        this.addMessage('user', message);
        
        // Show typing indicator
        this.showTypingIndicator();
        
        try {
            this.isLoading = true;
            this.elements.sendBtn.disabled = true;
            
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: message,
                    session_id: this.sessionId,
                    role: this.role
                })
            });
            
            if (!response.ok) {
                throw new Error('Error en la respuesta del servidor');
            }
            
            const data = await response.json();
            
            // Remove typing indicator
            this.hideTypingIndicator();
            
            // Add assistant message
            this.addMessage('assistant', data.answer, data.sources);
            
            // Update session ID if provided
            if (data.session_id) {
                this.sessionId = data.session_id;
            }
            
        } catch (error) {
            console.error('Error sending message:', error);
            this.hideTypingIndicator();
            this.addErrorMessage('Lo siento, hubo un error al procesar tu mensaje. Por favor, intenta de nuevo.');
        } finally {
            this.isLoading = false;
            this.elements.sendBtn.disabled = false;
        }
    }
    
    addMessage(role, content, sources = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${role}`;
        
        let html = `<div class="message-bubble">${this.formatMessage(content)}</div>`;
        
        // Add sources for assistant messages
        if (sources && sources.length > 0) {
            html += `
                <div class="message-sources">
                    <div class="message-sources-title">Fuentes consultadas:</div>
                    ${sources.slice(0, 3).map(s => `
                        <div class="source-item">${s.document} (p.${s.page})</div>
                    `).join('')}
                </div>
            `;
        }
        
        messageDiv.innerHTML = html;
        this.elements.messages.appendChild(messageDiv);
        
        // Store message
        this.messages.push({ role, content, sources });
        
        // Scroll to bottom
        this.scrollToBottom();
    }
    
    formatMessage(text) {
        // Convert markdown-like formatting to HTML
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>')
            .replace(/⚠️/g, '<span class="warning-icon">⚠️</span>');
    }
    
    addErrorMessage(text) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'chat-error';
        errorDiv.textContent = text;
        this.elements.messages.appendChild(errorDiv);
        this.scrollToBottom();
    }
    
    showTypingIndicator() {
        const typing = document.createElement('div');
        typing.className = 'typing-indicator';
        typing.id = 'typingIndicator';
        typing.innerHTML = '<span></span><span></span><span></span>';
        this.elements.messages.appendChild(typing);
        this.scrollToBottom();
    }
    
    hideTypingIndicator() {
        const typing = document.getElementById('typingIndicator');
        if (typing) {
            typing.remove();
        }
    }
    
    scrollToBottom() {
        this.elements.messages.scrollTop = this.elements.messages.scrollHeight;
    }
}

// Initialize chatbox when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Only initialize on pages that should have chat
    const excludedPages = ['/'];  // Don't show on role selection
    
    if (!excludedPages.includes(window.location.pathname)) {
        window.chatBox = new ChatBox();
    }
});
