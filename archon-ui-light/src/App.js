import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:3001';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [agent, setAgent] = useState('chat-agent');
  const [agents, setAgents] = useState([]);
  const [conversationId, setConversationId] = useState(null);
  const [health, setHealth] = useState(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    // Load agents and health check
    Promise.all([
      fetch(`${API_BASE}/api/agents`).then(r => r.json()),
      fetch(`${API_BASE}/api/health`).then(r => r.json())
    ]).then(([agentsData, healthData]) => {
      setAgents(agentsData);
      setHealth(healthData);
    }).catch(err => {
      console.error('Failed to load initial data:', err);
    });
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage = { role: 'user', content: input, timestamp: new Date() };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          message: input, 
          agent, 
          conversation_id: conversationId 
        })
      });

      const data = await response.json();
      
      if (response.ok) {
        const aiMessage = { 
          role: 'assistant', 
          content: data.response, 
          agent: data.agent,
          timestamp: new Date(data.timestamp)
        };
        setMessages(prev => [...prev, aiMessage]);
        if (!conversationId) setConversationId(data.conversation_id);
      } else {
        throw new Error(data.error || 'Failed to get response');
      }
    } catch (error) {
      const errorMessage = { 
        role: 'error', 
        content: `Error: ${error.message}`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
    setConversationId(null);
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <h1>🪶 Archon Light</h1>
          <div className="status">
            {health ? (
              <span className="status-good">
                ✅ {health.llmProvider || 'No LLM'} • {health.agents.length} agents
              </span>
            ) : (
              <span className="status-loading">Loading...</span>
            )}
          </div>
        </div>
      </header>

      <main className="main">
        <div className="chat-container">
          <div className="messages">
            {messages.length === 0 && (
              <div className="welcome">
                <h2>Welcome to Archon Light!</h2>
                <p>Start chatting with AI agents to get help with your development tasks.</p>
                <div className="quick-actions">
                  <button onClick={() => setInput("Help me create a React component")}>
                    React Help
                  </button>
                  <button onClick={() => setInput("Review my Python code")}>
                    Code Review
                  </button>
                  <button onClick={() => setInput("Write documentation for my API")}>
                    Documentation
                  </button>
                </div>
              </div>
            )}
            
            {messages.map((message, index) => (
              <div key={index} className={`message message-${message.role}`}>
                <div className="message-header">
                  <span className="role">
                    {message.role === 'user' ? '👤 You' : 
                     message.role === 'assistant' ? `🤖 ${message.agent || 'AI'}` : 
                     '❌ Error'}
                  </span>
                  <span className="timestamp">
                    {message.timestamp?.toLocaleTimeString()}
                  </span>
                </div>
                <div className="message-content">
                  <pre>{message.content}</pre>
                </div>
              </div>
            ))}
            
            {loading && (
              <div className="message message-loading">
                <div className="spinner"></div>
                <span>AI is thinking...</span>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>
          
          <div className="input-area">
            <div className="input-controls">
              <select 
                value={agent} 
                onChange={(e) => setAgent(e.target.value)}
                className="agent-select"
              >
                {agents.map(a => (
                  <option key={a.id} value={a.id}>{a.name}</option>
                ))}
              </select>
              
              <button onClick={clearChat} className="clear-btn">
                Clear Chat
              </button>
            </div>
            
            <div className="input-row">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type your message... (Enter to send, Shift+Enter for new line)"
                className="message-input"
                rows="1"
              />
              <button 
                onClick={sendMessage}
                disabled={loading || !input.trim()}
                className="send-btn"
              >
                Send
              </button>
            </div>
          </div>
        </div>
      </main>
      
      <footer className="footer">
        <p>
          Archon Light • <a href="https://github.com/VeloF2025/Archon">GitHub</a> • 
          Upgrade to <strong>Full Mode</strong> for advanced features
        </p>
      </footer>
    </div>
  );
}

export default App;