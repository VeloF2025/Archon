#!/usr/bin/env node
/**
 * Archon Light Mode Server
 * ========================
 * Lightweight API server with minimal dependencies
 */

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');
const morgan = require('morgan');
const fs = require('fs-extra');
const path = require('path');
const { v4: uuidv4 } = require('uuid');
require('dotenv').config();

const app = express();
const PORT = process.env.ARCHON_API_PORT || 3001;
const STORAGE_PATH = process.env.STORAGE_PATH || './archon-light-data';

// Middleware
app.use(helmet());
app.use(compression());
app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(morgan('combined'));

// Initialize LLM client based on available API keys
let llmClient = null;
const initLLM = () => {
  if (process.env.OPENAI_API_KEY && process.env.OPENAI_API_KEY.startsWith('sk-')) {
    const OpenAI = require('openai');
    llmClient = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
    console.log('✅ OpenAI client initialized');
    return 'openai';
  }
  
  if (process.env.ANTHROPIC_API_KEY && process.env.ANTHROPIC_API_KEY.startsWith('sk-ant-')) {
    const Anthropic = require('@anthropic-ai/sdk');
    llmClient = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
    console.log('✅ Anthropic client initialized');
    return 'anthropic';
  }
  
  if (process.env.GEMINI_API_KEY) {
    const { GoogleGenerativeAI } = require('@google/generative-ai');
    llmClient = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    console.log('✅ Gemini client initialized');
    return 'gemini';
  }
  
  console.log('❌ No valid API key found');
  return null;
};

const llmProvider = initLLM();

// Simple agents configuration
const LIGHT_AGENTS = {
  'chat-agent': {
    name: 'General Chat Assistant',
    systemPrompt: 'You are a helpful AI assistant focused on coding and development tasks. Be concise and practical.'
  },
  'react-expert': {
    name: 'React Expert',
    systemPrompt: 'You are a React and JavaScript expert. Help with component design, hooks, state management, and best practices.'
  },
  'python-expert': {
    name: 'Python Expert', 
    systemPrompt: 'You are a Python expert. Help with code, libraries, debugging, and best practices.'
  },
  'code-reviewer': {
    name: 'Code Reviewer',
    systemPrompt: 'You are a code reviewer. Analyze code for bugs, performance issues, security, and suggest improvements.'
  },
  'documentation-writer': {
    name: 'Documentation Writer',
    systemPrompt: 'You are a technical writer. Help create clear, comprehensive documentation for code and projects.'
  }
};

// Helper function to call LLM
async function callLLM(messages, agent = 'chat-agent') {
  if (!llmClient) {
    throw new Error('No LLM client configured');
  }
  
  const systemPrompt = LIGHT_AGENTS[agent]?.systemPrompt || LIGHT_AGENTS['chat-agent'].systemPrompt;
  const fullMessages = [
    { role: 'system', content: systemPrompt },
    ...messages
  ];
  
  try {
    if (llmProvider === 'openai') {
      const response = await llmClient.chat.completions.create({
        model: process.env.OPENAI_MODEL || 'gpt-4',
        messages: fullMessages,
        temperature: 0.7,
        max_tokens: 2000
      });
      return response.choices[0].message.content;
    }
    
    if (llmProvider === 'anthropic') {
      const response = await llmClient.messages.create({
        model: process.env.ANTHROPIC_MODEL || 'claude-3-sonnet-20240229',
        messages: fullMessages.slice(1), // Anthropic handles system prompt separately
        system: systemPrompt,
        max_tokens: 2000
      });
      return response.content[0].text;
    }
    
    if (llmProvider === 'gemini') {
      const model = llmClient.getGenerativeModel({ 
        model: process.env.GEMINI_MODEL || 'gemini-pro' 
      });
      const prompt = fullMessages.map(m => `${m.role}: ${m.content}`).join('\\n\\n');
      const result = await model.generateContent(prompt);
      return result.response.text();
    }
    
  } catch (error) {
    console.error('LLM Error:', error);
    throw new Error(`LLM request failed: ${error.message}`);
  }
}

// Storage helpers
const saveData = async (collection, data) => {
  const filePath = path.join(STORAGE_PATH, `${collection}.json`);
  await fs.ensureFile(filePath);
  
  let existing = [];
  try {
    existing = await fs.readJson(filePath);
  } catch (err) {
    // File doesn't exist or is empty
  }
  
  if (!Array.isArray(existing)) existing = [];
  existing.push({ ...data, id: uuidv4(), timestamp: new Date().toISOString() });
  
  await fs.writeJson(filePath, existing, { spaces: 2 });
  return existing[existing.length - 1];
};

const loadData = async (collection) => {
  try {
    const filePath = path.join(STORAGE_PATH, `${collection}.json`);
    return await fs.readJson(filePath) || [];
  } catch (err) {
    return [];
  }
};

// API Routes

// Health check
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    mode: 'light',
    llmProvider,
    agents: Object.keys(LIGHT_AGENTS),
    timestamp: new Date().toISOString()
  });
});

// Test API key
app.post('/api/test-key', async (req, res) => {
  try {
    const response = await callLLM([{ role: 'user', content: 'Say "API key working!" in exactly those words.' }]);
    res.json({ status: 'success', response });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Chat with agents
app.post('/api/chat', async (req, res) => {
  try {
    const { message, agent = 'chat-agent', conversation_id } = req.body;
    
    if (!message) {
      return res.status(400).json({ error: 'Message is required' });
    }
    
    if (!LIGHT_AGENTS[agent]) {
      return res.status(400).json({ error: `Unknown agent: ${agent}` });
    }
    
    // Load conversation history if provided
    let messages = [{ role: 'user', content: message }];
    if (conversation_id) {
      const chats = await loadData('chats');
      const conversation = chats.find(c => c.conversation_id === conversation_id);
      if (conversation && conversation.messages) {
        messages = [...conversation.messages, { role: 'user', content: message }];
      }
    }
    
    const response = await callLLM(messages, agent);
    
    // Save conversation
    const chatData = {
      conversation_id: conversation_id || uuidv4(),
      agent,
      messages: [...messages, { role: 'assistant', content: response }]
    };
    
    await saveData('chats', chatData);
    
    res.json({ 
      response, 
      agent,
      conversation_id: chatData.conversation_id,
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    console.error('Chat error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Knowledge management
app.post('/api/knowledge', async (req, res) => {
  try {
    const { content, title, url, tags = [] } = req.body;
    
    if (!content || !title) {
      return res.status(400).json({ error: 'Content and title are required' });
    }
    
    const knowledgeItem = await saveData('knowledge', {
      title,
      content,
      url,
      tags,
      searchable: content.toLowerCase() + ' ' + title.toLowerCase()
    });
    
    res.json(knowledgeItem);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Search knowledge
app.get('/api/search', async (req, res) => {
  try {
    const { q: query } = req.query;
    
    if (!query) {
      return res.status(400).json({ error: 'Query parameter "q" is required' });
    }
    
    const knowledge = await loadData('knowledge');
    const searchTerm = query.toLowerCase();
    
    const results = knowledge.filter(item => 
      item.searchable && item.searchable.includes(searchTerm)
    ).map(item => ({
      id: item.id,
      title: item.title,
      content: item.content.substring(0, 200) + '...',
      url: item.url,
      tags: item.tags,
      timestamp: item.timestamp
    }));
    
    res.json({ query, results, count: results.length });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// List available agents
app.get('/api/agents', (req, res) => {
  const agents = Object.entries(LIGHT_AGENTS).map(([key, value]) => ({
    id: key,
    name: value.name,
    description: value.systemPrompt
  }));
  
  res.json(agents);
});

// MCP Protocol Support (Basic)
app.post('/mcp/tools/archon:chat', async (req, res) => {
  try {
    const { arguments: args } = req.body;
    const { message, agent = 'chat-agent' } = args || {};
    
    const response = await callLLM([{ role: 'user', content: message }], agent);
    
    res.json({
      content: [{ type: 'text', text: response }]
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/mcp/tools/archon:search_knowledge', async (req, res) => {
  try {
    const { arguments: args } = req.body;
    const { query } = args || {};
    
    const knowledge = await loadData('knowledge');
    const searchTerm = query.toLowerCase();
    
    const results = knowledge.filter(item => 
      item.searchable && item.searchable.includes(searchTerm)
    ).slice(0, 5);
    
    const text = results.length > 0 
      ? results.map(r => `${r.title}: ${r.content.substring(0, 100)}...`).join('\\n\\n')
      : 'No results found';
    
    res.json({
      content: [{ type: 'text', text }]
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Static file serving for UI (if built)
app.use(express.static('archon-ui-light/build'));
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../../archon-ui-light/build/index.html'));
});

// Error handling
app.use((error, req, res, next) => {
  console.error('Server error:', error);
  res.status(500).json({ error: 'Internal server error' });
});

// Start server
app.listen(PORT, () => {
  console.log(`🪶 Archon Light Mode API running on port ${PORT}`);
  console.log(`📡 Mode: ${process.env.ARCHON_MODE || 'light'}`);
  console.log(`🤖 LLM Provider: ${llmProvider || 'none'}`);
  console.log(`💾 Storage: ${STORAGE_PATH}`);
  console.log(`🌐 Health check: http://localhost:${PORT}/api/health`);
  
  if (!llmProvider) {
    console.log('❌ No API key configured - please add to .env file');
  }
});

module.exports = app;