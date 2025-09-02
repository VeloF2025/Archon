# 🪶 Archon Light - Quick Start Mode

**Get started with Archon in under 5 minutes - no Docker, no Supabase, no complex setup!**

Perfect for:
- 🚀 **Quick evaluation** - Try Archon's core features instantly
- 💻 **Local development** - Simple setup for individual developers  
- 🧪 **Testing & experimentation** - Minimal configuration required
- 📚 **Learning** - Focus on features, not infrastructure

---

## 🎯 What You Get in Light Mode

### ✅ **Included (Zero Config)**
- **🤖 AI Agent Chat** - Direct AI assistant interaction
- **📝 Basic Knowledge Management** - File-based storage with search
- **💬 MCP Integration** - Connect with Claude Code, Cursor, Windsurf
- **📊 Simple UI** - Clean, responsive interface
- **🔌 API Access** - RESTful endpoints for automation

### ⚠️ **Not Included (Full Mode Only)**
- Real-time collaboration (Socket.IO)
- Advanced knowledge persistence (requires database)
- Project management features
- Multi-user support
- Advanced agent orchestration
- Background task processing

---

## ⚡ Quick Start (3 Steps)

### **Step 1: Clone & Setup** ⏱️ *1 minute*
```bash
# Clone the repository
git clone https://github.com/VeloF2025/Archon.git
cd Archon

# Copy light mode configuration
cp .env.light .env
```

### **Step 2: Configure API Key** ⏱️ *30 seconds*
```bash
# Edit .env file - only ONE required setting:
echo "OPENAI_API_KEY=your-openai-api-key-here" > .env

# OR use other providers:
# echo "ANTHROPIC_API_KEY=your-claude-api-key" > .env
# echo "GEMINI_API_KEY=your-gemini-api-key" > .env
```

### **Step 3: Launch** ⏱️ *30 seconds*
```bash
# Start Archon Light
npm run light

# Opens automatically at: http://localhost:3000
```

**🎉 That's it! Archon Light is running!**

---

## 🛠️ Light Mode Architecture

```
┌─────────────────────────────────────────┐
│             Archon Light                │
├─────────────────────────────────────────┤
│  🖥️  React UI (Port 3000)              │
│     • Agent Chat Interface             │
│     • Knowledge Search                 │
│     • MCP Client                       │
├─────────────────────────────────────────┤
│  🔧  Node.js Backend (Port 3001)       │
│     • Simple API Server                │
│     • File-based Storage               │
│     • LLM Provider Integration         │
├─────────────────────────────────────────┤
│  🤖  AI Agent Core                     │
│     • Direct LLM Integration           │
│     • Basic Agent Framework           │
│     • MCP Protocol Support            │
├─────────────────────────────────────────┤
│  💾  Local Storage                     │
│     • JSON files for data             │
│     • Local file indexing             │
│     • Simple search                   │
└─────────────────────────────────────────┘
```

### **vs Full Mode:**

| Feature | Light Mode | Full Mode |
|---------|------------|-----------|
| **Setup Time** | 3 minutes | 15+ minutes |
| **Dependencies** | Node.js only | Docker, Supabase, PostgreSQL |
| **Storage** | Local files | Database + Vector storage |
| **Agents** | Basic (5 agents) | Advanced (21+ agents) |
| **UI** | Simple interface | Full dashboard |
| **Collaboration** | Single user | Multi-user + real-time |
| **MCP Support** | ✅ Full support | ✅ Full support |

---

## 🎮 Usage Examples

### **Chat with AI Agents**
```bash
# Start a coding session
curl -X POST http://localhost:3001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Help me create a React component", "agent": "react-expert"}'
```

### **Search Knowledge**  
```bash
# Add documents to knowledge base
curl -X POST http://localhost:3001/api/knowledge \
  -H "Content-Type: application/json" \
  -d '{"content": "API documentation...", "title": "API Guide"}'

# Search knowledge
curl "http://localhost:3001/api/search?q=API%20endpoints"
```

### **MCP Integration**
```javascript
// Connect from Claude Code / Cursor
const mcp = new MCPClient('http://localhost:3001/mcp');
await mcp.callTool('archon:search_knowledge', { query: 'React hooks' });
```

---

## 🚀 Upgrade to Full Mode

When you're ready for advanced features:

```bash
# Stop Light Mode
npm run light:stop

# Switch to Full Mode  
cp .env.example .env
# Add Supabase credentials to .env

# Start Full Mode
docker-compose up -d
```

**Migration is seamless** - your data and configuration are preserved!

---

## 🔧 Configuration Options

### **Light Mode Environment Variables**

```bash
# Required (choose one)
OPENAI_API_KEY=sk-...           # OpenAI GPT models
ANTHROPIC_API_KEY=sk-ant-...    # Claude models  
GEMINI_API_KEY=...              # Google Gemini

# Optional
ARCHON_LIGHT_PORT=3000          # UI port (default: 3000)
ARCHON_API_PORT=3001            # API port (default: 3001)
LOG_LEVEL=info                  # Logging: debug, info, warn, error
STORAGE_PATH=./data             # Local storage directory
```

### **Available Agents (Light Mode)**

- **💬 chat-agent** - General conversation and assistance
- **⚛️ react-expert** - React/JavaScript development 
- **🐍 python-expert** - Python development and scripting
- **📝 documentation-writer** - Documentation generation
- **🔍 code-reviewer** - Code analysis and suggestions

---

## 🆘 Troubleshooting

### **Common Issues**

**Port already in use:**
```bash
# Change ports in .env
ARCHON_LIGHT_PORT=3002
ARCHON_API_PORT=3003
```

**API key not working:**
```bash
# Test your API key
curl -X POST http://localhost:3001/api/test-key
```

**Node.js errors:**
```bash
# Ensure Node.js 18+
node --version

# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

### **Need Help?**

- 💬 [GitHub Discussions](https://github.com/VeloF2025/Archon/discussions)
- 📖 [Full Documentation](README.md)
- 🐛 [Report Issues](https://github.com/VeloF2025/Archon/issues)

---

## 🎯 Next Steps

1. **Try the Chat Interface** - Start with simple coding questions
2. **Add Knowledge** - Upload your project documentation
3. **Test MCP Integration** - Connect with your favorite AI coding assistant  
4. **Explore APIs** - Build custom integrations
5. **Upgrade to Full Mode** - When you need advanced features

**Welcome to Archon! 🚀**

---

*Archon Light gets you from zero to productive AI-assisted development in under 5 minutes!*