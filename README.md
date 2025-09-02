<div align="center">

# 🏗️ Archon

**The Complete AI-Powered Development Workspace**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://docs.docker.com/compose/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue.svg)](https://www.typescriptlang.org/)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://reactjs.org/)

<img src="./archon-ui-main/public/archon-main-graphic.png" alt="Archon Main Graphic" width="853" height="422">

*Transform your development workflow with AI-powered agents, intelligent knowledge management, and real-time collaboration*

[🚀 Quick Start](#-quick-start) • [📚 Documentation](#-documentation) • [🏗️ Architecture](#-architecture) • [🤝 Contributing](CONTRIBUTING.md)

</div>

---

## 🎯 What is Archon?

**Archon is a complete AI-powered development workspace** that transforms how you build software with artificial intelligence. It combines intelligent agents, advanced knowledge management, and real-time collaboration into a single, powerful platform.

### 🌟 Key Value Propositions

- **🤖 Multi-Agent AI System**: Deploy 21+ specialized AI agents for every aspect of development - from architecture design to security auditing
- **📚 Intelligent Knowledge Management**: Automatically crawl, index, and search your documentation with advanced RAG strategies
- **🔄 Real-Time Collaboration**: Live updates, progress tracking, and seamless integration with your favorite AI coding assistants
- **🏗️ Project Management**: Built-in task management with AI-assisted planning and execution
- **🔌 MCP Integration**: Connect with Claude Code, Cursor, Windsurf, and any MCP-compatible client
- **📊 Confidence Scoring**: Advanced DeepConf engine provides reliability metrics for AI-generated code
- **🎯 SCWT Metrics**: Real-time quality dashboards with comprehensive testing frameworks

### 💡 Perfect For

- **Individual Developers**: Supercharge your coding workflow with AI agents
- **Development Teams**: Collaborate on complex projects with shared knowledge bases
- **AI Researchers**: Build and deploy custom AI agents with the integrated framework
- **Enterprise**: Scale AI-assisted development across your organization

> **Whether you're building a new project or enhancing an existing codebase, Archon's AI agents and knowledge management will dramatically improve your development productivity and code quality.**

## 🔗 Important Links

- **[GitHub Discussions](https://github.com/coleam00/Archon/discussions)** - Join the conversation and share ideas about Archon
- **[Contributing Guide](CONTRIBUTING.md)** - How to get involved and contribute to Archon
- **[Introduction Video](https://youtu.be/8pRc_s2VQIo)** - Getting started guide and vision for Archon
- **[Archon Kanban Board](https://github.com/users/coleam00/projects/1)** - Where maintainers are managing issues/features
- **[Dynamous AI Mastery](https://dynamous.ai)** - The birthplace of Archon - come join a vibrant community of other early AI adopters all helping each other transform their careers and businesses!

## Quick Start

### 🚀 Choose Your Setup Method

#### 🪶 **Archon Light (Recommended for first-time users)**
*Get started in under 5 minutes - no Docker, no database setup required!*

**Prerequisites:**
- [Node.js 18+](https://nodejs.org/)
- [OpenAI API key](https://platform.openai.com/api-keys) (or Anthropic/Gemini)

**Setup:**
```bash
git clone https://github.com/VeloF2025/Archon.git
cd Archon
cp .env.light .env
# Edit .env and add your API key
npm run light
```

**✨ Perfect for:** Quick evaluation, learning, individual development  
**📖 [Complete Light Mode Guide](QUICK_START_LIGHT.md)**

---

#### 🏗️ **Full Mode (Production & Teams)**
*Complete feature set with database, real-time collaboration, and advanced agents*

**Prerequisites:**
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Node.js 18+](https://nodejs.org/)  
- [Supabase](https://supabase.com/) account (free tier works)
- [OpenAI API key](https://platform.openai.com/api-keys) (Gemini and Ollama supported too!)

### Full Mode Setup Instructions

1. **Clone Repository**:
   ```bash
   git clone https://github.com/coleam00/archon.git
   ```
   ```bash
   cd archon
   ```
2. **Environment Configuration**:

   ```bash
   cp .env.example .env
   # Edit .env and add your Supabase credentials:
   # SUPABASE_URL=https://your-project.supabase.co
   # SUPABASE_SERVICE_KEY=your-service-key-here
   ```

   NOTE: Supabase introduced a new type of service key but use the legacy one (the longer one).

   OPTIONAL: If you want to enable the reranking RAG strategy, uncomment lines 20-22 in `python\requirements.server.txt`. This will significantly increase the size of the Archon Server container which is why it's off by default.

3. **Database Setup**: In your [Supabase project](https://supabase.com/dashboard) SQL Editor, copy, paste, and execute the contents of `migration/complete_setup.sql`

4. **Start Services** (choose one):

   **Full Docker Mode (Recommended for Normal Archon Usage)**

   ```bash
   docker compose up --build -d
   # or, to match a previously used explicit profile:
   docker compose --profile full up --build -d
   # or
   make dev-docker # (Alternative: Requires make installed )
   ```

   This starts all core microservices in Docker:
   - **Server**: Core API and business logic (Port: 8181)
   - **MCP Server**: Protocol interface for AI clients (Port: 8051)
   - **Agents (coming soon!)**: AI operations and streaming (Port: 8052)
   - **UI**: Web interface (Port: 3737)

   Ports are configurable in your .env as well!

5. **Configure API Keys**:
   - Open http://localhost:3737
   - Go to **Settings** → Select your LLM/embedding provider and set the API key (OpenAI is default)
   - Test by uploading a document or crawling a website

### 🚀 Quick Command Reference

| Command           | Description                                             |
| ----------------- | ------------------------------------------------------- |
| `make dev`        | Start hybrid dev (backend in Docker, frontend local) ⭐ |
| `make dev-docker` | Everything in Docker                                    |
| `make stop`       | Stop all services                                       |
| `make test`       | Run all tests                                           |
| `make lint`       | Run linters                                             |
| `make install`    | Install dependencies                                    |
| `make check`      | Check environment setup                                 |
| `make clean`      | Remove containers and volumes (with confirmation)       |

## 🔄 Database Reset (Start Fresh if Needed)

If you need to completely reset your database and start fresh:

<details>
<summary>⚠️ <strong>Reset Database - This will delete ALL data for Archon!</strong></summary>

1. **Run Reset Script**: In your Supabase SQL Editor, run the contents of `migration/RESET_DB.sql`

   ⚠️ WARNING: This will delete all Archon specific tables and data! Nothing else will be touched in your DB though.

2. **Rebuild Database**: After reset, run `migration/complete_setup.sql` to create all the tables again.

3. **Restart Services**:

   ```bash
   docker compose --profile full up -d
   ```

4. **Reconfigure**:
   - Select your LLM/embedding provider and set the API key again
   - Re-upload any documents or re-crawl websites

The reset script safely removes all tables, functions, triggers, and policies with proper dependency handling.

</details>

## 🛠️ Installing Make (OPTIONAL)

Make is required for the local development workflow. Installation varies by platform:

### Windows

```bash
# Option 1: Using Chocolatey
choco install make

# Option 2: Using Scoop
scoop install make

# Option 3: Using WSL2
wsl --install
# Then in WSL: sudo apt-get install make
```

### macOS

```bash
# Make comes pre-installed on macOS
# If needed: brew install make
```

### Linux

```bash
# Debian/Ubuntu
sudo apt-get install make

# RHEL/CentOS/Fedora
sudo yum install make
```

## ⚡ Quick Test

Once everything is running:

1. **Test Web Crawling**: Go to http://localhost:3737 → Knowledge Base → "Crawl Website" → Enter a doc URL (such as https://ai.pydantic.dev/llms-full.txt)
2. **Test Document Upload**: Knowledge Base → Upload a PDF
3. **Test Projects**: Projects → Create a new project and add tasks
4. **Integrate with your AI coding assistant**: MCP Dashboard → Copy connection config for your AI coding assistant

## 📚 Documentation

### 📖 Essential Documentation

| Document | Description | Audience |
|----------|-------------|----------|
| **[Installation Guide](INSTALLATION.md)** | Detailed setup instructions for all platforms | All Users |
| **[Configuration Guide](CONFIGURATION.md)** | Complete configuration reference | All Users |
| **[Architecture Overview](ARCHITECTURE.md)** | Technical architecture and design | Developers |
| **[Troubleshooting Guide](TROUBLESHOOTING.md)** | Solutions to common issues | All Users |
| **[Changelog](CHANGELOG.md)** | Version history and migration guides | All Users |

### 🚀 Deployment Guides

| Platform | Complexity | Time | Best For |
|----------|------------|------|----------|
| **[Docker Compose](docs/deployment/docker-compose.md)** | ⭐ Easy | 10 min | Local development, small teams |
| **[AWS Deployment](docs/deployment/aws.md)** | ⭐⭐ Medium | 30 min | Production, scalable workloads |
| **[Google Cloud](docs/deployment/gcp.md)** | ⭐⭐ Medium | 30 min | AI/ML workloads |
| **[Azure Deployment](docs/deployment/azure.md)** | ⭐⭐ Medium | 30 min | Enterprise environments |
| **[DigitalOcean](docs/deployment/digitalocean.md)** | ⭐ Easy | 20 min | Cost-effective hosting |

### 🏗️ Core Services

| Service            | Container Name | Default URL           | Purpose                           |
| ------------------ | -------------- | --------------------- | --------------------------------- |
| **Web Interface**  | archon-ui      | http://localhost:3737 | Main dashboard and controls       |
| **API Service**    | archon-server  | http://localhost:8181 | Web crawling, document processing |
| **MCP Server**     | archon-mcp     | http://localhost:8051 | Model Context Protocol interface  |
| **Agents Service** | archon-agents  | http://localhost:8052 | AI/ML operations, reranking       |

### 🤖 AI Agent System

Archon includes 21+ specialized AI agents:

#### 🏗️ Development Agents
- **System Architect**: Architecture design and planning
- **Code Implementer**: Zero-error code implementation  
- **Code Quality Reviewer**: Code review and validation
- **Test Coverage Validator**: Test creation and >95% coverage
- **Performance Optimizer**: Performance analysis and optimization
- **Security Auditor**: Security scanning and vulnerability detection

#### 🎨 Specialized Agents
- **UI/UX Designer**: Interface design and usability optimization
- **Database Architect**: Data modeling and optimization
- **API Design Architect**: RESTful API design and documentation
- **Documentation Writer**: Technical documentation generation
- **Error Handler**: Error detection and resolution strategies
- **Deployment Coordinator**: CI/CD and deployment automation

#### 📊 Analysis Agents
- **Strategic Planner**: Task breakdown and project planning
- **Data Analyst**: Data analysis and business insights
- **Configuration Manager**: System configuration management
- **Monitoring Agent**: System health and performance monitoring
- **Integration Tester**: Integration testing and validation
- **Quality Assurance**: QA processes and compliance validation

### 🔧 Advanced Features

#### 📊 DeepConf Confidence Engine
Advanced reliability scoring system that provides confidence metrics for AI-generated code:
- **Uncertainty Estimation**: Statistical uncertainty analysis
- **Consensus Validation**: Multi-agent agreement scoring
- **Dynamic Scoring**: Real-time confidence adjustment
- **Quality Gates**: Automatic quality threshold enforcement

#### 🎯 SCWT Metrics Dashboard
Real-time quality monitoring across four dimensions:
- **S**pecialized Code Quality: Complexity, coverage, technical debt
- **C**ollaboration Workflow: Team efficiency, knowledge sharing
- **W**orkflow Efficiency: Task completion, agent utilization
- **T**esting Coverage: Unit, integration, E2E test metrics

#### 🔌 MCP Integration
Full Model Context Protocol support with 10+ tools:
- `archon:perform_rag_query` - Knowledge base search with advanced RAG
- `archon:search_code_examples` - Code snippet discovery
- `archon:manage_project` - Project management operations
- `archon:manage_task` - Task lifecycle management
- `archon:get_available_sources` - Knowledge source inventory
- `archon:execute_agent` - AI agent deployment and execution

## What's Included

### 🧠 Knowledge Management

- **Smart Web Crawling**: Automatically detects and crawls entire documentation sites, sitemaps, and individual pages
- **Document Processing**: Upload and process PDFs, Word docs, markdown files, and text documents with intelligent chunking
- **Code Example Extraction**: Automatically identifies and indexes code examples from documentation for enhanced search
- **Vector Search**: Advanced semantic search with contextual embeddings for precise knowledge retrieval
- **Source Management**: Organize knowledge by source, type, and tags for easy filtering

### 🤖 AI Integration

- **Model Context Protocol (MCP)**: Connect any MCP-compatible client (Claude Code, Cursor, even non-AI coding assistants like Claude Desktop)
- **10 MCP Tools**: Comprehensive yet simple set of tools for RAG queries, task management, and project operations
- **Multi-LLM Support**: Works with OpenAI, Ollama, and Google Gemini models
- **RAG Strategies**: Hybrid search, contextual embeddings, and result reranking for optimal AI responses
- **Real-time Streaming**: Live responses from AI agents with progress tracking

### 📋 Project & Task Management

- **Hierarchical Projects**: Organize work with projects, features, and tasks in a structured workflow
- **AI-Assisted Creation**: Generate project requirements and tasks using integrated AI agents
- **Document Management**: Version-controlled documents with collaborative editing capabilities
- **Progress Tracking**: Real-time updates and status management across all project activities

### 🔄 Real-time Collaboration

- **WebSocket Updates**: Live progress tracking for crawling, processing, and AI operations
- **Multi-user Support**: Collaborative knowledge building and project management
- **Background Processing**: Asynchronous operations that don't block the user interface
- **Health Monitoring**: Built-in service health checks and automatic reconnection

## Architecture

### Microservices Structure

Archon uses true microservices architecture with clear separation of concerns:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend UI   │    │  Server (API)   │    │   MCP Server    │    │ Agents Service  │
│                 │    │                 │    │                 │    │                 │
│  React + Vite   │◄──►│    FastAPI +    │◄──►│    Lightweight  │◄──►│   PydanticAI    │
│  Port 3737      │    │    SocketIO     │    │    HTTP Wrapper │    │   Port 8052     │
│                 │    │    Port 8181    │    │    Port 8051    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │                        │
         └────────────────────────┼────────────────────────┼────────────────────────┘
                                  │                        │
                         ┌─────────────────┐               │
                         │    Database     │               │
                         │                 │               │
                         │    Supabase     │◄──────────────┘
                         │    PostgreSQL   │
                         │    PGVector     │
                         └─────────────────┘
```

### Service Responsibilities

| Service        | Location             | Purpose                      | Key Features                                                       |
| -------------- | -------------------- | ---------------------------- | ------------------------------------------------------------------ |
| **Frontend**   | `archon-ui-main/`    | Web interface and dashboard  | React, TypeScript, TailwindCSS, Socket.IO client                   |
| **Server**     | `python/src/server/` | Core business logic and APIs | FastAPI, service layer, Socket.IO broadcasts, all ML/AI operations |
| **MCP Server** | `python/src/mcp/`    | MCP protocol interface       | Lightweight HTTP wrapper, 10 MCP tools, session management         |
| **Agents**     | `python/src/agents/` | PydanticAI agent hosting     | Document and RAG agents, streaming responses                       |

### Communication Patterns

- **HTTP-based**: All inter-service communication uses HTTP APIs
- **Socket.IO**: Real-time updates from Server to Frontend
- **MCP Protocol**: AI clients connect to MCP Server via SSE or stdio
- **No Direct Imports**: Services are truly independent with no shared code dependencies

### Key Architectural Benefits

- **Lightweight Containers**: Each service contains only required dependencies
- **Independent Scaling**: Services can be scaled independently based on load
- **Development Flexibility**: Teams can work on different services without conflicts
- **Technology Diversity**: Each service uses the best tools for its specific purpose

## 🔧 Configuring Custom Ports & Hostname

By default, Archon services run on the following ports:

- **archon-ui**: 3737
- **archon-server**: 8181
- **archon-mcp**: 8051
- **archon-agents**: 8052
- **archon-docs**: 3838 (optional)

### Changing Ports

To use custom ports, add these variables to your `.env` file:

```bash
# Service Ports Configuration
ARCHON_UI_PORT=3737
ARCHON_SERVER_PORT=8181
ARCHON_MCP_PORT=8051
ARCHON_AGENTS_PORT=8052
ARCHON_DOCS_PORT=3838
```

Example: Running on different ports:

```bash
ARCHON_SERVER_PORT=8282
ARCHON_MCP_PORT=8151
```

### Configuring Hostname

By default, Archon uses `localhost` as the hostname. You can configure a custom hostname or IP address by setting the `HOST` variable in your `.env` file:

```bash
# Hostname Configuration
HOST=localhost  # Default

# Examples of custom hostnames:
HOST=192.168.1.100     # Use specific IP address
HOST=archon.local      # Use custom domain
HOST=myserver.com      # Use public domain
```

This is useful when:

- Running Archon on a different machine and accessing it remotely
- Using a custom domain name for your installation
- Deploying in a network environment where `localhost` isn't accessible

After changing hostname or ports:

1. Restart Docker containers: `docker compose down && docker compose --profile full up -d`
2. Access the UI at: `http://${HOST}:${ARCHON_UI_PORT}`
3. Update your AI client configuration with the new hostname and MCP port

## 🔧 Development

### Quick Start

```bash
# Install dependencies
make install

# Start development (recommended)
make dev        # Backend in Docker, frontend local with hot reload

# Alternative: Everything in Docker
make dev-docker # All services in Docker

# Stop everything (local FE needs to be stopped manually)
make stop
```

### Development Modes

#### Hybrid Mode (Recommended) - `make dev`

Best for active development with instant frontend updates:

- Backend services run in Docker (isolated, consistent)
- Frontend runs locally with hot module replacement
- Instant UI updates without Docker rebuilds

#### Full Docker Mode - `make dev-docker`

For all services in Docker environment:

- All services run in Docker containers
- Better for integration testing
- Slower frontend updates

### Testing & Code Quality

```bash
# Run tests
make test       # Run all tests
make test-fe    # Run frontend tests
make test-be    # Run backend tests

# Run linters
make lint       # Lint all code
make lint-fe    # Lint frontend code
make lint-be    # Lint backend code

# Check environment
make check      # Verify environment setup

# Clean up
make clean      # Remove containers and volumes (asks for confirmation)
```

### Viewing Logs

```bash
# View logs using Docker Compose directly
docker compose logs -f              # All services
docker compose logs -f archon-server # API server
docker compose logs -f archon-mcp    # MCP server
docker compose logs -f archon-ui     # Frontend
```

**Note**: The backend services are configured with `--reload` flag in their uvicorn commands and have source code mounted as volumes for automatic hot reloading when you make changes.

## 🔍 Troubleshooting

### Common Issues and Solutions

#### Port Conflicts

If you see "Port already in use" errors:

```bash
# Check what's using a port (e.g., 3737)
lsof -i :3737

# Stop all containers and local services
make stop

# Change the port in .env
```

#### Docker Permission Issues (Linux)

If you encounter permission errors with Docker:

```bash
# Add your user to the docker group
sudo usermod -aG docker $USER

# Log out and back in, or run
newgrp docker
```

#### Windows-Specific Issues

- **Make not found**: Install Make via Chocolatey, Scoop, or WSL2 (see [Installing Make](#installing-make))
- **Line ending issues**: Configure Git to use LF endings:
  ```bash
  git config --global core.autocrlf false
  ```

#### Frontend Can't Connect to Backend

- Check backend is running: `curl http://localhost:8181/health`
- Verify port configuration in `.env`
- For custom ports, ensure both `ARCHON_SERVER_PORT` and `VITE_ARCHON_SERVER_PORT` are set

#### Docker Compose Hangs

If `docker compose` commands hang:

```bash
# Reset Docker Compose
docker compose down --remove-orphans
docker system prune -f

# Restart Docker Desktop (if applicable)
```

#### Hot Reload Not Working

- **Frontend**: Ensure you're running in hybrid mode (`make dev`) for best HMR experience
- **Backend**: Check that volumes are mounted correctly in `docker-compose.yml`
- **File permissions**: On some systems, mounted volumes may have permission issues

## 📈 Progress

<p align="center">
  <a href="https://star-history.com/#coleam00/Archon&Date">
    <img src="https://api.star-history.com/svg?repos=coleam00/Archon&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📄 License

Archon Community License (ACL) v1.2 - see [LICENSE](LICENSE) file for details.

**TL;DR**: Archon is free, open, and hackable. Run it, fork it, share it - just don't sell it as-a-service without permission.
