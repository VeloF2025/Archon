# 📦 Installation Guide

This guide provides detailed installation instructions for Archon across different platforms and deployment scenarios.

## 📋 Prerequisites

### System Requirements

- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+ recommended)
- **Memory**: Minimum 8GB RAM (16GB recommended for full agent system)
- **Storage**: 10GB free disk space
- **Network**: Internet connection for external APIs and package downloads

### Required Software

#### Essential Dependencies

1. **Docker Desktop** (Required)
   - [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)
   - Ensure Docker Compose is included (v2.0+)

2. **Node.js** (Required for hybrid development)
   - Version 18+ required
   - [Download from nodejs.org](https://nodejs.org/)

3. **Supabase Account** (Required)
   - Free tier available at [supabase.com](https://supabase.com/)
   - Or set up local Supabase instance

#### Optional Tools

4. **Make** (Optional but recommended)
   - See [Installing Make](#installing-make) section below
   - Enables simplified development commands

5. **Git** (For cloning repository)
   - [Download Git](https://git-scm.com/)

## 🚀 Quick Installation (Recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/coleam00/archon.git
cd archon
```

### Step 2: Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your settings
# Required: SUPABASE_URL and SUPABASE_SERVICE_KEY
```

### Step 3: Configure Supabase

1. Create a new project at [supabase.com](https://supabase.com/)
2. Get your project URL and service key from Settings → API
3. Add to your `.env` file:

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key-here
```

⚠️ **Important**: Use the **service_role** key, not the anon key!

### Step 4: Database Setup

1. Open your Supabase project's SQL Editor
2. Copy and paste the contents of `migration/complete_setup.sql`
3. Execute the SQL to create all required tables

### Step 5: Start Archon

```bash
# Using Make (recommended)
make dev-docker

# Or using Docker Compose directly
docker compose --profile full up --build -d
```

### Step 6: Configure API Keys

1. Open http://localhost:3737
2. Go to **Settings** → **API Keys**
3. Add your OpenAI API key (or other provider)
4. Test by uploading a document or crawling a website

🎉 **You're ready to go!** Archon is now running at http://localhost:3737

## 📦 Detailed Installation Options

### Option 1: Full Docker Mode (Production-like)

Best for: Production deployments, consistent environments

```bash
# Start all services in Docker
docker compose --profile full up --build -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

**Services Started:**
- Frontend UI (Port 3737)
- API Server (Port 8181)
- MCP Server (Port 8051)
- Agents Service (Port 8052)

### Option 2: Hybrid Development Mode

Best for: Active development with fast frontend updates

```bash
# Using Make
make dev

# Or manually
docker compose --profile backend up -d --build
cd archon-ui-main && npm run dev
```

**Services Started:**
- Backend services in Docker
- Frontend runs locally with hot reload

### Option 3: Local Development

Best for: Full control over all services

```bash
# Install dependencies
make install

# Start backend services
docker compose --profile backend up -d --build

# Start frontend locally
cd archon-ui-main
npm run dev

# In another terminal, start additional services as needed
```

## 🔧 Platform-Specific Instructions

### Windows

#### Using WSL2 (Recommended)

1. Install WSL2 and Docker Desktop
2. Enable WSL2 integration in Docker Desktop settings
3. Clone and run Archon inside WSL2

```bash
# In WSL2 terminal
git clone https://github.com/coleam00/archon.git
cd archon
make dev-docker
```

#### Native Windows

```bash
# Using PowerShell or Command Prompt
git clone https://github.com/coleam00/archon.git
cd archon
docker compose --profile full up --build -d
```

### macOS

```bash
# Install dependencies using Homebrew (optional)
brew install make git node

# Clone and start
git clone https://github.com/coleam00/archon.git
cd archon
make dev-docker
```

### Linux

#### Ubuntu/Debian

```bash
# Install dependencies
sudo apt update
sudo apt install make git nodejs npm

# Add user to docker group (if needed)
sudo usermod -aG docker $USER
newgrp docker

# Clone and start
git clone https://github.com/coleam00/archon.git
cd archon
make dev-docker
```

#### CentOS/RHEL/Fedora

```bash
# Install dependencies
sudo yum install make git nodejs npm

# Or for newer versions
sudo dnf install make git nodejs npm

# Clone and start
git clone https://github.com/coleam00/archon.git
cd archon
make dev-docker
```

## 🛠️ Installing Make

Make is optional but recommended for simplified commands.

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
# Make comes pre-installed
# If needed: brew install make
```

### Linux

```bash
# Debian/Ubuntu
sudo apt-get install make

# RHEL/CentOS/Fedora
sudo yum install make
# Or: sudo dnf install make
```

## 🏭 Production Deployment

### Docker Compose Production

```bash
# Copy production environment
cp .env.example .env.production

# Edit production settings
vim .env.production

# Start with production configuration
docker compose -f docker-compose.yml -f docker-compose.production.yml up -d
```

### Environment Variables for Production

```bash
# .env.production
PROD=true
HOST=your-domain.com
ARCHON_UI_PORT=80
ARCHON_SERVER_PORT=8181

# Security settings
SUPABASE_URL=https://your-prod-project.supabase.co
SUPABASE_SERVICE_KEY=your-prod-service-key

# Optional: Reverse proxy settings
VITE_ALLOWED_HOSTS=your-domain.com,www.your-domain.com
```

### Reverse Proxy Configuration

#### Nginx

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:3737;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
```

#### Apache

```apache
<VirtualHost *:80>
    ServerName your-domain.com
    ProxyPreserveHost On
    ProxyRequests Off
    ProxyPass / http://localhost:3737/
    ProxyPassReverse / http://localhost:3737/
</VirtualHost>
```

### Cloud Deployment

See our [deployment guides](docs/deployment/) for specific cloud providers:

- [AWS Deployment](docs/deployment/aws.md)
- [Google Cloud Deployment](docs/deployment/gcp.md)
- [Azure Deployment](docs/deployment/azure.md)
- [DigitalOcean Deployment](docs/deployment/digitalocean.md)

## 🔍 Verification and Testing

### Health Check

```bash
# Check all services are running
curl http://localhost:3737/health
curl http://localhost:8181/health
curl http://localhost:8051/health

# Or using Make
make check
```

### Test Basic Functionality

1. **Web Interface**: Open http://localhost:3737
2. **Knowledge Base**: Try crawling a documentation site
3. **MCP Integration**: Test connection with an AI client
4. **Project Management**: Create a test project

### Performance Test

```bash
# Run built-in performance tests
make test

# Load test (if available)
npm run test:load
```

## 🔧 Customization

### Port Configuration

Edit `.env` to change default ports:

```bash
ARCHON_UI_PORT=3737
ARCHON_SERVER_PORT=8181
ARCHON_MCP_PORT=8051
ARCHON_AGENTS_PORT=8052
```

### Service Selection

Choose which services to run:

```bash
# Minimal setup (no agents)
docker compose up -d

# Full setup with agents
docker compose --profile full up -d

# Custom service selection
docker compose up archon-server archon-ui -d
```

### Resource Limits

For resource-constrained environments, modify `docker-compose.yml`:

```yaml
services:
  archon-server:
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.5'
```

## 📚 Next Steps

After successful installation:

1. **[Configuration Guide](CONFIGURATION.md)** - Configure API keys and settings
2. **[Architecture Overview](ARCHITECTURE.md)** - Understand how Archon works
3. **[User Guide](docs/user-guide/)** - Learn how to use all features
4. **[Development Setup](CONTRIBUTING.md)** - Set up for development

## 🆘 Troubleshooting

For installation issues, see our [Troubleshooting Guide](TROUBLESHOOTING.md) or visit:

- [GitHub Issues](https://github.com/coleam00/archon/issues)
- [GitHub Discussions](https://github.com/coleam00/archon/discussions)
- [Discord Community](https://discord.gg/archon) (if available)

---

**Need help?** Join our community or open an issue on GitHub!