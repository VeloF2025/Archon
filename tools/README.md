# Archon Production Tools

This directory contains production-ready tools for system administration, deployment, and maintenance.

## 📁 Directory Structure

### `/migration/`
Database and system migration tools:
- `*.sql` - Database migration scripts
- `api_migration.py` - API migration utilities
- `curl_migration.sh` - API testing migration
- Database schema updates and optimizations

### `/deployment/`
Deployment automation tools:
- Container orchestration helpers
- Environment configuration tools
- Production deployment scripts

### `/monitoring/`
System monitoring and health check tools:
- Performance monitoring utilities
- System health checks
- Metrics collection tools

## 🔧 Migration Tools

The migration directory contains critical database and API migration scripts:

### Database Migrations
- `20250830_add_metadata_column.sql` - Metadata column additions
- `add_source_url_display_name.sql` - URL display enhancements  
- `complete_setup.sql` - Complete system setup
- `RESET_DB.sql` - Database reset utility
- Performance optimization SQL scripts

### API Migrations
- `api_migration.py` - API endpoint migration utility
- `curl_migration.sh` - API testing and validation

## 🚀 Usage

### Running Migrations
```bash
# Database migrations
psql -f tools/migration/complete_setup.sql

# API migrations  
python tools/migration/api_migration.py
bash tools/migration/curl_migration.sh
```

### Production Deployment
```bash
# Use deployment tools for production setup
tools/deployment/deploy.sh
```

## ⚠️ Production Safety

**Critical**: These are production tools that directly affect system state:
- Always backup before running migrations
- Test in development environment first
- Review SQL scripts before execution
- Monitor system after applying changes

## 📋 Best Practices

- Run migrations in maintenance windows
- Keep migration logs for troubleshooting
- Test rollback procedures
- Validate system integrity after changes

---
*Production Tools: September 2025*  
*Safety Level: Production Critical*