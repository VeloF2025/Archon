#!/usr/bin/env python3
"""
Database Migration Script

Creates the workflow management tables in the database.
This script should be run to set up the initial database schema.
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from database.connection import DATABASE_URL, Base
from database.workflow_models import (
    WorkflowDefinition, WorkflowVersion, WorkflowStep, WorkflowExecution,
    StepExecution, WorkflowMetrics, WorkflowAnalytics, WorkflowSchedule,
    WebhookTrigger, EventTrigger
)

def create_workflow_tables():
    """Create all workflow management tables"""
    print(f"Connecting to database: {DATABASE_URL}")

    # Create engine
    engine = create_engine(DATABASE_URL)

    print("Creating workflow management tables...")

    # Create all tables
    Base.metadata.create_all(bind=engine)

    print("Workflow tables created successfully!")

    # Verify tables were created
    with engine.connect() as conn:
        result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        tables = [row[0] for row in result.fetchall()]

        workflow_tables = [
            'workflow_definitions', 'workflow_versions', 'workflow_steps',
            'workflow_executions', 'step_executions', 'workflow_metrics',
            'workflow_analytics', 'workflow_schedules', 'webhook_triggers',
            'event_triggers'
        ]

        print("\nCreated tables:")
        for table in workflow_tables:
            if table in tables:
                print(f"  [OK] {table}")
            else:
                print(f"  [FAIL] {table} (missing)")

    print("\nDatabase migration completed successfully!")

if __name__ == "__main__":
    create_workflow_tables()