"""
Workflow Templates Module
Pre-built workflow templates and template management
"""

import json
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from .workflow_engine import Workflow, WorkflowStep, WorkflowTrigger, StepType, TriggerType
from .automation_rules import Rule, RuleType, Condition, ConditionOperator, Action, ActionType


class TemplateCategory(Enum):
    """Template categories"""
    DEVELOPMENT = "development"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    TESTING = "testing"
    SECURITY = "security"
    DATA_PROCESSING = "data_processing"
    INTEGRATION = "integration"
    NOTIFICATION = "notification"
    APPROVAL = "approval"
    MAINTENANCE = "maintenance"


@dataclass
class WorkflowTemplate:
    """Workflow template definition"""
    template_id: str
    name: str
    description: str
    category: TemplateCategory
    version: str = "1.0.0"
    author: str = "system"
    tags: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)  # Template parameters
    variables: Dict[str, Any] = field(default_factory=dict)  # Default variables
    workflow_definition: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    rating: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TemplateLibrary:
    """
    Library of workflow templates
    """
    
    def __init__(self):
        self.templates: Dict[str, WorkflowTemplate] = {}
        self.categories: Dict[TemplateCategory, List[str]] = {}
        
        # Initialize default templates
        self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """Initialize default workflow templates"""
        
        # CI/CD Pipeline Template
        self.add_template(self._create_cicd_template())
        
        # Code Review Template
        self.add_template(self._create_code_review_template())
        
        # Bug Triage Template
        self.add_template(self._create_bug_triage_template())
        
        # Data Pipeline Template
        self.add_template(self._create_data_pipeline_template())
        
        # Incident Response Template
        self.add_template(self._create_incident_response_template())
        
        # Release Management Template
        self.add_template(self._create_release_management_template())
        
        # Security Scan Template
        self.add_template(self._create_security_scan_template())
        
        # Backup and Recovery Template
        self.add_template(self._create_backup_recovery_template())
        
        # User Onboarding Template
        self.add_template(self._create_user_onboarding_template())
        
        # Performance Monitoring Template
        self.add_template(self._create_performance_monitoring_template())
    
    def _create_cicd_template(self) -> WorkflowTemplate:
        """Create CI/CD pipeline template"""
        workflow_def = {
            "name": "CI/CD Pipeline",
            "description": "Continuous Integration and Deployment Pipeline",
            "triggers": [
                {
                    "type": "webhook",
                    "config": {
                        "event": "push",
                        "branch": "{{branch_name}}"
                    }
                }
            ],
            "steps": [
                {
                    "id": "checkout",
                    "name": "Checkout Code",
                    "type": "action",
                    "action": "git_checkout",
                    "parameters": {
                        "repository": "{{repository_url}}",
                        "branch": "{{branch_name}}"
                    }
                },
                {
                    "id": "install_deps",
                    "name": "Install Dependencies",
                    "type": "action",
                    "action": "run_command",
                    "parameters": {
                        "command": "{{install_command}}",
                        "working_dir": "{{project_dir}}"
                    },
                    "next_steps": ["lint"]
                },
                {
                    "id": "lint",
                    "name": "Run Linting",
                    "type": "action",
                    "action": "run_command",
                    "parameters": {
                        "command": "{{lint_command}}"
                    },
                    "next_steps": ["test"]
                },
                {
                    "id": "test",
                    "name": "Run Tests",
                    "type": "action",
                    "action": "run_command",
                    "parameters": {
                        "command": "{{test_command}}"
                    },
                    "next_steps": ["build"]
                },
                {
                    "id": "build",
                    "name": "Build Application",
                    "type": "action",
                    "action": "run_command",
                    "parameters": {
                        "command": "{{build_command}}"
                    },
                    "next_steps": ["security_scan"]
                },
                {
                    "id": "security_scan",
                    "name": "Security Scan",
                    "type": "action",
                    "action": "security_scan",
                    "parameters": {
                        "scan_type": "dependency",
                        "fail_on_critical": true
                    },
                    "next_steps": ["deploy_decision"]
                },
                {
                    "id": "deploy_decision",
                    "name": "Deploy Decision",
                    "type": "decision",
                    "conditions": [
                        {
                            "field": "branch_name",
                            "operator": "equals",
                            "value": "main",
                            "next_step": "deploy_prod"
                        },
                        {
                            "field": "branch_name",
                            "operator": "equals",
                            "value": "develop",
                            "next_step": "deploy_staging"
                        }
                    ]
                },
                {
                    "id": "deploy_staging",
                    "name": "Deploy to Staging",
                    "type": "action",
                    "action": "deploy",
                    "parameters": {
                        "environment": "staging",
                        "strategy": "rolling"
                    },
                    "next_steps": ["notify"]
                },
                {
                    "id": "deploy_prod",
                    "name": "Deploy to Production",
                    "type": "approval",
                    "parameters": {
                        "approvers": "{{prod_approvers}}",
                        "timeout": 3600
                    },
                    "next_steps": ["deploy_prod_action"]
                },
                {
                    "id": "deploy_prod_action",
                    "name": "Deploy to Production",
                    "type": "action",
                    "action": "deploy",
                    "parameters": {
                        "environment": "production",
                        "strategy": "blue_green"
                    },
                    "next_steps": ["notify"]
                },
                {
                    "id": "notify",
                    "name": "Send Notifications",
                    "type": "notification",
                    "parameters": {
                        "channels": ["slack", "email"],
                        "message": "Deployment completed for {{branch_name}}"
                    }
                }
            ]
        }
        
        return WorkflowTemplate(
            template_id=str(uuid.uuid4()),
            name="CI/CD Pipeline",
            description="Complete CI/CD pipeline with testing, security scanning, and deployment",
            category=TemplateCategory.DEPLOYMENT,
            tags=["ci", "cd", "deployment", "automation"],
            parameters={
                "repository_url": {"type": "string", "required": True},
                "branch_name": {"type": "string", "default": "main"},
                "install_command": {"type": "string", "default": "npm install"},
                "lint_command": {"type": "string", "default": "npm run lint"},
                "test_command": {"type": "string", "default": "npm test"},
                "build_command": {"type": "string", "default": "npm run build"},
                "prod_approvers": {"type": "array", "default": ["team-lead", "devops"]}
            },
            workflow_definition=workflow_def
        )
    
    def _create_code_review_template(self) -> WorkflowTemplate:
        """Create code review workflow template"""
        workflow_def = {
            "name": "Code Review Process",
            "description": "Automated code review workflow",
            "triggers": [
                {
                    "type": "event",
                    "config": {
                        "event": "pull_request",
                        "action": "opened"
                    }
                }
            ],
            "steps": [
                {
                    "id": "analyze_changes",
                    "name": "Analyze Code Changes",
                    "type": "action",
                    "action": "analyze_diff",
                    "parameters": {
                        "pr_number": "{{pr_number}}",
                        "repository": "{{repository}}"
                    }
                },
                {
                    "id": "run_static_analysis",
                    "name": "Static Code Analysis",
                    "type": "parallel",
                    "parameters": {
                        "steps": [
                            {
                                "action": "run_sonarqube",
                                "parameters": {"project": "{{project_key}}"}
                            },
                            {
                                "action": "run_eslint",
                                "parameters": {"config": "{{eslint_config}}"}
                            },
                            {
                                "action": "check_complexity",
                                "parameters": {"threshold": 10}
                            }
                        ]
                    },
                    "next_steps": ["check_test_coverage"]
                },
                {
                    "id": "check_test_coverage",
                    "name": "Check Test Coverage",
                    "type": "action",
                    "action": "coverage_check",
                    "parameters": {
                        "minimum_coverage": "{{min_coverage}}",
                        "fail_on_decrease": true
                    },
                    "next_steps": ["ai_review"]
                },
                {
                    "id": "ai_review",
                    "name": "AI Code Review",
                    "type": "action",
                    "action": "ai_review",
                    "parameters": {
                        "model": "code-review-gpt",
                        "focus_areas": ["security", "performance", "best_practices"]
                    },
                    "next_steps": ["assign_reviewers"]
                },
                {
                    "id": "assign_reviewers",
                    "name": "Assign Reviewers",
                    "type": "action",
                    "action": "assign_reviewers",
                    "parameters": {
                        "strategy": "code_owners",
                        "min_reviewers": 2
                    },
                    "next_steps": ["wait_for_review"]
                },
                {
                    "id": "wait_for_review",
                    "name": "Wait for Reviews",
                    "type": "wait",
                    "parameters": {
                        "condition": "reviews_complete",
                        "timeout": 86400
                    },
                    "next_steps": ["review_decision"]
                },
                {
                    "id": "review_decision",
                    "name": "Review Decision",
                    "type": "decision",
                    "conditions": [
                        {
                            "field": "approved_reviews",
                            "operator": "greater_than_or_equal",
                            "value": 2,
                            "next_step": "merge"
                        },
                        {
                            "field": "changes_requested",
                            "operator": "greater_than",
                            "value": 0,
                            "next_step": "request_changes"
                        }
                    ]
                },
                {
                    "id": "merge",
                    "name": "Merge Pull Request",
                    "type": "action",
                    "action": "merge_pr",
                    "parameters": {
                        "strategy": "squash",
                        "delete_branch": true
                    }
                },
                {
                    "id": "request_changes",
                    "name": "Request Changes",
                    "type": "notification",
                    "parameters": {
                        "recipient": "{{pr_author}}",
                        "message": "Changes requested on PR #{{pr_number}}"
                    }
                }
            ]
        }
        
        return WorkflowTemplate(
            template_id=str(uuid.uuid4()),
            name="Code Review Process",
            description="Comprehensive code review workflow with static analysis and AI review",
            category=TemplateCategory.DEVELOPMENT,
            tags=["code-review", "quality", "pr", "automation"],
            parameters={
                "min_coverage": {"type": "number", "default": 80},
                "project_key": {"type": "string", "required": True},
                "eslint_config": {"type": "string", "default": ".eslintrc.json"}
            },
            workflow_definition=workflow_def
        )
    
    def _create_bug_triage_template(self) -> WorkflowTemplate:
        """Create bug triage workflow template"""
        workflow_def = {
            "name": "Bug Triage Process",
            "description": "Automated bug triage and assignment",
            "triggers": [
                {
                    "type": "event",
                    "config": {
                        "event": "issue_created",
                        "labels": ["bug"]
                    }
                }
            ],
            "steps": [
                {
                    "id": "analyze_bug",
                    "name": "Analyze Bug Report",
                    "type": "action",
                    "action": "analyze_issue",
                    "parameters": {
                        "issue_id": "{{issue_id}}",
                        "extract": ["severity", "component", "stack_trace"]
                    }
                },
                {
                    "id": "check_duplicate",
                    "name": "Check for Duplicates",
                    "type": "action",
                    "action": "find_similar_issues",
                    "parameters": {
                        "threshold": 0.8,
                        "status": ["open", "in_progress"]
                    },
                    "next_steps": ["duplicate_decision"]
                },
                {
                    "id": "duplicate_decision",
                    "name": "Duplicate Decision",
                    "type": "decision",
                    "conditions": [
                        {
                            "field": "duplicate_found",
                            "operator": "is_true",
                            "next_step": "mark_duplicate"
                        },
                        {
                            "field": "duplicate_found",
                            "operator": "is_false",
                            "next_step": "determine_severity"
                        }
                    ]
                },
                {
                    "id": "mark_duplicate",
                    "name": "Mark as Duplicate",
                    "type": "action",
                    "action": "close_issue",
                    "parameters": {
                        "reason": "duplicate",
                        "link_to": "{{duplicate_issue_id}}"
                    }
                },
                {
                    "id": "determine_severity",
                    "name": "Determine Severity",
                    "type": "action",
                    "action": "calculate_severity",
                    "parameters": {
                        "factors": ["user_impact", "frequency", "workaround_available"]
                    },
                    "next_steps": ["assign_priority"]
                },
                {
                    "id": "assign_priority",
                    "name": "Assign Priority",
                    "type": "action",
                    "action": "set_labels",
                    "parameters": {
                        "labels": ["priority:{{calculated_priority}}"]
                    },
                    "next_steps": ["assign_team"]
                },
                {
                    "id": "assign_team",
                    "name": "Assign to Team",
                    "type": "action",
                    "action": "assign_team",
                    "parameters": {
                        "strategy": "component_owner",
                        "component": "{{detected_component}}"
                    },
                    "next_steps": ["notify_team"]
                },
                {
                    "id": "notify_team",
                    "name": "Notify Team",
                    "type": "notification",
                    "parameters": {
                        "channel": "slack",
                        "message": "New {{calculated_priority}} priority bug assigned: {{issue_title}}"
                    }
                }
            ]
        }
        
        return WorkflowTemplate(
            template_id=str(uuid.uuid4()),
            name="Bug Triage Process",
            description="Automated bug triage with duplicate detection and smart assignment",
            category=TemplateCategory.DEVELOPMENT,
            tags=["bug", "triage", "issue-management"],
            workflow_definition=workflow_def
        )
    
    def _create_data_pipeline_template(self) -> WorkflowTemplate:
        """Create data pipeline workflow template"""
        workflow_def = {
            "name": "Data Processing Pipeline",
            "description": "ETL data processing pipeline",
            "triggers": [
                {
                    "type": "scheduled",
                    "config": {
                        "cron": "0 2 * * *"  # Daily at 2 AM
                    }
                }
            ],
            "steps": [
                {
                    "id": "extract_data",
                    "name": "Extract Data",
                    "type": "parallel",
                    "parameters": {
                        "steps": [
                            {
                                "action": "extract_database",
                                "parameters": {"source": "{{source_db}}"}
                            },
                            {
                                "action": "extract_api",
                                "parameters": {"endpoint": "{{api_endpoint}}"}
                            },
                            {
                                "action": "extract_files",
                                "parameters": {"path": "{{file_path}}"}
                            }
                        ]
                    },
                    "next_steps": ["validate_data"]
                },
                {
                    "id": "validate_data",
                    "name": "Validate Data",
                    "type": "action",
                    "action": "validate_schema",
                    "parameters": {
                        "schema": "{{data_schema}}",
                        "fail_on_error": false
                    },
                    "next_steps": ["transform_data"]
                },
                {
                    "id": "transform_data",
                    "name": "Transform Data",
                    "type": "action",
                    "action": "run_transformation",
                    "parameters": {
                        "transformations": "{{transformation_rules}}",
                        "parallel": true
                    },
                    "next_steps": ["quality_check"]
                },
                {
                    "id": "quality_check",
                    "name": "Data Quality Check",
                    "type": "action",
                    "action": "quality_check",
                    "parameters": {
                        "checks": ["completeness", "uniqueness", "consistency"],
                        "threshold": 0.95
                    },
                    "next_steps": ["load_data"]
                },
                {
                    "id": "load_data",
                    "name": "Load Data",
                    "type": "action",
                    "action": "load_to_warehouse",
                    "parameters": {
                        "destination": "{{warehouse_table}}",
                        "mode": "append",
                        "partition": "{{partition_key}}"
                    },
                    "next_steps": ["update_metadata"]
                },
                {
                    "id": "update_metadata",
                    "name": "Update Metadata",
                    "type": "action",
                    "action": "update_catalog",
                    "parameters": {
                        "catalog": "data_catalog",
                        "metrics": ["row_count", "processing_time", "quality_score"]
                    },
                    "next_steps": ["notify_completion"]
                },
                {
                    "id": "notify_completion",
                    "name": "Notify Completion",
                    "type": "notification",
                    "parameters": {
                        "recipients": "{{data_team}}",
                        "message": "Data pipeline completed successfully"
                    }
                }
            ]
        }
        
        return WorkflowTemplate(
            template_id=str(uuid.uuid4()),
            name="Data Processing Pipeline",
            description="Complete ETL pipeline with validation and quality checks",
            category=TemplateCategory.DATA_PROCESSING,
            tags=["etl", "data", "pipeline", "warehouse"],
            parameters={
                "source_db": {"type": "string", "required": True},
                "api_endpoint": {"type": "string"},
                "file_path": {"type": "string"},
                "warehouse_table": {"type": "string", "required": True},
                "partition_key": {"type": "string", "default": "date"}
            },
            workflow_definition=workflow_def
        )
    
    def _create_incident_response_template(self) -> WorkflowTemplate:
        """Create incident response workflow template"""
        workflow_def = {
            "name": "Incident Response",
            "description": "Automated incident response and escalation",
            "triggers": [
                {
                    "type": "event",
                    "config": {
                        "event": "alert_triggered",
                        "severity": ["critical", "high"]
                    }
                }
            ],
            "steps": [
                {
                    "id": "create_incident",
                    "name": "Create Incident",
                    "type": "action",
                    "action": "create_incident",
                    "parameters": {
                        "title": "{{alert_title}}",
                        "severity": "{{alert_severity}}",
                        "description": "{{alert_description}}"
                    }
                },
                {
                    "id": "page_oncall",
                    "name": "Page On-Call",
                    "type": "action",
                    "action": "page_oncall",
                    "parameters": {
                        "team": "{{affected_team}}",
                        "escalation_policy": "{{escalation_policy}}"
                    },
                    "next_steps": ["gather_diagnostics"]
                },
                {
                    "id": "gather_diagnostics",
                    "name": "Gather Diagnostics",
                    "type": "parallel",
                    "parameters": {
                        "steps": [
                            {
                                "action": "collect_logs",
                                "parameters": {"time_range": "1h"}
                            },
                            {
                                "action": "collect_metrics",
                                "parameters": {"services": "{{affected_services}}"}
                            },
                            {
                                "action": "run_diagnostics",
                                "parameters": {"scripts": "{{diagnostic_scripts}}"}
                            }
                        ]
                    },
                    "next_steps": ["auto_remediation"]
                },
                {
                    "id": "auto_remediation",
                    "name": "Attempt Auto-Remediation",
                    "type": "action",
                    "action": "run_playbook",
                    "parameters": {
                        "playbook": "{{remediation_playbook}}",
                        "dry_run": false
                    },
                    "next_steps": ["check_resolution"]
                },
                {
                    "id": "check_resolution",
                    "name": "Check Resolution",
                    "type": "wait",
                    "parameters": {
                        "duration": 300,
                        "check_condition": "alert_resolved"
                    },
                    "next_steps": ["resolution_decision"]
                },
                {
                    "id": "resolution_decision",
                    "name": "Resolution Decision",
                    "type": "decision",
                    "conditions": [
                        {
                            "field": "incident_resolved",
                            "operator": "is_true",
                            "next_step": "close_incident"
                        },
                        {
                            "field": "incident_resolved",
                            "operator": "is_false",
                            "next_step": "escalate"
                        }
                    ]
                },
                {
                    "id": "escalate",
                    "name": "Escalate Incident",
                    "type": "action",
                    "action": "escalate",
                    "parameters": {
                        "level": "next",
                        "notify_management": true
                    },
                    "next_steps": ["create_war_room"]
                },
                {
                    "id": "create_war_room",
                    "name": "Create War Room",
                    "type": "action",
                    "action": "create_channel",
                    "parameters": {
                        "name": "incident-{{incident_id}}",
                        "participants": "{{incident_team}}"
                    }
                },
                {
                    "id": "close_incident",
                    "name": "Close Incident",
                    "type": "action",
                    "action": "close_incident",
                    "parameters": {
                        "resolution": "{{resolution_notes}}",
                        "post_mortem_required": true
                    }
                }
            ]
        }
        
        return WorkflowTemplate(
            template_id=str(uuid.uuid4()),
            name="Incident Response",
            description="Complete incident response with auto-remediation and escalation",
            category=TemplateCategory.MONITORING,
            tags=["incident", "response", "escalation", "ops"],
            parameters={
                "escalation_policy": {"type": "string", "default": "standard"},
                "remediation_playbook": {"type": "string", "required": True},
                "diagnostic_scripts": {"type": "array", "default": []}
            },
            workflow_definition=workflow_def
        )
    
    def _create_release_management_template(self) -> WorkflowTemplate:
        """Create release management workflow template"""
        workflow_def = {
            "name": "Release Management",
            "description": "End-to-end release management process",
            "triggers": [
                {
                    "type": "manual",
                    "config": {
                        "allowed_users": "{{release_managers}}"
                    }
                }
            ],
            "steps": [
                {
                    "id": "create_release_branch",
                    "name": "Create Release Branch",
                    "type": "action",
                    "action": "create_branch",
                    "parameters": {
                        "name": "release/{{version}}",
                        "from": "develop"
                    }
                },
                {
                    "id": "update_version",
                    "name": "Update Version",
                    "type": "action",
                    "action": "update_version",
                    "parameters": {
                        "version": "{{version}}",
                        "files": ["package.json", "VERSION"]
                    },
                    "next_steps": ["generate_changelog"]
                },
                {
                    "id": "generate_changelog",
                    "name": "Generate Changelog",
                    "type": "action",
                    "action": "generate_changelog",
                    "parameters": {
                        "from_tag": "{{previous_version}}",
                        "to_branch": "release/{{version}}"
                    },
                    "next_steps": ["run_release_tests"]
                },
                {
                    "id": "run_release_tests",
                    "name": "Run Release Tests",
                    "type": "parallel",
                    "parameters": {
                        "steps": [
                            {"action": "run_unit_tests"},
                            {"action": "run_integration_tests"},
                            {"action": "run_e2e_tests"},
                            {"action": "run_performance_tests"}
                        ]
                    },
                    "next_steps": ["approval_gate"]
                },
                {
                    "id": "approval_gate",
                    "name": "Release Approval",
                    "type": "approval",
                    "parameters": {
                        "approvers": "{{release_approvers}}",
                        "min_approvals": 2,
                        "timeout": 86400
                    },
                    "next_steps": ["tag_release"]
                },
                {
                    "id": "tag_release",
                    "name": "Tag Release",
                    "type": "action",
                    "action": "create_tag",
                    "parameters": {
                        "tag": "v{{version}}",
                        "message": "Release version {{version}}"
                    },
                    "next_steps": ["deploy_production"]
                },
                {
                    "id": "deploy_production",
                    "name": "Deploy to Production",
                    "type": "action",
                    "action": "deploy",
                    "parameters": {
                        "environment": "production",
                        "version": "{{version}}",
                        "strategy": "canary",
                        "canary_percentage": 10
                    },
                    "next_steps": ["monitor_deployment"]
                },
                {
                    "id": "monitor_deployment",
                    "name": "Monitor Deployment",
                    "type": "wait",
                    "parameters": {
                        "duration": 1800,
                        "monitor": ["error_rate", "response_time", "cpu_usage"]
                    },
                    "next_steps": ["deployment_decision"]
                },
                {
                    "id": "deployment_decision",
                    "name": "Deployment Decision",
                    "type": "decision",
                    "conditions": [
                        {
                            "field": "deployment_healthy",
                            "operator": "is_true",
                            "next_step": "complete_rollout"
                        },
                        {
                            "field": "deployment_healthy",
                            "operator": "is_false",
                            "next_step": "rollback"
                        }
                    ]
                },
                {
                    "id": "complete_rollout",
                    "name": "Complete Rollout",
                    "type": "action",
                    "action": "update_canary",
                    "parameters": {
                        "percentage": 100
                    },
                    "next_steps": ["publish_release"]
                },
                {
                    "id": "rollback",
                    "name": "Rollback Deployment",
                    "type": "action",
                    "action": "rollback",
                    "parameters": {
                        "to_version": "{{previous_version}}"
                    }
                },
                {
                    "id": "publish_release",
                    "name": "Publish Release",
                    "type": "action",
                    "action": "publish_release",
                    "parameters": {
                        "version": "{{version}}",
                        "changelog": "{{changelog}}",
                        "platforms": ["github", "npm", "docker"]
                    }
                }
            ]
        }
        
        return WorkflowTemplate(
            template_id=str(uuid.uuid4()),
            name="Release Management",
            description="Complete release process with canary deployment and rollback",
            category=TemplateCategory.DEPLOYMENT,
            tags=["release", "deployment", "canary", "rollback"],
            parameters={
                "version": {"type": "string", "required": True},
                "previous_version": {"type": "string", "required": True},
                "release_managers": {"type": "array", "default": ["release-team"]},
                "release_approvers": {"type": "array", "default": ["product", "engineering"]}
            },
            workflow_definition=workflow_def
        )
    
    def _create_security_scan_template(self) -> WorkflowTemplate:
        """Create security scanning workflow template"""
        workflow_def = {
            "name": "Security Scan",
            "description": "Comprehensive security scanning workflow",
            "triggers": [
                {
                    "type": "scheduled",
                    "config": {
                        "cron": "0 0 * * 0"  # Weekly on Sunday
                    }
                }
            ],
            "steps": [
                {
                    "id": "dependency_scan",
                    "name": "Dependency Scan",
                    "type": "action",
                    "action": "scan_dependencies",
                    "parameters": {
                        "tools": ["npm-audit", "snyk", "owasp-dependency-check"],
                        "fail_on": "high"
                    }
                },
                {
                    "id": "sast_scan",
                    "name": "Static Application Security Testing",
                    "type": "action",
                    "action": "run_sast",
                    "parameters": {
                        "tools": ["sonarqube", "checkmarx"],
                        "languages": "{{project_languages}}"
                    },
                    "next_steps": ["container_scan"]
                },
                {
                    "id": "container_scan",
                    "name": "Container Scan",
                    "type": "action",
                    "action": "scan_containers",
                    "parameters": {
                        "registries": "{{container_registries}}",
                        "tools": ["trivy", "clair"]
                    },
                    "next_steps": ["secrets_scan"]
                },
                {
                    "id": "secrets_scan",
                    "name": "Secrets Scan",
                    "type": "action",
                    "action": "scan_secrets",
                    "parameters": {
                        "paths": [".", ".git"],
                        "tools": ["trufflehog", "gitleaks"]
                    },
                    "next_steps": ["generate_report"]
                },
                {
                    "id": "generate_report",
                    "name": "Generate Security Report",
                    "type": "action",
                    "action": "generate_report",
                    "parameters": {
                        "format": "html",
                        "include": ["vulnerabilities", "recommendations", "trends"]
                    },
                    "next_steps": ["create_issues"]
                },
                {
                    "id": "create_issues",
                    "name": "Create Security Issues",
                    "type": "action",
                    "action": "create_issues",
                    "parameters": {
                        "severity_threshold": "medium",
                        "auto_assign": true
                    }
                }
            ]
        }
        
        return WorkflowTemplate(
            template_id=str(uuid.uuid4()),
            name="Security Scan",
            description="Comprehensive security scanning with multiple tools",
            category=TemplateCategory.SECURITY,
            tags=["security", "scanning", "vulnerability"],
            parameters={
                "project_languages": {"type": "array", "default": ["javascript", "python"]},
                "container_registries": {"type": "array", "default": []}
            },
            workflow_definition=workflow_def
        )
    
    def _create_backup_recovery_template(self) -> WorkflowTemplate:
        """Create backup and recovery workflow template"""
        workflow_def = {
            "name": "Backup and Recovery",
            "description": "Automated backup and disaster recovery",
            "triggers": [
                {
                    "type": "scheduled",
                    "config": {
                        "cron": "0 3 * * *"  # Daily at 3 AM
                    }
                }
            ],
            "steps": [
                {
                    "id": "snapshot_databases",
                    "name": "Snapshot Databases",
                    "type": "parallel",
                    "parameters": {
                        "databases": "{{database_list}}",
                        "snapshot_type": "incremental"
                    }
                },
                {
                    "id": "backup_files",
                    "name": "Backup Files",
                    "type": "action",
                    "action": "backup_files",
                    "parameters": {
                        "paths": "{{backup_paths}}",
                        "compression": "gzip",
                        "encryption": true
                    },
                    "next_steps": ["verify_backups"]
                },
                {
                    "id": "verify_backups",
                    "name": "Verify Backups",
                    "type": "action",
                    "action": "verify_integrity",
                    "parameters": {
                        "checksum": true,
                        "test_restore": true
                    },
                    "next_steps": ["replicate_offsite"]
                },
                {
                    "id": "replicate_offsite",
                    "name": "Replicate to Offsite",
                    "type": "action",
                    "action": "replicate",
                    "parameters": {
                        "destination": "{{offsite_location}}",
                        "retention_days": 30
                    },
                    "next_steps": ["cleanup_old"]
                },
                {
                    "id": "cleanup_old",
                    "name": "Cleanup Old Backups",
                    "type": "action",
                    "action": "cleanup_backups",
                    "parameters": {
                        "retention_policy": "{{retention_policy}}",
                        "keep_monthly": 12
                    }
                }
            ]
        }
        
        return WorkflowTemplate(
            template_id=str(uuid.uuid4()),
            name="Backup and Recovery",
            description="Automated backup with verification and offsite replication",
            category=TemplateCategory.MAINTENANCE,
            tags=["backup", "recovery", "disaster-recovery"],
            parameters={
                "database_list": {"type": "array", "required": True},
                "backup_paths": {"type": "array", "required": True},
                "offsite_location": {"type": "string", "required": True},
                "retention_policy": {"type": "object", "default": {"daily": 7, "weekly": 4}}
            },
            workflow_definition=workflow_def
        )
    
    def _create_user_onboarding_template(self) -> WorkflowTemplate:
        """Create user onboarding workflow template"""
        workflow_def = {
            "name": "User Onboarding",
            "description": "New user onboarding process",
            "triggers": [
                {
                    "type": "event",
                    "config": {
                        "event": "user_created"
                    }
                }
            ],
            "steps": [
                {
                    "id": "create_accounts",
                    "name": "Create User Accounts",
                    "type": "parallel",
                    "parameters": {
                        "systems": ["email", "slack", "github", "jira"]
                    }
                },
                {
                    "id": "assign_permissions",
                    "name": "Assign Permissions",
                    "type": "action",
                    "action": "assign_roles",
                    "parameters": {
                        "role": "{{user_role}}",
                        "department": "{{user_department}}"
                    },
                    "next_steps": ["provision_equipment"]
                },
                {
                    "id": "provision_equipment",
                    "name": "Provision Equipment",
                    "type": "action",
                    "action": "create_ticket",
                    "parameters": {
                        "type": "equipment_request",
                        "items": "{{equipment_list}}"
                    },
                    "next_steps": ["schedule_training"]
                },
                {
                    "id": "schedule_training",
                    "name": "Schedule Training",
                    "type": "action",
                    "action": "schedule_sessions",
                    "parameters": {
                        "sessions": ["orientation", "security_training", "tool_training"]
                    },
                    "next_steps": ["send_welcome"]
                },
                {
                    "id": "send_welcome",
                    "name": "Send Welcome Package",
                    "type": "notification",
                    "parameters": {
                        "template": "welcome_email",
                        "attachments": ["handbook", "policies", "org_chart"]
                    }
                }
            ]
        }
        
        return WorkflowTemplate(
            template_id=str(uuid.uuid4()),
            name="User Onboarding",
            description="Complete user onboarding with account provisioning",
            category=TemplateCategory.INTEGRATION,
            tags=["onboarding", "user", "provisioning"],
            parameters={
                "user_role": {"type": "string", "required": True},
                "user_department": {"type": "string", "required": True},
                "equipment_list": {"type": "array", "default": ["laptop", "monitor"]}
            },
            workflow_definition=workflow_def
        )
    
    def _create_performance_monitoring_template(self) -> WorkflowTemplate:
        """Create performance monitoring workflow template"""
        workflow_def = {
            "name": "Performance Monitoring",
            "description": "Continuous performance monitoring and optimization",
            "triggers": [
                {
                    "type": "scheduled",
                    "config": {
                        "interval": 300  # Every 5 minutes
                    }
                }
            ],
            "steps": [
                {
                    "id": "collect_metrics",
                    "name": "Collect Performance Metrics",
                    "type": "parallel",
                    "parameters": {
                        "metrics": ["response_time", "throughput", "error_rate", "cpu", "memory"]
                    }
                },
                {
                    "id": "analyze_trends",
                    "name": "Analyze Trends",
                    "type": "action",
                    "action": "trend_analysis",
                    "parameters": {
                        "window": "1h",
                        "algorithms": ["moving_average", "anomaly_detection"]
                    },
                    "next_steps": ["performance_decision"]
                },
                {
                    "id": "performance_decision",
                    "name": "Performance Decision",
                    "type": "decision",
                    "conditions": [
                        {
                            "field": "anomaly_detected",
                            "operator": "is_true",
                            "next_step": "investigate_anomaly"
                        },
                        {
                            "field": "performance_degraded",
                            "operator": "is_true",
                            "next_step": "auto_scale"
                        }
                    ]
                },
                {
                    "id": "investigate_anomaly",
                    "name": "Investigate Anomaly",
                    "type": "action",
                    "action": "root_cause_analysis",
                    "parameters": {
                        "depth": "detailed",
                        "correlation_window": "30m"
                    },
                    "next_steps": ["create_alert"]
                },
                {
                    "id": "auto_scale",
                    "name": "Auto Scale Resources",
                    "type": "action",
                    "action": "scale_resources",
                    "parameters": {
                        "strategy": "predictive",
                        "min_instances": 2,
                        "max_instances": 10
                    }
                },
                {
                    "id": "create_alert",
                    "name": "Create Performance Alert",
                    "type": "action",
                    "action": "create_alert",
                    "parameters": {
                        "severity": "{{calculated_severity}}",
                        "include_diagnostics": true
                    }
                }
            ]
        }
        
        return WorkflowTemplate(
            template_id=str(uuid.uuid4()),
            name="Performance Monitoring",
            description="Continuous performance monitoring with auto-scaling",
            category=TemplateCategory.MONITORING,
            tags=["performance", "monitoring", "auto-scale"],
            workflow_definition=workflow_def
        )
    
    def add_template(self, template: WorkflowTemplate) -> bool:
        """Add a template to the library"""
        if template.template_id in self.templates:
            return False
        
        self.templates[template.template_id] = template
        
        # Add to category index
        if template.category not in self.categories:
            self.categories[template.category] = []
        self.categories[template.category].append(template.template_id)
        
        return True
    
    def get_template(self, template_id: str) -> Optional[WorkflowTemplate]:
        """Get a template by ID"""
        return self.templates.get(template_id)
    
    def get_templates_by_category(self, 
                                 category: TemplateCategory) -> List[WorkflowTemplate]:
        """Get all templates in a category"""
        template_ids = self.categories.get(category, [])
        return [self.templates[tid] for tid in template_ids if tid in self.templates]
    
    def search_templates(self, query: str) -> List[WorkflowTemplate]:
        """Search templates by name, description, or tags"""
        query_lower = query.lower()
        results = []
        
        for template in self.templates.values():
            if (query_lower in template.name.lower() or
                query_lower in template.description.lower() or
                any(query_lower in tag for tag in template.tags)):
                results.append(template)
        
        return results
    
    def instantiate_workflow(self, template_id: str,
                           parameters: Dict[str, Any]) -> Optional[Workflow]:
        """Create a workflow instance from a template"""
        template = self.get_template(template_id)
        if not template:
            return None
        
        # Merge parameters with defaults
        effective_params = {}
        for param_name, param_config in template.parameters.items():
            if param_name in parameters:
                effective_params[param_name] = parameters[param_name]
            elif "default" in param_config:
                effective_params[param_name] = param_config["default"]
            elif param_config.get("required", False):
                raise ValueError(f"Required parameter '{param_name}' not provided")
        
        # Create workflow from definition
        workflow_def = self._substitute_parameters(
            template.workflow_definition,
            effective_params
        )
        
        workflow = Workflow(
            workflow_id=str(uuid.uuid4()),
            name=workflow_def.get("name", template.name),
            description=workflow_def.get("description", template.description),
            variables={**template.variables, **effective_params}
        )
        
        # Add triggers
        for trigger_def in workflow_def.get("triggers", []):
            trigger = WorkflowTrigger(
                trigger_id=str(uuid.uuid4()),
                trigger_type=TriggerType(trigger_def["type"]),
                conditions=trigger_def.get("config", {})
            )
            workflow.triggers.append(trigger)
        
        # Add steps
        for step_def in workflow_def.get("steps", []):
            step = WorkflowStep(
                step_id=step_def["id"],
                name=step_def["name"],
                step_type=StepType(step_def["type"]),
                action=step_def.get("action"),
                parameters=step_def.get("parameters", {}),
                next_steps=step_def.get("next_steps", [])
            )
            workflow.steps.append(step)
        
        # Update template usage
        template.usage_count += 1
        
        return workflow
    
    def _substitute_parameters(self, definition: Any,
                             parameters: Dict[str, Any]) -> Any:
        """Substitute template parameters in definition"""
        if isinstance(definition, str):
            # Replace {{param}} with actual values
            import re
            
            def replacer(match):
                param = match.group(1)
                return str(parameters.get(param, match.group(0)))
            
            return re.sub(r'\{\{([^}]+)\}\}', replacer, definition)
        
        elif isinstance(definition, dict):
            return {
                key: self._substitute_parameters(value, parameters)
                for key, value in definition.items()
            }
        
        elif isinstance(definition, list):
            return [
                self._substitute_parameters(item, parameters)
                for item in definition
            ]
        
        return definition
    
    def export_template(self, template_id: str) -> Optional[str]:
        """Export a template as JSON"""
        template = self.get_template(template_id)
        if not template:
            return None
        
        export_data = {
            "name": template.name,
            "description": template.description,
            "category": template.category.value,
            "version": template.version,
            "author": template.author,
            "tags": template.tags,
            "parameters": template.parameters,
            "variables": template.variables,
            "workflow_definition": template.workflow_definition
        }
        
        return json.dumps(export_data, indent=2)
    
    def import_template(self, template_json: str) -> Optional[WorkflowTemplate]:
        """Import a template from JSON"""
        try:
            data = json.loads(template_json)
            
            template = WorkflowTemplate(
                template_id=str(uuid.uuid4()),
                name=data["name"],
                description=data["description"],
                category=TemplateCategory(data["category"]),
                version=data.get("version", "1.0.0"),
                author=data.get("author", "imported"),
                tags=data.get("tags", []),
                parameters=data.get("parameters", {}),
                variables=data.get("variables", {}),
                workflow_definition=data["workflow_definition"]
            )
            
            self.add_template(template)
            return template
            
        except Exception as e:
            print(f"Failed to import template: {e}")
            return None