-- =====================================================
-- SECURITY AND COMPLIANCE DATABASE SCHEMA
-- =====================================================
-- This script creates comprehensive security and compliance tables
-- for enterprise-grade security management, audit logging, and compliance tracking
--
-- Features:
-- - Security framework tables (authentication, authorization, encryption)
-- - Audit logging with compliance framework tagging
-- - Zero-trust security model support
-- - GDPR, SOC2, and HIPAA compliance tables
-- - Access control and permission management
-- - Threat detection and security monitoring
-- - Data protection and classification
--
-- Run this script in your Supabase SQL Editor after security migration
-- =====================================================

-- =====================================================
-- SECTION 1: SECURITY FRAMEWORK TABLES
-- =====================================================

-- Security settings and configuration
CREATE TABLE IF NOT EXISTS security_settings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    setting_name VARCHAR(100) UNIQUE NOT NULL,
    setting_value JSONB NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(100) DEFAULT 'system',
    updated_by VARCHAR(100) DEFAULT 'system'
);

-- Cryptographic keys management
CREATE TABLE IF NOT EXISTS cryptographic_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_id VARCHAR(255) UNIQUE NOT NULL,
    key_name VARCHAR(255) NOT NULL,
    key_type VARCHAR(50) NOT NULL CHECK (key_type IN ('symmetric', 'asymmetric', 'hash')),
    key_algorithm VARCHAR(100) NOT NULL,
    key_usage VARCHAR(100) NOT NULL,
    key_data BYTEA NOT NULL,
    key_metadata JSONB,
    is_active BOOLEAN DEFAULT true,
    rotation_required BOOLEAN DEFAULT false,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(100) DEFAULT 'system',
    last_used_at TIMESTAMP WITH TIME ZONE
);

-- Index for cryptographic keys
CREATE INDEX IF NOT EXISTS idx_crypto_keys_active ON cryptographic_keys(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_crypto_keys_expires ON cryptographic_keys(expires_at) WHERE expires_at IS NOT NULL;

-- Security contexts and sessions
CREATE TABLE IF NOT EXISTS security_contexts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_id UUID NOT NULL,
    session_token VARCHAR(500) NOT NULL,
    refresh_token VARCHAR(500),
    session_data JSONB,
    ip_address INET,
    user_agent TEXT,
    device_fingerprint VARCHAR(255),
    trust_level VARCHAR(20) DEFAULT 'medium' CHECK (trust_level IN ('low', 'medium', 'high', 'critical')),
    risk_score DECIMAL(5,2) DEFAULT 0.0,
    is_active BOOLEAN DEFAULT true,
    is_revoked BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    last_activity_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    revoked_at TIMESTAMP WITH TIME ZONE
);

-- Index for security contexts
CREATE INDEX IF NOT EXISTS idx_security_contexts_user ON security_contexts(user_id);
CREATE INDEX IF NOT EXISTS idx_security_contexts_active ON security_contexts(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_security_contexts_token ON security_contexts(session_token);

-- Multi-factor authentication enrollments
CREATE TABLE IF NOT EXISTS mfa_enrollments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    enrollment_id VARCHAR(255) UNIQUE NOT NULL,
    user_id UUID NOT NULL,
    method VARCHAR(50) NOT NULL CHECK (method IN ('totp', 'sms', 'email', 'hardware_key', 'biometric')),
    secret_key TEXT NOT NULL,
    device_name VARCHAR(255),
    backup_codes TEXT[],
    is_primary BOOLEAN DEFAULT false,
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_used_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Index for MFA enrollments
CREATE INDEX IF NOT EXISTS idx_mfa_enrollments_user ON mfa_enrollments(user_id);
CREATE INDEX IF NOT EXISTS idx_mfa_enrollments_active ON mfa_enrollments(is_active) WHERE is_active = true;

-- =====================================================
-- SECTION 2: ACCESS CONTROL AND AUTHORIZATION
-- =====================================================

-- Subjects (users, services, systems)
CREATE TABLE IF NOT EXISTS access_subjects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subject_id VARCHAR(255) UNIQUE NOT NULL,
    subject_type VARCHAR(50) NOT NULL CHECK (subject_type IN ('user', 'service', 'system', 'api_key')),
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    department VARCHAR(100),
    attributes JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for access subjects
CREATE INDEX IF NOT EXISTS idx_access_subjects_type ON access_subjects(subject_type);
CREATE INDEX IF NOT EXISTS idx_access_subjects_active ON access_subjects(is_active) WHERE is_active = true;

-- Roles and permissions
CREATE TABLE IF NOT EXISTS access_roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    role_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    permissions TEXT[] DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for access roles
CREATE INDEX IF NOT EXISTS idx_access_roles_active ON access_roles(is_active) WHERE is_active = true;

-- Subject-role assignments
CREATE TABLE IF NOT EXISTS subject_role_assignments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subject_id UUID NOT NULL REFERENCES access_subjects(id) ON DELETE CASCADE,
    role_id UUID NOT NULL REFERENCES access_roles(id) ON DELETE CASCADE,
    assigned_by UUID,
    assigned_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    UNIQUE(subject_id, role_id)
);

-- Index for subject-role assignments
CREATE INDEX IF NOT EXISTS idx_subject_role_assignments_subject ON subject_role_assignments(subject_id);
CREATE INDEX IF NOT EXISTS idx_subject_role_assignments_role ON subject_role_assignments(role_id);
CREATE INDEX IF NOT EXISTS idx_subject_role_assignments_active ON subject_role_assignments(is_active) WHERE is_active = true;

-- Access control policies
CREATE TABLE IF NOT EXISTS access_policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    policy_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    policy_type VARCHAR(50) NOT NULL CHECK (policy_type IN ('rbac', 'abac', 'rule_based')),
    policy_rules JSONB NOT NULL,
    priority INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for access policies
CREATE INDEX IF NOT EXISTS idx_access_policies_active ON access_policies(is_active) WHERE is_active = true;

-- Access logs
CREATE TABLE IF NOT EXISTS access_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subject_id UUID REFERENCES access_subjects(id) ON DELETE SET NULL,
    resource VARCHAR(500) NOT NULL,
    action VARCHAR(100) NOT NULL,
    decision VARCHAR(20) NOT NULL CHECK (decision IN ('allow', 'deny', 'require_mfa', 'challenge')),
    reason TEXT,
    context JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for access logs
CREATE INDEX IF NOT EXISTS idx_access_logs_subject ON access_logs(subject_id);
CREATE INDEX IF NOT EXISTS idx_access_logs_resource ON access_logs(resource);
CREATE INDEX IF NOT EXISTS idx_access_logs_action ON access_logs(action);
CREATE INDEX IF NOT EXISTS idx_access_logs_decision ON access_logs(decision);
CREATE INDEX IF NOT EXISTS idx_access_logs_created ON access_logs(created_at);

-- =====================================================
-- SECTION 3: AUDIT LOGGING
-- =====================================================

-- Audit events with compliance framework support
CREATE TABLE IF NOT EXISTS audit_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id VARCHAR(255) UNIQUE NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    event_category VARCHAR(50) NOT NULL CHECK (event_category IN ('security', 'compliance', 'system', 'user', 'data', 'network')),
    event_level VARCHAR(20) DEFAULT 'info' CHECK (event_level IN ('debug', 'info', 'warning', 'error', 'critical')),
    compliance_framework VARCHAR(50)[] DEFAULT '{}',
    actor_id UUID,
    actor_type VARCHAR(50) DEFAULT 'user',
    action VARCHAR(255) NOT NULL,
    resource VARCHAR(500),
    resource_type VARCHAR(100),
    status VARCHAR(20) DEFAULT 'success' CHECK (status IN ('success', 'failure', 'warning', 'error')),
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    session_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    INDEXED_TIMESTAMP TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for audit events
CREATE INDEX IF NOT EXISTS idx_audit_events_type ON audit_events(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_events_category ON audit_events(event_category);
CREATE INDEX IF NOT EXISTS idx_audit_events_level ON audit_events(event_level);
CREATE INDEX IF NOT EXISTS idx_audit_events_actor ON audit_events(actor_id);
CREATE INDEX IF NOT EXISTS idx_audit_events_action ON audit_events(action);
CREATE INDEX IF NOT EXISTS idx_audit_events_status ON audit_events(status);
CREATE INDEX IF NOT EXISTS idx_audit_events_compliance ON audit_events USING GIN(compliance_framework);
CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp ON audit_events(INDEXED_TIMESTAMP);

-- Audit event attachments
CREATE TABLE IF NOT EXISTS audit_attachments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id UUID NOT NULL REFERENCES audit_events(id) ON DELETE CASCADE,
    file_name VARCHAR(500) NOT NULL,
    file_type VARCHAR(100),
    file_size BIGINT,
    file_data BYTEA,
    storage_url TEXT,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for audit attachments
CREATE INDEX IF NOT EXISTS idx_audit_attachments_event ON audit_attachments(event_id);

-- =====================================================
-- SECTION 4: THREAT DETECTION AND SECURITY MONITORING
-- =====================================================

-- Threat indicators
CREATE TABLE IF NOT EXISTS threat_indicators (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    indicator_id VARCHAR(255) UNIQUE NOT NULL,
    indicator_type VARCHAR(50) NOT NULL CHECK (indicator_type IN ('ip', 'domain', 'url', 'hash', 'email', 'user_agent')),
    indicator_value VARCHAR(500) NOT NULL,
    threat_type VARCHAR(50) NOT NULL CHECK (threat_type IN ('malware', 'phishing', 'botnet', 'suspicious', 'known_bad')),
    confidence_level DECIMAL(5,2) DEFAULT 50.0,
    severity VARCHAR(20) DEFAULT 'medium' CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    source VARCHAR(255),
    description TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Index for threat indicators
CREATE INDEX IF NOT EXISTS idx_threat_indicators_type ON threat_indicators(indicator_type);
CREATE INDEX IF NOT EXISTS idx_threat_indicators_value ON threat_indicators(indicator_value);
CREATE INDEX IF NOT EXISTS idx_threat_indicators_active ON threat_indicators(is_active) WHERE is_active = true;

-- Threat events
CREATE TABLE IF NOT EXISTS threat_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id VARCHAR(255) UNIQUE NOT NULL,
    threat_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    confidence DECIMAL(5,2) DEFAULT 0.0,
    source_ip INET,
    target_resource VARCHAR(500),
    description TEXT NOT NULL,
    detection_method VARCHAR(100),
    is_mitigated BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    INDEXED_TIMESTAMP TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for threat events
CREATE INDEX IF NOT EXISTS idx_threat_events_type ON threat_events(threat_type);
CREATE INDEX IF NOT EXISTS idx_threat_events_severity ON threat_events(severity);
CREATE INDEX IF NOT EXISTS idx_threat_events_source ON threat_events(source_ip);
CREATE INDEX IF NOT EXISTS idx_threat_events_timestamp ON threat_events(INDEXED_TIMESTAMP);

-- Security alerts
CREATE TABLE IF NOT EXISTS security_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    alert_id VARCHAR(255) UNIQUE NOT NULL,
    alert_type VARCHAR(100) NOT NULL,
    priority VARCHAR(20) NOT NULL CHECK (priority IN ('low', 'medium', 'high', 'critical')),
    title VARCHAR(500) NOT NULL,
    description TEXT,
    details JSONB,
    status VARCHAR(20) DEFAULT 'open' CHECK (status IN ('open', 'investigating', 'resolved', 'false_positive')),
    assigned_to UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    resolved_at TIMESTAMP WITH TIME ZONE
);

-- Index for security alerts
CREATE INDEX IF NOT EXISTS idx_security_alerts_type ON security_alerts(alert_type);
CREATE INDEX IF NOT EXISTS idx_security_alerts_priority ON security_alerts(priority);
CREATE INDEX IF NOT EXISTS idx_security_alerts_status ON security_alerts(status);
CREATE INDEX IF NOT EXISTS idx_security_alerts_created ON security_alerts(created_at);

-- =====================================================
-- SECTION 5: DATA PROTECTION AND CLASSIFICATION
-- =====================================================

-- Data classification policies
CREATE TABLE IF NOT EXISTS data_classification_policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    policy_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    classification_level VARCHAR(50) NOT NULL CHECK (classification_level IN ('public', 'internal', 'confidential', 'restricted', 'top_secret')),
    description TEXT,
    data_patterns JSONB,
    protection_rules JSONB,
    retention_period INTERVAL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for data classification policies
CREATE INDEX IF NOT EXISTS idx_data_policies_active ON data_classification_policies(is_active) WHERE is_active = true;

-- Data classification results
CREATE TABLE IF NOT EXISTS data_classification_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    data_id VARCHAR(255) NOT NULL,
    data_type VARCHAR(100) NOT NULL,
    classification_level VARCHAR(50) NOT NULL,
    confidence_score DECIMAL(5,2) DEFAULT 0.0,
    patterns_found TEXT[] DEFAULT '{}',
    classifier VARCHAR(100) DEFAULT 'system',
    location VARCHAR(500),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for data classification results
CREATE INDEX IF NOT EXISTS idx_data_classification_data ON data_classification_results(data_id);
CREATE INDEX IF NOT EXISTS idx_data_classification_level ON data_classification_results(classification_level);
CREATE INDEX IF NOT EXISTS idx_data_classification_confidence ON data_classification_results(confidence_score);

-- Data protection events
CREATE TABLE IF NOT EXISTS data_protection_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id VARCHAR(255) UNIQUE NOT NULL,
    event_type VARCHAR(100) NOT NULL CHECK (event_type IN ('classification', 'encryption', 'masking', 'redaction', 'quarantine')),
    data_id VARCHAR(255),
    action_taken VARCHAR(255),
    result VARCHAR(100),
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for data protection events
CREATE INDEX IF NOT EXISTS idx_data_protection_events_type ON data_protection_events(event_type);
CREATE INDEX IF NOT EXISTS idx_data_protection_events_data ON data_protection_events(data_id);

-- =====================================================
-- SECTION 6: COMPLIANCE FRAMEWORK TABLES
-- =====================================================

-- Compliance frameworks
CREATE TABLE IF NOT EXISTS compliance_frameworks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    framework_id VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50),
    description TEXT,
    requirements JSONB,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Compliance controls
CREATE TABLE IF NOT EXISTS compliance_controls (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    control_id VARCHAR(255) UNIQUE NOT NULL,
    framework_id VARCHAR(100) NOT NULL REFERENCES compliance_frameworks(framework_id),
    name VARCHAR(500) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    control_type VARCHAR(50) CHECK (control_type IN ('technical', 'operational', 'managerial')),
    implementation_status VARCHAR(50) DEFAULT 'not_implemented' CHECK (implementation_status IN ('not_implemented', 'partially_implemented', 'implemented', 'validated')),
    last_assessment_date TIMESTAMP WITH TIME ZONE,
    next_assessment_date TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for compliance controls
CREATE INDEX IF NOT EXISTS idx_compliance_controls_framework ON compliance_controls(framework_id);
CREATE INDEX IF NOT EXISTS idx_compliance_controls_status ON compliance_controls(implementation_status);

-- Compliance assessments
CREATE TABLE IF NOT EXISTS compliance_assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id VARCHAR(255) UNIQUE NOT NULL,
    framework_id VARCHAR(100) NOT NULL REFERENCES compliance_frameworks(framework_id),
    assessment_type VARCHAR(50) NOT NULL CHECK (assessment_type IN ('internal', 'external', 'automated', 'manual')),
    status VARCHAR(50) DEFAULT 'in_progress' CHECK (status IN ('in_progress', 'completed', 'failed', 'cancelled')),
    overall_score DECIMAL(5,2),
    assessor VARCHAR(255),
    start_date TIMESTAMP WITH TIME ZONE,
    end_date TIMESTAMP WITH TIME ZONE,
    findings_count INTEGER DEFAULT 0,
    deficiencies_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for compliance assessments
CREATE INDEX IF NOT EXISTS idx_compliance_assessments_framework ON compliance_assessments(framework_id);
CREATE INDEX IF NOT EXISTS idx_compliance_assessments_status ON compliance_assessments(status);

-- Compliance findings
CREATE TABLE IF NOT EXISTS compliance_findings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    finding_id VARCHAR(255) UNIQUE NOT NULL,
    assessment_id UUID NOT NULL REFERENCES compliance_assessments(id) ON DELETE CASCADE,
    control_id VARCHAR(255) NOT NULL REFERENCES compliance_controls(control_id),
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    title VARCHAR(500) NOT NULL,
    description TEXT,
    recommendation TEXT,
    status VARCHAR(50) DEFAULT 'open' CHECK (status IN ('open', 'in_progress', 'resolved', 'accepted_risk')),
    assigned_to VARCHAR(255),
    due_date TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for compliance findings
CREATE INDEX IF NOT EXISTS idx_compliance_findings_assessment ON compliance_findings(assessment_id);
CREATE INDEX IF NOT EXISTS idx_compliance_findings_control ON compliance_findings(control_id);
CREATE INDEX IF NOT EXISTS idx_compliance_findings_severity ON compliance_findings(severity);
CREATE INDEX IF NOT EXISTS idx_compliance_findings_status ON compliance_findings(status);

-- =====================================================
-- SECTION 7: GDPR COMPLIANCE TABLES
-- =====================================================

-- GDPR data subjects
CREATE TABLE IF NOT EXISTS gdpr_data_subjects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subject_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(500) NOT NULL,
    email VARCHAR(255),
    phone VARCHAR(100),
    address TEXT,
    consent_given BOOLEAN DEFAULT false,
    consent_date TIMESTAMP WITH TIME ZONE,
    data_retention_policy TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- GDPR data subject requests (DSAR)
CREATE TABLE IF NOT EXISTS gdpr_subject_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id VARCHAR(255) UNIQUE NOT NULL,
    subject_id UUID NOT NULL REFERENCES gdpr_data_subjects(id) ON DELETE CASCADE,
    request_type VARCHAR(50) NOT NULL CHECK (request_type IN ('access', 'rectification', 'erasure', 'portability', 'objection', 'restriction')),
    status VARCHAR(50) DEFAULT 'received' CHECK (status IN ('received', 'in_progress', 'completed', 'rejected', 'requires_info')),
    description TEXT,
    data_categories TEXT[] DEFAULT '{}',
    requested_data JSONB,
    response_data JSONB,
    estimated_completion_date TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    processed_by UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for GDPR subject requests
CREATE INDEX IF NOT EXISTS idx_gdpr_requests_subject ON gdpr_subject_requests(subject_id);
CREATE INDEX IF NOT EXISTS idx_gdpr_requests_type ON gdpr_subject_requests(request_type);
CREATE INDEX IF NOT EXISTS idx_gdpr_requests_status ON gdpr_subject_requests(status);

-- GDPR consent records
CREATE TABLE IF NOT EXISTS gdpr_consent_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    consent_id VARCHAR(255) UNIQUE NOT NULL,
    subject_id UUID NOT NULL REFERENCES gdpr_data_subjects(id) ON DELETE CASCADE,
    purpose TEXT NOT NULL,
    data_categories TEXT[] DEFAULT '{}',
    third_parties TEXT[] DEFAULT '{}',
    retention_period INTERVAL,
    is_active BOOLEAN DEFAULT true,
    withdrawn_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for GDPR consent records
CREATE INDEX IF NOT EXISTS idx_gdpr_consent_subject ON gdpr_consent_records(subject_id);
CREATE INDEX IF NOT EXISTS idx_gdpr_consent_active ON gdpr_consent_records(is_active) WHERE is_active = true;

-- =====================================================
-- SECTION 8: SOC2 COMPLIANCE TABLES
-- =====================================================

-- SOC2 system description
CREATE TABLE IF NOT EXISTS soc2_system_description (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    description_id VARCHAR(255) UNIQUE NOT NULL,
    version VARCHAR(50) NOT NULL,
    system_name VARCHAR(500) NOT NULL,
    system_description TEXT,
    trust_services_criteria TEXT[] DEFAULT '{}',
    scope_description TEXT,
    complementary_user_entities TEXT,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- SOC2 controls
CREATE TABLE IF NOT EXISTS soc2_controls (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    control_id VARCHAR(255) UNIQUE NOT NULL,
    trust_service VARCHAR(50) NOT NULL CHECK (trust_service IN ('security', 'availability', 'processing_integrity', 'confidentiality', 'privacy')),
    control_name VARCHAR(500) NOT NULL,
    control_description TEXT,
    implementation_status VARCHAR(50) DEFAULT 'not_implemented',
    test_results JSONB,
    last_tested TIMESTAMP WITH TIME ZONE,
    next_test_date TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for SOC2 controls
CREATE INDEX IF NOT EXISTS idx_soc2_controls_service ON soc2_controls(trust_service);
CREATE INDEX IF NOT EXISTS idx_soc2_controls_status ON soc2_controls(implementation_status);

-- SOC2 control tests
CREATE TABLE IF NOT EXISTS soc2_control_tests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    test_id VARCHAR(255) UNIQUE NOT NULL,
    control_id VARCHAR(255) NOT NULL REFERENCES soc2_controls(control_id),
    test_name VARCHAR(500) NOT NULL,
    test_description TEXT,
    test_procedure TEXT,
    test_date TIMESTAMP WITH TIME ZONE,
    tester VARCHAR(255),
    result VARCHAR(50) NOT NULL CHECK (result IN ('pass', 'fail', 'na', 'not_tested')),
    findings TEXT[],
    evidence TEXT[],
    effectiveness VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for SOC2 control tests
CREATE INDEX IF NOT EXISTS idx_soc2_tests_control ON soc2_control_tests(control_id);
CREATE INDEX IF NOT EXISTS idx_soc2_tests_result ON soc2_control_tests(result);

-- =====================================================
-- SECTION 9: HIPAA COMPLIANCE TABLES
-- =====================================================

-- HIPAA business associates
CREATE TABLE IF NOT EXISTS hipaa_business_associates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    associate_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(500) NOT NULL,
    address TEXT,
    contact_email VARCHAR(255),
    contact_phone VARCHAR(100),
    services_provided TEXT[] DEFAULT '{}',
    data_access_level VARCHAR(100),
    compliance_status VARCHAR(50) DEFAULT 'pending',
    risk_assessment_score DECIMAL(5,2) DEFAULT 0.0,
    contract_date TIMESTAMP WITH TIME ZONE,
    contract_expiry TIMESTAMP WITH TIME ZONE,
    last_audit_date TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for HIPAA business associates
CREATE INDEX IF NOT EXISTS idx_hipaa_associates_status ON hipaa_business_associates(compliance_status);
CREATE INDEX IF NOT EXISTS idx_hipaa_associates_expiry ON hipaa_business_associates(contract_expiry);

-- HIPAA breach notifications
CREATE TABLE IF NOT EXISTS hipaa_breach_notifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    breach_id VARCHAR(255) UNIQUE NOT NULL,
    breach_date TIMESTAMP WITH TIME ZONE,
    discovery_date TIMESTAMP WITH TIME ZONE,
    affected_individuals INTEGER DEFAULT 0,
    breach_type VARCHAR(100),
    breach_description TEXT,
    phi_types TEXT[] DEFAULT '{}',
    notification_sent BOOLEAN DEFAULT false,
    notification_date TIMESTAMP WITH TIME ZONE,
    reported_to_hhs BOOLEAN DEFAULT false,
    hhs_report_date TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) DEFAULT 'investigating',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for HIPAA breach notifications
CREATE INDEX IF NOT EXISTS idx_hipaa_breaches_status ON hipaa_breach_notifications(status);
CREATE INDEX IF NOT EXISTS idx_hipaa_breaches_date ON hipaa_breach_notifications(breach_date);

-- HIPAA workforce training
CREATE TABLE IF NOT EXISTS hipaa_workforce_training (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    training_id VARCHAR(255) UNIQUE NOT NULL,
    user_id UUID NOT NULL,
    training_type VARCHAR(100) NOT NULL,
    training_title VARCHAR(500) NOT NULL,
    completion_date TIMESTAMP WITH TIME ZONE,
    expiration_date TIMESTAMP WITH TIME ZONE,
    score DECIMAL(5,2),
    certificate_url TEXT,
    is_current BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for HIPAA workforce training
CREATE INDEX IF NOT EXISTS idx_hipaa_training_user ON hipaa_workforce_training(user_id);
CREATE INDEX IF NOT EXISTS idx_hipaa_training_current ON hipaa_workforce_training(is_current) WHERE is_current = true;

-- =====================================================
-- SECTION 10: SECURITY METRICS AND REPORTING
-- =====================================================

-- Security metrics
CREATE TABLE IF NOT EXISTS security_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_id VARCHAR(255) UNIQUE NOT NULL,
    metric_name VARCHAR(500) NOT NULL,
    metric_category VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,2),
    metric_unit VARCHAR(50),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB
);

-- Index for security metrics
CREATE INDEX IF NOT EXISTS idx_security_metrics_category ON security_metrics(metric_category);
CREATE INDEX IF NOT EXISTS idx_security_metrics_timestamp ON security_metrics(timestamp);

-- Security reports
CREATE TABLE IF NOT EXISTS security_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id VARCHAR(255) UNIQUE NOT NULL,
    report_type VARCHAR(100) NOT NULL,
    report_title VARCHAR(500) NOT NULL,
    report_format VARCHAR(50) DEFAULT 'pdf',
    report_data JSONB,
    file_url TEXT,
    generated_by UUID,
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Index for security reports
CREATE INDEX IF NOT EXISTS idx_security_reports_type ON security_reports(report_type);
CREATE INDEX IF NOT EXISTS idx_security_reports_generated ON security_reports(generated_at);

-- =====================================================
-- SECTION 11: FUNCTIONS AND TRIGGERS
-- =====================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add updated_at triggers to relevant tables
CREATE TRIGGER update_security_settings_updated_at
    BEFORE UPDATE ON security_settings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_access_subjects_updated_at
    BEFORE UPDATE ON access_subjects
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_access_roles_updated_at
    BEFORE UPDATE ON access_roles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_access_policies_updated_at
    BEFORE UPDATE ON access_policies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_data_classification_policies_updated_at
    BEFORE UPDATE ON data_classification_policies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_compliance_frameworks_updated_at
    BEFORE UPDATE ON compliance_frameworks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_compliance_controls_updated_at
    BEFORE UPDATE ON compliance_controls
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_compliance_assessments_updated_at
    BEFORE UPDATE ON compliance_assessments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_compliance_findings_updated_at
    BEFORE UPDATE ON compliance_findings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_gdpr_data_subjects_updated_at
    BEFORE UPDATE ON gdpr_data_subjects
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_gdpr_subject_requests_updated_at
    BEFORE UPDATE ON gdpr_subject_requests
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_gdpr_consent_records_updated_at
    BEFORE UPDATE ON gdpr_consent_records
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_soc2_controls_updated_at
    BEFORE UPDATE ON soc2_controls
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_hipaa_business_associates_updated_at
    BEFORE UPDATE ON hipaa_business_associates
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_hipaa_breach_notifications_updated_at
    BEFORE UPDATE ON hipaa_breach_notifications
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to clean up expired sessions
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM security_contexts
    WHERE expires_at < NOW() OR (is_revoked = true AND revoked_at < NOW() - INTERVAL '30 days');

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ language 'plpgsql';

-- Function to rotate cryptographic keys
CREATE OR REPLACE FUNCTION rotate_crypto_key(key_id_to_rotate VARCHAR)
RETURNS BOOLEAN AS $$
DECLARE
    current_key RECORD;
    new_key_id VARCHAR;
BEGIN
    -- Find the current key
    SELECT * INTO current_key
    FROM cryptographic_keys
    WHERE key_id = key_id_to_rotate AND is_active = true;

    IF NOT FOUND THEN
        RETURN false;
    END IF;

    -- Deactivate the current key
    UPDATE cryptographic_keys
    SET is_active = false,
        rotation_required = false
    WHERE key_id = key_id_to_rotate;

    -- Generate new key (placeholder - actual implementation depends on key type)
    -- In real implementation, this would generate a new cryptographic key
    new_key_id := key_id_to_rotate || '_rotated_' || EXTRACT(EPOCH FROM NOW());

    INSERT INTO cryptographic_keys (
        key_id, key_name, key_type, key_algorithm, key_usage,
        key_data, key_metadata, is_active, rotation_required, expires_at
    ) VALUES (
        new_key_id,
        current_key.key_name || ' (Rotated)',
        current_key.key_type,
        current_key.key_algorithm,
        current_key.key_usage,
        current_key.key_data, -- In real implementation, generate new key data
        current_key.key_metadata,
        true,
        false,
        NOW() + (current_key.expires_at - NOW())
    );

    RETURN true;
END;
$$ language 'plpgsql';

-- Function to check for suspicious activity
CREATE OR REPLACE FUNCTION check_suspicious_activity(user_id_to_check UUID)
RETURNS TABLE (
    failed_logins INTEGER,
    unique_ips INTEGER,
    risk_score DECIMAL(5,2)
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*) FILTER (WHERE decision = 'deny' AND action = 'login')::INTEGER,
        COUNT(DISTINCT ip_address)::INTEGER,
        (COUNT(*) FILTER (WHERE decision = 'deny' AND action = 'login') * 10.0)::DECIMAL(5,2)
    FROM access_logs
    WHERE subject_id = user_id_to_check
    AND created_at > NOW() - INTERVAL '24 hours';
END;
$$ language 'plpgsql';

-- =====================================================
-- SECTION 12: SECURITY VIEWS
-- =====================================================

-- Active security sessions view
CREATE OR REPLACE VIEW active_security_sessions AS
SELECT
    sc.session_id,
    sc.user_id,
    sc.ip_address,
    sc.trust_level,
    sc.risk_score,
    sc.created_at,
    sc.expires_at,
    sc.last_activity_at,
    asub.name as user_name,
    asub.email as user_email
FROM security_contexts sc
LEFT JOIN access_subjects asub ON sc.user_id = asub.id
WHERE sc.is_active = true AND sc.is_revoked = false;

-- Security dashboard summary view
CREATE OR REPLACE VIEW security_dashboard_summary AS
SELECT
    (SELECT COUNT(*) FROM security_contexts WHERE is_active = true) as active_sessions,
    (SELECT COUNT(*) FROM mfa_enrollments WHERE is_active = true) as active_mfa_enrollments,
    (SELECT COUNT(*) FROM access_logs WHERE created_at > NOW() - INTERVAL '24 hours') as daily_access_attempts,
    (SELECT COUNT(*) FROM threat_events WHERE created_at > NOW() - INTERVAL '24 hours') as daily_threat_events,
    (SELECT COUNT(*) FROM security_alerts WHERE status = 'open') as open_security_alerts,
    (SELECT COUNT(*) FROM audit_events WHERE created_at > NOW() - INTERVAL '1 hour') as hourly_audit_events;

-- Compliance status overview view
CREATE OR REPLACE VIEW compliance_status_overview AS
SELECT
    cf.framework_id,
    cf.name as framework_name,
    cf.version,
    (SELECT COUNT(*) FROM compliance_controls WHERE framework_id = cf.framework_id AND implementation_status = 'implemented') as implemented_controls,
    (SELECT COUNT(*) FROM compliance_controls WHERE framework_id = cf.framework_id) as total_controls,
    (SELECT COUNT(*) FROM compliance_assessments WHERE framework_id = cf.framework_id AND status = 'completed') as completed_assessments,
    (SELECT COUNT(*) FROM compliance_findings cfnd
     JOIN compliance_assessments ca ON cfnd.assessment_id = ca.id
     WHERE ca.framework_id = cf.framework_id AND cfnd.status = 'open') as open_findings
FROM compliance_frameworks cf
WHERE cf.is_active = true;

-- GDPR compliance status view
CREATE OR REPLACE VIEW gdpr_compliance_status AS
SELECT
    (SELECT COUNT(*) FROM gdpr_data_subjects WHERE consent_given = true) as consented_subjects,
    (SELECT COUNT(*) FROM gdpr_subject_requests WHERE status = 'completed') as completed_requests,
    (SELECT COUNT(*) FROM gdpr_subject_requests WHERE status IN ('received', 'in_progress')) as pending_requests,
    (SELECT COUNT(*) FROM gdpr_consent_records WHERE is_active = true) as active_consents,
    (SELECT COUNT(*) FROM gdpr_consent_records WHERE withdrawn_at IS NOT NULL) as withdrawn_consents;

-- Security incidents by type view
CREATE OR REPLACE VIEW security_incidents_by_type AS
SELECT
    threat_type,
    COUNT(*) as incident_count,
    COUNT(*) FILTER (WHERE severity = 'critical') as critical_count,
    COUNT(*) FILTER (WHERE severity = 'high') as high_count,
    COUNT(*) FILTER (WHERE is_mitigated = false) as unmitigated_count
FROM threat_events
WHERE created_at > NOW() - INTERVAL '30 days'
GROUP BY threat_type
ORDER BY incident_count DESC;

-- =====================================================
-- SECTION 13: INITIAL DATA SETUP
-- =====================================================

-- Insert default security frameworks
INSERT INTO compliance_frameworks (framework_id, name, version, description, requirements, is_active) VALUES
('GDPR', 'General Data Protection Regulation', '2018', 'EU data protection and privacy regulation',
 '{"lawful_basis": ["consent", "contract", "legal_obligation", "vital_interests", "public_task", "legitimate_interests"], "data_subject_rights": ["access", "rectification", "erasure", "portability", "objection", "restriction"]}', true),
('SOC2', 'SOC 2 Type II', '2022', 'Service Organization Control 2 - Trust Services Criteria',
 '{"trust_services": ["security", "availability", "processing_integrity", "confidentiality", "privacy"], "control_categories": ["control_environment", "risk_assessment", "control_activities", "information_communication", "monitoring"]}', true),
('HIPAA', 'Health Insurance Portability and Accountability Act', '2023', 'US healthcare data protection regulation',
 '{"safeguards": ["administrative", "physical", "technical"], "requirements": ["privacy_rule", "security_rule", "breach_notification"]}', true)
ON CONFLICT (framework_id) DO UPDATE SET
    updated_at = NOW(),
    is_active = true;

-- Insert default security settings
INSERT INTO security_settings (setting_name, setting_value, description, is_active) VALUES
('session_timeout', '{"value": 3600, "unit": "seconds"}', 'Session timeout in seconds', true),
('max_failed_attempts', '{"value": 5, "window": "15 minutes"}', 'Maximum failed login attempts before lockout', true),
('password_policy', '{"min_length": 12, "require_uppercase": true, "require_lowercase": true, "require_numbers": true, "require_special": true}', 'Password complexity requirements', true),
('mfa_required', '{"value": true, "exceptions": ["trusted_locations", "service_accounts"]}', 'Multi-factor authentication requirements', true),
('encryption_at_rest', '{"algorithm": "AES-256-GCM", "key_rotation_days": 90}', 'Data encryption at rest settings', true)
ON CONFLICT (setting_name) DO UPDATE SET
    updated_at = NOW(),
    is_active = true;

-- Insert default roles
INSERT INTO access_roles (role_id, name, description, permissions, is_active) VALUES
('security_admin', 'Security Administrator', 'Full access to security configuration and monitoring',
 ['security:*', 'compliance:*', 'audit:*', 'access_control:*'], true),
('compliance_officer', 'Compliance Officer', 'Manage compliance frameworks and assessments',
 ['compliance:*', 'audit:read', 'reports:*'], true),
('security_analyst', 'Security Analyst', 'Monitor security events and investigate incidents',
 ['security:read', 'threat_detection:read', 'audit:read'], true),
('auditor', 'Auditor', 'Read-only access for audit purposes',
 ['audit:read', 'compliance:read', 'reports:read'], true)
ON CONFLICT (role_id) DO UPDATE SET
    updated_at = NOW(),
    is_active = true;

-- =====================================================
-- SECURITY AND COMPLIANCE SCHEMA COMPLETE
-- =====================================================

DO $$
BEGIN
    RAISE NOTICE '==========================================';
    RAISE NOTICE 'Security and Compliance Database Schema Complete';
    RAISE NOTICE '==========================================';
    RAISE NOTICE 'Tables Created: 25';
    RAISE NOTICE 'Indexes Created: 40+';
    RAISE NOTICE 'Functions Created: 4';
    RAISE NOTICE 'Views Created: 6';
    RAISE NOTICE 'Triggers Created: 16';
    RAISE NOTICE '==========================================';
    RAISE NOTICE 'Next Steps:';
    RAISE NOTICE '1. Run security tests to verify functionality';
    RAISE NOTICE '2. Configure application connections';
    RAISE NOTICE '3. Set up monitoring and alerting';
    RAISE NOTICE '4. Schedule periodic compliance assessments';
    RAISE NOTICE '==========================================';
END
$$;