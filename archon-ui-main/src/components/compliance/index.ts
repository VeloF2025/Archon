/**
 * Compliance Components Index
 *
 * Central export point for all compliance-related UI components
 */

export { ComplianceDashboard } from './ComplianceDashboard';

// Compliance utility types and interfaces
export interface ComplianceStatus {
  gdpr: {
    overall_status: string;
    compliance_score: number;
    data_subject_requests: number;
    data_breaches: number;
    last_assessment: string;
  };
  soc2: {
    overall_status: string;
    compliance_score: number;
    controls_implemented: number;
    total_controls: number;
    last_audit_date: string;
  };
  hipaa: {
    overall_status: string;
    compliance_score: number;
    business_associates: number;
    training_compliance: number;
    risk_score: number;
  };
}

export interface ComplianceReport {
  id: string;
  framework: string;
  report_type: string;
  generated_at: string;
  compliance_score: number;
  status: string;
  download_url?: string;
}

export interface ComplianceAssessment {
  id: string;
  framework: string;
  overall_status: string;
  compliance_score: number;
  findings: number;
  deficiencies: number;
  assessment_date: string;
}

export interface GDPRDataSubjectRequest {
  id: string;
  request_type: string;
  subject_id: string;
  status: string;
  created_at: string;
  estimated_completion_date: string;
  processing_steps: string[];
  requirements_met: string[];
}

export interface SOC2ControlTest {
  id: string;
  control_id: string;
  test_name: string;
  test_date: string;
  tester: string;
  result: string;
  findings: string[];
  evidence: string[];
  effectiveness: string;
}

export interface HIPAABusinessAssociate {
  id: string;
  name: string;
  services_provided: string[];
  data_access_level: string;
  compliance_status: string;
  risk_assessment_score: number;
  contract_date: string;
  contract_expiry: string;
  last_audit_date: string;
}

// Compliance utility functions
export const getComplianceStatusColor = (status: string): string => {
  switch (status) {
    case 'compliant': return 'text-green-600';
    case 'partially_compliant': return 'text-yellow-600';
    case 'non_compliant': return 'text-red-600';
    default: return 'text-gray-600';
  }
};

export const getComplianceStatusIcon = (status: string) => {
  switch (status) {
    case 'compliant': return 'CheckCircle';
    case 'partially_compliant': return 'AlertTriangle';
    case 'non_compliant': return 'XCircle';
    default: return 'Clock';
  }
};

export const formatComplianceScore = (score: number): string => {
  return `${score.toFixed(1)}%`;
};

export const getRiskLevelColor = (score: number): string => {
  if (score >= 80) return 'text-red-600';
  if (score >= 60) return 'text-yellow-600';
  return 'text-green-600';
};

export const getRiskLevel = (score: number): string => {
  if (score >= 80) return 'High';
  if (score >= 60) return 'Medium';
  return 'Low';
};

// Compliance framework constants
export const COMPLIANCE_FRAMEWORKS = {
  GDPR: 'gdpr',
  SOC2: 'soc2',
  HIPAA: 'hipaa'
} as const;

export const REPORT_TYPES = {
  ASSESSMENT: 'assessment',
  SUMMARY: 'summary',
  DETAILED: 'detailed',
  EXECUTIVE: 'executive'
} as const;

export const GDPR_REQUEST_TYPES = {
  ACCESS: 'access',
  RECTIFICATION: 'rectification',
  ERASURE: 'erasure',
  PORTABILITY: 'portability',
  OBJECTION: 'objection'
} as const;

export const SOC2_TRUST_SERVICES_CRITERIA = {
  SECURITY: 'security',
  AVAILABILITY: 'availability',
  PROCESSING_INTEGRITY: 'processing_integrity',
  CONFIDENTIALITY: 'confidentiality',
  PRIVACY: 'privacy'
} as const;