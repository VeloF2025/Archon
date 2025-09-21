/**
 * Security Components Index
 *
 * Central export point for all security-related UI components
 */

export { SecurityDashboard } from './SecurityDashboard';
export { AccessControlManager } from './AccessControlManager';

// Security utility types and interfaces
export interface SecurityMetrics {
  security_framework: {
    active_sessions: number;
    failed_authentications: number;
    compliance_score: number;
  };
  zero_trust: {
    verified_entities: number;
    high_risk_entities: number;
    trust_level_distribution: Record<string, number>;
  };
  threat_detection: {
    threats_detected_24h: number;
    high_severity_threats: number;
    blocked_attempts: number;
    false_positives: number;
  };
  encryption: {
    keys_managed: number;
    encryption_operations: number;
    key_rotations: number;
  };
  audit: {
    audit_events_24h: number;
    integrity_verified: boolean;
    last_verification: string;
  };
  access_control: {
    total_subjects: number;
    total_roles: number;
    total_permissions: number;
    access_grants_24h: number;
  };
}

export interface ThreatEvent {
  id: string;
  threat_type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  confidence: number;
  timestamp: string;
  source_ip: string;
  target_resource: string;
  description: string;
  status: 'active' | 'mitigated' | 'investigating';
}

export interface SystemHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  components: Record<string, string>;
  metrics: Record<string, number>;
}

// Security utility functions
export const getSeverityColor = (severity: string): string => {
  switch (severity) {
    case 'critical': return 'bg-red-500';
    case 'high': return 'bg-red-400';
    case 'medium': return 'bg-yellow-400';
    case 'low': return 'bg-green-400';
    default: return 'bg-gray-400';
  }
};

export const getStatusColor = (status: string): string => {
  switch (status) {
    case 'healthy': return 'text-green-600';
    case 'degraded': return 'text-yellow-600';
    case 'unhealthy': return 'text-red-600';
    default: return 'text-gray-600';
  }
};

export const formatTimestamp = (timestamp: string): string => {
  return new Date(timestamp).toLocaleString();
};

export const calculateComplianceScore = (compliant: number, total: number): number => {
  return total > 0 ? Math.round((compliant / total) * 100) : 0;
};