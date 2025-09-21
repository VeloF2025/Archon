/**
 * Security Dashboard Component
 *
 * Comprehensive security monitoring and management dashboard
 * Provides real-time security metrics, threat detection, and system health monitoring
 */

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Shield, AlertTriangle, CheckCircle, XCircle, Activity, Lock, Users, Database, Network } from 'lucide-react';

interface SecurityMetrics {
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

interface ThreatEvent {
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

interface SystemHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  components: Record<string, string>;
  metrics: Record<string, number>;
}

export const SecurityDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<SecurityMetrics | null>(null);
  const [threats, setThreats] = useState<ThreatEvent[]>([]);
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedTab, setSelectedTab] = useState('overview');

  useEffect(() => {
    loadSecurityData();
    const interval = setInterval(loadSecurityData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const loadSecurityData = async () => {
    try {
      const [metricsRes, threatsRes, healthRes] = await Promise.all([
        fetch('/api/security/metrics'),
        fetch('/api/security/threat/recent?hours=24'),
        fetch('/api/security/health')
      ]);

      if (metricsRes.ok) {
        const metricsData = await metricsRes.json();
        setMetrics(metricsData);
      }

      if (threatsRes.ok) {
        const threatsData = await threatsRes.json();
        setThreats(threatsData.threats || []);
      }

      if (healthRes.ok) {
        const healthData = await healthRes.json();
        setHealth(healthData);
      }
    } catch (error) {
      console.error('Error loading security data:', error);
    } finally {
      setLoading(false);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-500';
      case 'high': return 'bg-red-400';
      case 'medium': return 'bg-yellow-400';
      case 'low': return 'bg-green-400';
      default: return 'bg-gray-400';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-600';
      case 'degraded': return 'text-yellow-600';
      case 'unhealthy': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Security Dashboard</h1>
          <p className="text-gray-600">Real-time security monitoring and threat detection</p>
        </div>
        <div className="flex items-center space-x-2">
          <Badge variant={health?.status === 'healthy' ? 'default' : 'destructive'}>
            {health?.status || 'unknown'}
          </Badge>
          <Button onClick={loadSecurityData} variant="outline">
            Refresh
          </Button>
        </div>
      </div>

      {/* System Health Alert */}
      {health?.status !== 'healthy' && (
        <Alert>
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>System Status: {health?.status}</AlertTitle>
          <AlertDescription>
            Some security components may require attention. Check component health below.
          </AlertDescription>
        </Alert>
      )}

      <Tabs value={selectedTab} onValueChange={setSelectedTab} className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="threats">Threat Detection</TabsTrigger>
          <TabsTrigger value="access">Access Control</TabsTrigger>
          <TabsTrigger value="audit">Audit & Compliance</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          {/* Key Metrics Row */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Active Sessions</CardTitle>
                <Users className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{metrics?.security_framework.active_sessions || 0}</div>
                <p className="text-xs text-muted-foreground">
                  Failed auths: {metrics?.security_framework.failed_authentications || 0}
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Threats (24h)</CardTitle>
                <Shield className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{metrics?.threat_detection.threats_detected_24h || 0}</div>
                <p className="text-xs text-muted-foreground">
                  High severity: {metrics?.threat_detection.high_severity_threats || 0}
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Compliance Score</CardTitle>
                <CheckCircle className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{metrics?.security_framework.compliance_score || 0}%</div>
                <Progress value={metrics?.security_framework.compliance_score || 0} className="mt-2" />
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Encryption Keys</CardTitle>
                <Lock className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{metrics?.encryption.keys_managed || 0}</div>
                <p className="text-xs text-muted-foreground">
                  Operations: {metrics?.encryption.encryption_operations || 0}
                </p>
              </CardContent>
            </Card>
          </div>

          {/* Component Health */}
          <Card>
            <CardHeader>
              <CardTitle>Component Health</CardTitle>
              <CardDescription>Status of security system components</CardDescription>
            </CardHeader>
            <CardContent>
              {health?.components && (
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                  {Object.entries(health.components).map(([component, status]) => (
                    <div key={component} className="flex items-center space-x-2">
                      <div className={`w-2 h-2 rounded-full ${
                        status === 'operational' ? 'bg-green-500' :
                        status === 'degraded' ? 'bg-yellow-500' : 'bg-red-500'
                      }`} />
                      <span className="text-sm capitalize">{component.replace('_', ' ')}</span>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="threats" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Recent Threat Events</CardTitle>
              <CardDescription>Threats detected in the last 24 hours</CardDescription>
            </CardHeader>
            <CardContent>
              {threats.length === 0 ? (
                <div className="text-center py-8">
                  <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-4" />
                  <p className="text-gray-600">No threats detected</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {threats.map((threat) => (
                    <div key={threat.id} className="flex items-center justify-between p-4 border rounded-lg">
                      <div className="flex items-center space-x-4">
                        <div className={`w-3 h-3 rounded-full ${getSeverityColor(threat.severity)}`} />
                        <div>
                          <p className="font-medium">{threat.threat_type}</p>
                          <p className="text-sm text-gray-600">{threat.description}</p>
                          <p className="text-xs text-gray-500">
                            {new Date(threat.timestamp).toLocaleString()}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Badge variant={threat.status === 'mitigated' ? 'default' : 'secondary'}>
                          {threat.status}
                        </Badge>
                        <Badge variant="outline">
                          {threat.confidence}% confidence
                        </Badge>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Threat Detection Metrics</CardTitle>
            </CardHeader>
            <CardContent>
              {metrics?.threat_detection && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-red-600">
                      {metrics.threat_detection.high_severity_threats}
                    </div>
                    <p className="text-sm text-gray-600">High Severity</p>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-yellow-600">
                      {metrics.threat_detection.threats_detected_24h}
                    </div>
                    <p className="text-sm text-gray-600">Total Threats</p>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600">
                      {metrics.threat_detection.blocked_attempts}
                    </div>
                    <p className="text-sm text-gray-600">Blocked Attempts</p>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">
                      {metrics.threat_detection.false_positives}
                    </div>
                    <p className="text-sm text-gray-600">False Positives</p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="access" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Access Control Summary</CardTitle>
              </CardHeader>
              <CardContent>
                {metrics?.access_control && (
                  <div className="space-y-4">
                    <div className="flex justify-between">
                      <span>Total Subjects</span>
                      <span className="font-medium">{metrics.access_control.total_subjects}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Total Roles</span>
                      <span className="font-medium">{metrics.access_control.total_roles}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Total Permissions</span>
                      <span className="font-medium">{metrics.access_control.total_permissions}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Access Grants (24h)</span>
                      <span className="font-medium">{metrics.access_control.access_grants_24h}</span>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Zero Trust Model</CardTitle>
              </CardHeader>
              <CardContent>
                {metrics?.zero_trust && (
                  <div className="space-y-4">
                    <div className="flex justify-between">
                      <span>Verified Entities</span>
                      <span className="font-medium">{metrics.zero_trust.verified_entities}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>High Risk Entities</span>
                      <span className="font-medium text-red-600">{metrics.zero_trust.high_risk_entities}</span>
                    </div>
                    <div>
                      <p className="text-sm font-medium mb-2">Trust Level Distribution</p>
                      {Object.entries(metrics.zero_trust.trust_level_distribution).map(([level, count]) => (
                        <div key={level} className="flex justify-between text-sm">
                          <span className="capitalize">{level}</span>
                          <span>{count}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="audit" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Audit & Compliance</CardTitle>
            </CardHeader>
            <CardContent>
              {metrics?.audit && (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold">{metrics.audit.audit_events_24h}</div>
                    <p className="text-sm text-gray-600">Audit Events (24h)</p>
                  </div>
                  <div className="text-center">
                    <div className={`text-2xl font-bold ${metrics.audit.integrity_verified ? 'text-green-600' : 'text-red-600'}`}>
                      {metrics.audit.integrity_verified ? 'Verified' : 'Failed'}
                    </div>
                    <p className="text-sm text-gray-600">Integrity Status</p>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold">
                      {new Date(metrics.audit.last_verification).toLocaleDateString()}
                    </div>
                    <p className="text-sm text-gray-600">Last Verification</p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Encryption Status</CardTitle>
              </CardHeader>
              <CardContent>
                {metrics?.encryption && (
                  <div className="space-y-4">
                    <div className="flex justify-between">
                      <span>Keys Managed</span>
                      <span className="font-medium">{metrics.encryption.keys_managed}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Operations (24h)</span>
                      <span className="font-medium">{metrics.encryption.encryption_operations}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Key Rotations</span>
                      <span className="font-medium">{metrics.encryption.key_rotations}</span>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Quick Actions</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <Button className="w-full" variant="outline">
                  Generate Security Report
                </Button>
                <Button className="w-full" variant="outline">
                  Export Audit Logs
                </Button>
                <Button className="w-full" variant="outline">
                  Run Security Scan
                </Button>
                <Button className="w-full" variant="outline">
                  View Compliance Status
                </Button>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};