/**
 * Compliance Dashboard Component
 *
 * Comprehensive compliance management dashboard for GDPR, SOC2, and HIPAA
 * Provides compliance monitoring, assessment tools, and reporting capabilities
 */

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { FileText, CheckCircle, AlertTriangle, XCircle, Clock, Download, Eye, Settings } from 'lucide-react';

interface ComplianceStatus {
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

interface ComplianceReport {
  id: string;
  framework: string;
  report_type: string;
  generated_at: string;
  compliance_score: number;
  status: string;
  download_url?: string;
}

interface ComplianceAssessment {
  id: string;
  framework: string;
  overall_status: string;
  compliance_score: number;
  findings: number;
  deficiencies: number;
  assessment_date: string;
}

export const ComplianceDashboard: React.FC = () => {
  const [status, setStatus] = useState<ComplianceStatus | null>(null);
  const [reports, setReports] = useState<ComplianceReport[]>([]);
  const [assessments, setAssessments] = useState<ComplianceAssessment[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedFramework, setSelectedFramework] = useState('all');

  useEffect(() => {
    loadComplianceData();
  }, []);

  const loadComplianceData = async () => {
    try {
      // Load compliance status for each framework
      const [gdprStatus, soc2Status, hipaaStatus] = await Promise.all([
        fetch('/api/compliance/gdpr/assessment'),
        fetch('/api/compliance/soc2/assessment'),
        fetch('/api/compliance/hipaa/assessment')
      ]);

      const complianceData: ComplianceStatus = {
        gdpr: gdprStatus.ok ? await gdprStatus.json() : null,
        soc2: soc2Status.ok ? await soc2Status.json() : null,
        hipaa: hipaaStatus.ok ? await hipaaStatus.json() : null
      };

      setStatus(complianceData);

      // Load recent reports and assessments
      const [reportsRes, assessmentsRes] = await Promise.all([
        fetch('/api/compliance/reports'),
        fetch('/api/compliance/assessments')
      ]);

      if (reportsRes.ok) {
        const reportsData = await reportsRes.json();
        setReports(reportsData.reports || []);
      }

      if (assessmentsRes.ok) {
        const assessmentsData = await assessmentsRes.json();
        setAssessments(assessmentsData.assessments || []);
      }
    } catch (error) {
      console.error('Error loading compliance data:', error);
    } finally {
      setLoading(false);
    }
  };

  const runAssessment = async (framework: string) => {
    try {
      const response = await fetch(`/api/compliance/${framework}/assessment`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ framework, include_recommendations: true })
      });

      if (response.ok) {
        const assessment = await response.json();
        setAssessments(prev => [assessment, ...prev]);
        loadComplianceData(); // Refresh status
      }
    } catch (error) {
      console.error('Error running assessment:', error);
    }
  };

  const generateReport = async (framework: string, reportType: string) => {
    try {
      const response = await fetch('/api/compliance/reports', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          framework,
          report_type: reportType,
          format: 'pdf'
        })
      });

      if (response.ok) {
        const report = await response.json();
        setReports(prev => [report, ...prev]);
      }
    } catch (error) {
      console.error('Error generating report:', error);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'compliant': return 'text-green-600';
      case 'partially_compliant': return 'text-yellow-600';
      case 'non_compliant': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'compliant': return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'partially_compliant': return <AlertTriangle className="h-4 w-4 text-yellow-600" />;
      case 'non_compliant': return <XCircle className="h-4 w-4 text-red-600" />;
      default: return <Clock className="h-4 w-4 text-gray-600" />;
    }
  };

  const getFrameworkStatus = (framework: string) => {
    if (!status || !status[framework]) return null;
    return status[framework].overall_status;
  };

  const getFrameworkScore = (framework: string) => {
    if (!status || !status[framework]) return 0;
    return status[framework].compliance_score;
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
          <h1 className="text-3xl font-bold text-gray-900">Compliance Dashboard</h1>
          <p className="text-gray-600">GDPR, SOC2, and HIPAA compliance management</p>
        </div>
        <Button onClick={loadComplianceData} variant="outline">
          Refresh Data
        </Button>
      </div>

      {/* Framework Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {['gdpr', 'soc2', 'hipaa'].map((framework) => (
          <Card key={framework}>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span className="uppercase">{framework}</span>
                {getStatusIcon(getFrameworkStatus(framework))}
              </CardTitle>
              <CardDescription>Compliance Status</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium">Compliance Score</span>
                    <span className="text-sm font-bold">{getFrameworkScore(framework)}%</span>
                  </div>
                  <Progress value={getFrameworkScore(framework)} className="h-2" />
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm">Status</span>
                  <Badge
                    variant={getFrameworkStatus(framework) === 'compliant' ? 'default' : 'secondary'}
                    className={getStatusColor(getFrameworkStatus(framework))}
                  >
                    {getFrameworkStatus(framework)?.replace('_', ' ')}
                  </Badge>
                </div>

                {framework === 'gdpr' && status?.gdpr && (
                  <div className="text-xs text-gray-600">
                    <p>DSARs: {status.gdpr.data_subject_requests}</p>
                    <p>Breaches: {status.gdpr.data_breaches}</p>
                  </div>
                )}

                {framework === 'soc2' && status?.soc2 && (
                  <div className="text-xs text-gray-600">
                    <p>Controls: {status.soc2.controls_implemented}/{status.soc2.total_controls}</p>
                    <p>Last Audit: {new Date(status.soc2.last_audit_date).toLocaleDateString()}</p>
                  </div>
                )}

                {framework === 'hipaa' && status?.hipaa && (
                  <div className="text-xs text-gray-600">
                    <p>BAs: {status.hipaa.business_associates}</p>
                    <p>Training: {status.hipaa.training_compliance}%</p>
                    <p>Risk Score: {status.hipaa.risk_score}</p>
                  </div>
                )}

                <div className="flex space-x-2">
                  <Button
                    size="sm"
                    onClick={() => runAssessment(framework)}
                    className="flex-1"
                  >
                    Run Assessment
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => generateReport(framework, 'assessment')}
                  >
                    <Download className="h-3 w-3" />
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <Tabs defaultValue="assessments" className="space-y-4">
        <TabsList>
          <TabsTrigger value="assessments">Recent Assessments</TabsTrigger>
          <TabsTrigger value="reports">Reports</TabsTrigger>
          <TabsTrigger value="actions">Quick Actions</TabsTrigger>
        </TabsList>

        <TabsContent value="assessments">
          <Card>
            <CardHeader>
              <CardTitle>Recent Compliance Assessments</CardTitle>
              <CardDescription>Latest compliance assessment results</CardDescription>
            </CardHeader>
            <CardContent>
              {assessments.length === 0 ? (
                <div className="text-center py-8">
                  <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-600">No assessments completed yet</p>
                  <p className="text-sm text-gray-500">Run your first compliance assessment above</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {assessments.map((assessment) => (
                    <div key={assessment.id} className="flex items-center justify-between p-4 border rounded-lg">
                      <div className="flex items-center space-x-4">
                        {getStatusIcon(assessment.overall_status)}
                        <div>
                          <p className="font-medium capitalize">{assessment.framework}</p>
                          <p className="text-sm text-gray-600">
                            {assessment.overall_status.replace('_', ' ')} - {assessment.compliance_score}%
                          </p>
                          <p className="text-xs text-gray-500">
                            {new Date(assessment.assessment_date).toLocaleString()}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Badge variant="outline">
                          {assessment.findings} findings
                        </Badge>
                        <Badge variant={assessment.deficiencies > 0 ? 'destructive' : 'secondary'}>
                          {assessment.deficiencies} deficiencies
                        </Badge>
                        <Button size="sm" variant="outline">
                          <Eye className="h-3 w-3 mr-1" />
                          View
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="reports">
          <Card>
            <CardHeader>
              <CardTitle>Compliance Reports</CardTitle>
              <CardDescription>Generated compliance reports and documentation</CardDescription>
            </CardHeader>
            <CardContent>
              {reports.length === 0 ? (
                <div className="text-center py-8">
                  <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-600">No reports generated yet</p>
                  <p className="text-sm text-gray-500">Generate your first compliance report</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {reports.map((report) => (
                    <div key={report.id} className="flex items-center justify-between p-4 border rounded-lg">
                      <div>
                        <p className="font-medium capitalize">{report.framework} - {report.report_type}</p>
                        <p className="text-sm text-gray-600">
                          {report.compliance_score}% compliant
                        </p>
                        <p className="text-xs text-gray-500">
                          {new Date(report.generated_at).toLocaleString()}
                        </p>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Badge variant={report.status === 'completed' ? 'default' : 'secondary'}>
                          {report.status}
                        </Badge>
                        {report.download_url && (
                          <Button size="sm" variant="outline">
                            <Download className="h-3 w-3 mr-1" />
                            Download
                          </Button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="actions">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">GDPR Actions</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <Button className="w-full" variant="outline" onClick={() => runAssessment('gdpr')}>
                  Run GDPR Assessment
                </Button>
                <Button className="w-full" variant="outline" onClick={() => generateReport('gdpr', 'detailed')}>
                  Generate GDPR Report
                </Button>
                <Button className="w-full" variant="outline">
                  Manage DSARs
                </Button>
                <Button className="w-full" variant="outline">
                  Update Data Mapping
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">SOC2 Actions</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <Button className="w-full" variant="outline" onClick={() => runAssessment('soc2')}>
                  Run SOC2 Assessment
                </Button>
                <Button className="w-full" variant="outline" onClick={() => generateReport('soc2', 'detailed')}>
                  Generate SOC2 Report
                </Button>
                <Button className="w-full" variant="outline">
                  Test Controls
                </Button>
                <Button className="w-full" variant="outline">
                  Review Evidence
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">HIPAA Actions</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <Button className="w-full" variant="outline" onClick={() => runAssessment('hipaa')}>
                  Run HIPAA Assessment
                </Button>
                <Button className="w-full" variant="outline" onClick={() => generateReport('hipaa', 'detailed')}>
                  Generate HIPAA Report
                </Button>
                <Button className="w-full" variant="outline">
                  Manage BAs
                </Button>
                <Button className="w-full" variant="outline">
                  Training Records
                </Button>
              </CardContent>
            </Card>
          </div>

          <Card className="mt-6">
            <CardHeader>
              <CardTitle>Compliance Settings</CardTitle>
              <CardDescription>Configure compliance management preferences</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-medium mb-2">Assessment Frequency</h4>
                  <p className="text-sm text-gray-600">How often to run automatic assessments</p>
                </div>
                <div>
                  <h4 className="font-medium mb-2">Report Retention</h4>
                  <p className="text-sm text-gray-600">How long to keep compliance reports</p>
                </div>
                <div>
                  <h4 className="font-medium mb-2">Alert Thresholds</h4>
                  <p className="text-sm text-gray-600">Compliance score alert thresholds</p>
                </div>
                <div>
                  <h4 className="font-medium mb-2">Integration Settings</h4>
                  <p className="text-sm text-gray-600">External compliance tool integrations</p>
                </div>
              </div>
              <div className="mt-4">
                <Button variant="outline">
                  <Settings className="h-4 w-4 mr-2" />
                  Configure Settings
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};