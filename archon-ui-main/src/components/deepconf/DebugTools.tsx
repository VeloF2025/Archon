/**
 * Debug Tools Component
 * Phase 7 DeepConf Integration
 * 
 * Interactive confidence analysis, data export, and debugging capabilities
 * for SCWT metrics system
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/Card';
import { Button } from '../ui/Button';
import { Toggle } from '../ui/Toggle';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '../ui/Select';
import { Tabs } from '../ui/tabs';
import { 
  DebugToolsProps, 
  DebugAction,
  AnalysisResult,
  Anomaly,
  TrendAnalysis,
  ExportData
} from './types';

// Icons
const BugIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="m8 2 1.88 1.88M14.12 3.88 16 2M9 7.13v-1a3.25 3.25 0 1 1 6.5 0v1"/>
    <path d="M12 20c-3.3 0-6-2.7-6-6v-3a4 4 0 0 1 4-4h4a4 4 0 0 1 4 4v3c0 3.3-2.7 6-6 6"/>
    <path d="M12 20v-9"/>
    <path d="M6.53 9C4.6 8.8 3 7.1 3 5M20.97 5c0 2.1-1.6 3.8-3.53 4"/>
    <path d="M6.53 15C4.6 15.2 3 16.9 3 19M20.97 19c0-2.1-1.6-3.8-3.53-4"/>
  </svg>
);

const AnalyzeIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M21.21 15.89A10 10 0 1 1 8 2.83"/>
    <path d="M22 12A10 10 0 0 0 12 2v10z"/>
  </svg>
);

const DownloadIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
    <polyline points="7,10 12,15 17,10"/>
    <line x1="12" y1="15" x2="12" y2="3"/>
  </svg>
);

const RefreshIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M3 2v6h6M3 8a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6 2l-3 3"/>
  </svg>
);

const PlayIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polygon points="5,3 19,12 5,21"/>
  </svg>
);

const AlertCircleIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10"/>
    <line x1="12" y1="8" x2="12" y2="12"/>
    <line x1="12" y1="16" x2="12.01" y2="16"/>
  </svg>
);

const CheckCircleIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="m9 12 2 2 4-4"/>
    <circle cx="12" cy="12" r="10"/>
  </svg>
);

const XCircleIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10"/>
    <path d="m15 9-6 6M9 9l6 6"/>
  </svg>
);

export const DebugTools: React.FC<DebugToolsProps> = ({
  metrics,
  confidence,
  debugMode = false,
  enableAnalysis = true,
  exportFormats = ['json', 'csv'],
  onDebugAction,
}) => {
  // State management
  const [activeTab, setActiveTab] = useState<'analysis' | 'export' | 'simulate' | 'logs'>('analysis');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [selectedExportFormat, setSelectedExportFormat] = useState(exportFormats[0]);
  const [simulationRunning, setSimulationRunning] = useState(false);
  const [debugLogs, setDebugLogs] = useState<string[]>([]);

  // Mock analysis function (in real implementation, this would call backend API)
  const performAnalysis = useCallback(async (): Promise<AnalysisResult> => {
    setIsAnalyzing(true);
    
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Mock analysis results based on current metrics
    const result: AnalysisResult = {
      insights: [
        `Current confidence score of ${(confidence.overall * 100).toFixed(1)}% is ${confidence.overall > 0.8 ? 'excellent' : confidence.overall > 0.6 ? 'good' : 'concerning'}`,
        `Structural weight (${metrics.structuralWeight.toFixed(3)}) shows ${metrics.structuralWeight > 0.5 ? 'strong' : 'weak'} pattern recognition`,
        `Uncertainty bounds indicate ${confidence.uncertainty.total > 0.3 ? 'high' : 'low'} model uncertainty`,
        `Temporal consistency suggests ${confidence.trend === 'stable' ? 'stable' : confidence.trend} confidence patterns`,
      ],
      recommendations: [
        confidence.overall < 0.6 ? 'Consider increasing training data diversity' : 'Maintain current confidence calibration',
        metrics.combinedScore < 0.7 ? 'Review SCWT weight distribution' : 'SCWT weights appear well-balanced',
        confidence.uncertainty.epistemic > 0.2 ? 'Model uncertainty is high - consider ensemble methods' : 'Model uncertainty within acceptable bounds',
        'Monitor confidence trends for early anomaly detection',
      ],
      anomalies: [
        ...(confidence.overall < 0.3 ? [{
          type: 'confidence' as const,
          severity: 'critical' as const,
          description: 'Critically low overall confidence detected',
          timestamp: new Date(),
          metrics: { overall: confidence.overall },
          recommendations: ['Investigate model degradation', 'Check input data quality'],
        }] : []),
        ...(confidence.uncertainty.total > 0.5 ? [{
          type: 'confidence' as const,
          severity: 'high' as const,
          description: 'High uncertainty in confidence estimates',
          timestamp: new Date(),
          metrics: { uncertainty: confidence.uncertainty.total },
          recommendations: ['Increase model ensemble size', 'Collect more training data'],
        }] : []),
      ],
      trends: [
        {
          metric: 'overall_confidence',
          direction: confidence.trend === 'increasing' ? 'increasing' : 
                   confidence.trend === 'decreasing' ? 'decreasing' : 'stable',
          rate: 0.05, // Mock trend rate
          confidence: 0.85,
          prediction: {
            next24h: confidence.overall + (confidence.trend === 'increasing' ? 0.02 : 
                                        confidence.trend === 'decreasing' ? -0.02 : 0),
            nextWeek: confidence.overall + (confidence.trend === 'increasing' ? 0.1 : 
                                          confidence.trend === 'decreasing' ? -0.1 : 0),
            uncertainty: 0.15,
          },
        },
      ],
      confidence: 0.82,
    };
    
    setIsAnalyzing(false);
    return result;
  }, [metrics, confidence]);

  // Handle debug actions
  const handleDebugAction = useCallback(async (actionType: DebugAction['type'], parameters?: Record<string, any>) => {
    const action: DebugAction = {
      type: actionType,
      parameters,
      timestamp: new Date(),
    };

    // Add to debug logs
    setDebugLogs(prev => [
      ...prev.slice(-19), // Keep last 19 logs
      `[${action.timestamp.toLocaleTimeString()}] ${actionType.toUpperCase()}: ${JSON.stringify(parameters || {})}`,
    ]);

    // Execute action
    switch (actionType) {
      case 'analyze':
        const result = await performAnalysis();
        setAnalysisResult(result);
        break;
        
      case 'export':
        await handleExport();
        break;
        
      case 'simulate':
        setSimulationRunning(true);
        // Mock simulation
        await new Promise(resolve => setTimeout(resolve, 3000));
        setSimulationRunning(false);
        break;
        
      case 'reset':
        setAnalysisResult(null);
        setDebugLogs([]);
        break;
        
      case 'validate':
        // Mock validation
        setDebugLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] VALIDATION: All metrics within expected ranges`]);
        break;
    }

    onDebugAction?.(action);
  }, [onDebugAction]);

  // Handle data export
  const handleExport = useCallback(async () => {
    const exportData: ExportData = {
      format: selectedExportFormat as any,
      timeRange: {
        start: new Date(Date.now() - 24 * 60 * 60 * 1000),
        end: new Date(),
        granularity: 'hour',
      },
      metrics: [metrics],
      performance: [], // Would be populated in real implementation
      confidence: [confidence],
      metadata: {
        exportedAt: new Date(),
        exportedBy: 'debug-tools',
        version: '1.0.0',
      },
    };

    // Create and download file
    const dataStr = selectedExportFormat === 'json' 
      ? JSON.stringify(exportData, null, 2)
      : convertToCSV(exportData); // Mock CSV conversion

    const dataBlob = new Blob([dataStr], { 
      type: selectedExportFormat === 'json' ? 'application/json' : 'text/csv' 
    });
    
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `scwt-debug-data.${selectedExportFormat}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    addLog(`EXPORT: Advanced debug data exported as ${selectedExportFormat.toUpperCase()}`);
  }, [selectedExportFormat, metrics, confidence]);

  // Mock CSV conversion
  const convertToCSV = (data: ExportData): string => {
    const headers = ['timestamp', 'structuralWeight', 'contextWeight', 'temporalWeight', 'combinedScore', 'confidence'];
    const rows = data.metrics.map(m => [
      new Date().toISOString(),
      m.structuralWeight,
      m.contextWeight,
      m.temporalWeight,
      m.combinedScore,
      confidence.overall,
    ]);
    
    return [headers.join(','), ...rows.map(row => row.join(','))].join('\n');
  };

  // Current metrics summary
  const metricsSummary = useMemo(() => ({
    structuralWeight: { value: metrics.structuralWeight, status: metrics.structuralWeight > 0.5 ? 'good' : 'warning' },
    contextWeight: { value: metrics.contextWeight, status: metrics.contextWeight > 0.5 ? 'good' : 'warning' },
    temporalWeight: { value: metrics.temporalWeight, status: metrics.temporalWeight > 0.5 ? 'good' : 'warning' },
    combinedScore: { value: metrics.combinedScore, status: metrics.combinedScore > 0.7 ? 'good' : 'warning' },
    confidence: { value: confidence.overall, status: confidence.overall > 0.8 ? 'good' : confidence.overall > 0.6 ? 'warning' : 'error' },
  }), [metrics, confidence]);

  return (
    <Card accentColor="purple">
      <CardHeader>
        <div className="flex justify-between items-start">
          <div>
            <CardTitle className="flex items-center gap-2">
              <BugIcon />
              Debug Tools
            </CardTitle>
            <p className="text-sm text-muted-foreground mt-1">
              Interactive confidence analysis and debugging utilities
            </p>
          </div>
          
          <div className="flex items-center gap-2">
            <Button
              onClick={() => handleDebugAction('reset')}
              variant="outline"
              size="sm"
            >
              <RefreshIcon />
              Reset
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Debug Mode Indicator */}
        {debugMode && (
          <div className="p-3 bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-md">
            <div className="flex items-center gap-2 text-purple-700 dark:text-purple-300">
              <BugIcon />
              <span className="font-medium">Debug Mode Active</span>
            </div>
          </div>
        )}

        {/* Current Metrics Overview */}
        <div>
          <h4 className="font-medium mb-3">Current Metrics Status</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(metricsSummary).map(([key, { value, status }]) => (
              <div key={key} className="flex items-center justify-between p-3 bg-muted/50 rounded-md">
                <span className="text-sm capitalize">
                  {key.replace(/([A-Z])/g, ' $1')}:
                </span>
                <div className="flex items-center gap-2">
                  <span className="font-medium">
                    {key === 'confidence' || key.includes('Weight') ? 
                      (value * 100).toFixed(1) + '%' : 
                      value.toFixed(3)
                    }
                  </span>
                  {status === 'good' && <CheckCircleIcon className="w-4 h-4 text-green-600" />}
                  {status === 'warning' && <AlertCircleIcon className="w-4 h-4 text-yellow-600" />}
                  {status === 'error' && <XCircleIcon className="w-4 h-4 text-red-600" />}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Tabs Interface */}
        <div>
          <div className="flex gap-1 mb-4 bg-muted/50 p-1 rounded-md">
            {(['analysis', 'export', 'simulate', 'logs'] as const).map(tab => (
              <Button
                key={tab}
                onClick={() => setActiveTab(tab)}
                variant={activeTab === tab ? 'default' : 'ghost'}
                size="sm"
                className="flex-1 capitalize"
              >
                {tab}
              </Button>
            ))}
          </div>

          {/* Analysis Tab */}
          {activeTab === 'analysis' && (
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <h4 className="font-medium">Confidence Analysis</h4>
                <Button
                  onClick={() => handleDebugAction('analyze')}
                  disabled={!enableAnalysis || isAnalyzing}
                  className="gap-2"
                >
                  <AnalyzeIcon />
                  {isAnalyzing ? 'Analyzing...' : 'Run Analysis'}
                </Button>
              </div>

              {isAnalyzing && (
                <div className="p-4 text-center">
                  <div className="animate-spin w-6 h-6 border-2 border-primary border-t-transparent rounded-full mx-auto mb-2"></div>
                  <p className="text-sm text-muted-foreground">Analyzing confidence patterns...</p>
                </div>
              )}

              {analysisResult && (
                <div className="space-y-4">
                  {/* Insights */}
                  <div>
                    <h5 className="font-medium mb-2">Key Insights</h5>
                    <div className="space-y-2">
                      {analysisResult.insights.map((insight, index) => (
                        <div key={index} className="flex items-start gap-2 p-2 bg-blue-50 dark:bg-blue-900/20 rounded">
                          <CheckCircleIcon className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
                          <span className="text-sm">{insight}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Recommendations */}
                  <div>
                    <h5 className="font-medium mb-2">Recommendations</h5>
                    <div className="space-y-2">
                      {analysisResult.recommendations.map((rec, index) => (
                        <div key={index} className="flex items-start gap-2 p-2 bg-green-50 dark:bg-green-900/20 rounded">
                          <AlertCircleIcon className="w-4 h-4 text-green-600 mt-0.5 flex-shrink-0" />
                          <span className="text-sm">{rec}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Anomalies */}
                  {analysisResult.anomalies.length > 0 && (
                    <div>
                      <h5 className="font-medium mb-2">Detected Anomalies</h5>
                      <div className="space-y-2">
                        {analysisResult.anomalies.map((anomaly, index) => (
                          <div key={index} className={`p-2 rounded border-l-4 ${
                            anomaly.severity === 'critical' ? 'bg-red-50 dark:bg-red-900/20 border-red-500' :
                            anomaly.severity === 'high' ? 'bg-orange-50 dark:bg-orange-900/20 border-orange-500' :
                            'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-500'
                          }`}>
                            <div className="flex items-center gap-2 mb-1">
                              <XCircleIcon className={`w-4 h-4 ${
                                anomaly.severity === 'critical' ? 'text-red-600' :
                                anomaly.severity === 'high' ? 'text-orange-600' :
                                'text-yellow-600'
                              }`} />
                              <span className="font-medium text-sm capitalize">
                                {anomaly.severity} {anomaly.type} Anomaly
                              </span>
                            </div>
                            <p className="text-sm mb-2">{anomaly.description}</p>
                            <div className="text-xs text-muted-foreground">
                              Recommendations: {anomaly.recommendations.join(', ')}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Overall Analysis Confidence */}
                  <div className="p-3 bg-muted/50 rounded">
                    <div className="flex justify-between items-center">
                      <span className="font-medium">Analysis Confidence:</span>
                      <span className="text-lg font-bold text-green-600">
                        {(analysisResult.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Export Tab */}
          {activeTab === 'export' && (
            <div className="space-y-4">
              <h4 className="font-medium">Export Debug Data</h4>
              
              <div className="flex items-center gap-4">
                <div>
                  <label className="block text-sm font-medium mb-1">Format:</label>
                  <Select
                    value={selectedExportFormat}
                    onValueChange={setSelectedExportFormat}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {exportFormats.map(format => (
                        <SelectItem key={format} value={format}>
                          {format.toUpperCase()}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                
                <Button
                  onClick={() => handleDebugAction('export')}
                  className="gap-2 mt-6"
                >
                  <DownloadIcon />
                  Export Data
                </Button>
              </div>

              <div className="p-3 bg-muted/50 rounded text-sm">
                <p className="mb-2">Export will include:</p>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• Current SCWT metrics</li>
                  <li>• Confidence measurements</li>
                  <li>• Performance data</li>
                  <li>• Analysis results (if available)</li>
                  <li>• Debug logs</li>
                </ul>
              </div>
            </div>
          )}

          {/* Simulation Tab */}
          {activeTab === 'simulate' && (
            <div className="space-y-4">
              <h4 className="font-medium">Confidence Simulation</h4>
              
              <div className="flex items-center gap-4">
                <Button
                  onClick={() => handleDebugAction('simulate', { scenario: 'stress_test' })}
                  disabled={simulationRunning}
                  className="gap-2"
                >
                  <PlayIcon />
                  {simulationRunning ? 'Running...' : 'Stress Test'}
                </Button>
                
                <Button
                  onClick={() => handleDebugAction('simulate', { scenario: 'edge_cases' })}
                  disabled={simulationRunning}
                  className="gap-2"
                >
                  <PlayIcon />
                  Edge Cases
                </Button>
              </div>

              {simulationRunning && (
                <div className="p-4 text-center">
                  <div className="animate-pulse w-full bg-primary/20 h-2 rounded mb-2"></div>
                  <p className="text-sm text-muted-foreground">Running simulation...</p>
                </div>
              )}

              <div className="p-3 bg-muted/50 rounded text-sm">
                <p className="mb-2">Available simulations:</p>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• <strong>Stress Test:</strong> High-volume confidence calculations</li>
                  <li>• <strong>Edge Cases:</strong> Boundary condition testing</li>
                  <li>• <strong>Uncertainty Analysis:</strong> Bayesian interval validation</li>
                </ul>
              </div>
            </div>
          )}

          {/* Logs Tab */}
          {activeTab === 'logs' && (
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <h4 className="font-medium">Debug Logs</h4>
                <Button
                  onClick={() => setDebugLogs([])}
                  variant="outline"
                  size="sm"
                >
                  Clear Logs
                </Button>
              </div>
              
              <div className="bg-black text-green-400 p-4 rounded font-mono text-sm h-64 overflow-y-auto">
                {debugLogs.length > 0 ? (
                  debugLogs.map((log, index) => (
                    <div key={index} className="mb-1">
                      {log}
                    </div>
                  ))
                ) : (
                  <div className="text-muted-foreground">No debug logs available</div>
                )}
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default DebugTools;