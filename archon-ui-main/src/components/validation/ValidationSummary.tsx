import React from 'react';
import { CheckCircle, XCircle, AlertTriangle, Clock, Info } from 'lucide-react';

interface ValidationResult {
  check_id: string;
  check_name: string;
  status: 'pass' | 'fail' | 'skip' | 'error';
  severity: 'info' | 'warning' | 'error' | 'critical';
  message: string;
  details?: Record<string, any>;
  execution_time: number;
  file_path?: string;
  line_number?: number;
}

interface ValidationVerdict {
  validation_id: string;
  timestamp: number;
  overall_status: 'pass' | 'fail' | 'skip' | 'error';
  total_checks: number;
  passed_checks: number;
  failed_checks: number;
  error_rate: number;
  false_positive_rate: number;
  results: ValidationResult[];
  metadata: Record<string, any>;
}

interface ValidationSummaryProps {
  verdict: ValidationVerdict | null;
  loading: boolean;
  onRetry?: () => void;
  onViewDetails?: (result: ValidationResult) => void;
}

export const ValidationSummary: React.FC<ValidationSummaryProps> = ({
  verdict,
  loading,
  onRetry,
  onViewDetails
}) => {
  const getStatusIcon = (status: ValidationResult['status']) => {
    switch (status) {
      case 'pass':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'fail':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'error':
        return <AlertTriangle className="h-4 w-4 text-orange-500" />;
      case 'skip':
        return <Info className="h-4 w-4 text-gray-400" />;
      default:
        return <Clock className="h-4 w-4 text-blue-500" />;
    }
  };

  const getSeverityColor = (severity: ValidationResult['severity']) => {
    switch (severity) {
      case 'critical':
        return 'text-red-700 bg-red-50 border-red-200';
      case 'error':
        return 'text-red-600 bg-red-50 border-red-200';
      case 'warning':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'info':
        return 'text-blue-600 bg-blue-50 border-blue-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getOverallStatusColor = (status: ValidationVerdict['overall_status']) => {
    switch (status) {
      case 'pass':
        return 'bg-green-500';
      case 'fail':
        return 'bg-red-500';
      case 'error':
        return 'bg-orange-500';
      default:
        return 'bg-gray-400';
    }
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <div className="flex items-center space-x-3">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
          <div className="text-gray-600">Running validation checks...</div>
        </div>
      </div>
    );
  }

  if (!verdict) {
    return (
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <div className="text-center text-gray-500">
          <Info className="h-12 w-12 mx-auto mb-4 text-gray-300" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Validation Results</h3>
          <p className="text-gray-500">Run validation to see results here.</p>
          {onRetry && (
            <button
              onClick={onRetry}
              className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
            >
              Run Validation
            </button>
          )}
        </div>
      </div>
    );
  }

  const successRate = (verdict.passed_checks / verdict.total_checks) * 100;

  return (
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className={`w-3 h-3 rounded-full ${getOverallStatusColor(verdict.overall_status)}`}></div>
            <h3 className="text-lg font-semibold text-gray-900">Validation Summary</h3>
          </div>
          <div className="flex items-center space-x-4 text-sm text-gray-500">
            <span>ID: {verdict.validation_id.slice(0, 8)}</span>
            <span>{new Date(verdict.timestamp * 1000).toLocaleString()}</span>
          </div>
        </div>
      </div>

      {/* Overall Stats */}
      <div className="px-6 py-4 bg-gray-50">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">{verdict.total_checks}</div>
            <div className="text-sm text-gray-500">Total Checks</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">{verdict.passed_checks}</div>
            <div className="text-sm text-gray-500">Passed</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-red-600">{verdict.failed_checks}</div>
            <div className="text-sm text-gray-500">Failed</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">{successRate.toFixed(1)}%</div>
            <div className="text-sm text-gray-500">Success Rate</div>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="mt-4">
          <div className="flex justify-between text-sm text-gray-500 mb-1">
            <span>Progress</span>
            <span>{successRate.toFixed(1)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className="bg-green-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${successRate}%` }}
            ></div>
          </div>
        </div>
      </div>

      {/* Results List */}
      <div className="divide-y divide-gray-200 max-h-96 overflow-y-auto">
        {verdict.results.map((result, index) => (
          <div key={result.check_id} className="px-6 py-4 hover:bg-gray-50">
            <div className="flex items-start justify-between">
              <div className="flex items-start space-x-3 flex-1">
                {getStatusIcon(result.status)}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center space-x-2">
                    <h4 className="text-sm font-medium text-gray-900">{result.check_name}</h4>
                    <span className={`px-2 py-1 text-xs rounded border ${getSeverityColor(result.severity)}`}>
                      {result.severity}
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 mt-1">{result.message}</p>
                  {result.file_path && (
                    <div className="text-xs text-gray-500 mt-1">
                      {result.file_path}
                      {result.line_number && `:${result.line_number}`}
                    </div>
                  )}
                </div>
              </div>
              <div className="flex items-center space-x-2 ml-4">
                <span className="text-xs text-gray-500">
                  {result.execution_time.toFixed(2)}s
                </span>
                {onViewDetails && result.details && (
                  <button
                    onClick={() => onViewDetails(result)}
                    className="text-xs text-blue-600 hover:text-blue-800 hover:underline"
                  >
                    Details
                  </button>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Footer */}
      <div className="px-6 py-3 bg-gray-50 border-t border-gray-200">
        <div className="flex items-center justify-between text-sm text-gray-500">
          <div>
            Error Rate: {(verdict.error_rate * 100).toFixed(1)}% | 
            False Positive Rate: {(verdict.false_positive_rate * 100).toFixed(1)}%
          </div>
          {onRetry && (
            <button
              onClick={onRetry}
              className="px-3 py-1 text-blue-600 hover:text-blue-800 hover:bg-blue-50 rounded transition-colors"
            >
              Re-run Validation
            </button>
          )}
        </div>
      </div>
    </div>
  );
};