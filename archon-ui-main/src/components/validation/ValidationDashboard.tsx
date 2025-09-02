import React, { useState, useEffect } from 'react';
import { Shield, Zap, Clock, TrendingUp, AlertCircle, CheckCircle } from 'lucide-react';
import { ValidationSummary } from './ValidationSummary';
import { PromptViewer } from './PromptViewer';

interface ValidationStats {
  total_validations: number;
  passed: number;
  failed: number;
  errors: number;
  success_rate: number;
  average_error_rate: number;
  average_false_positive_rate: number;
  recent_validations: any[];
}

interface EnhancementStats {
  total_enhancements: number;
  average_score: number;
  average_processing_time: number;
  direction_distribution: Record<string, number>;
  cache_hits: number;
  recent_enhancements: any[];
}

interface ValidationDashboardProps {
  onRunValidation?: (taskId: string, code: string) => Promise<any>;
  onEnhancePrompt?: (prompt: string, direction: 'to-sub' | 'from-sub', level: string) => Promise<any>;
}

export const ValidationDashboard: React.FC<ValidationDashboardProps> = ({
  onRunValidation,
  onEnhancePrompt
}) => {
  const [validationStats, setValidationStats] = useState<ValidationStats | null>(null);
  const [enhancementStats, setEnhancementStats] = useState<EnhancementStats | null>(null);
  const [currentValidation, setCurrentValidation] = useState<any>(null);
  const [currentEnhancement, setCurrentEnhancement] = useState<any>(null);
  const [activeTab, setActiveTab] = useState<'validation' | 'enhancement'>('validation');
  const [loading, setLoading] = useState<{ validation: boolean; enhancement: boolean; stats: boolean }>({
    validation: false,
    enhancement: false,
    stats: false
  });

  // User-provided code and prompt
  const [userCode, setUserCode] = useState('');
  const [userPrompt, setUserPrompt] = useState('');

  const fetchStats = async () => {
    setLoading(prev => ({ ...prev, stats: true }));
    
    // Simulate API calls - replace with real API calls
    setTimeout(() => {
      setValidationStats({
        total_validations: 45,
        passed: 38,
        failed: 5,
        errors: 2,
        success_rate: 0.844,
        average_error_rate: 0.156,
        average_false_positive_rate: 0.05,
        recent_validations: []
      });

      setEnhancementStats({
        total_enhancements: 123,
        average_score: 0.847,
        average_processing_time: 0.234,
        direction_distribution: {
          'to-sub': 78,
          'from-sub': 45
        },
        cache_hits: 34,
        recent_enhancements: []
      });

      setLoading(prev => ({ ...prev, stats: false }));
    }, 1000);
  };

  useEffect(() => {
    fetchStats();
  }, []);

  const handleRunValidation = async () => {
    if (!onRunValidation) return;
    
    setLoading(prev => ({ ...prev, validation: true }));
    
    try {
      const result = await onRunValidation(`user_task_${Date.now()}`, userCode);
      setCurrentValidation(result);
    } catch (error) {
      console.error('Validation failed:', error);
    } finally {
      setLoading(prev => ({ ...prev, validation: false }));
    }
  };

  const handleEnhancePrompt = async (level: 'basic' | 'enhanced' | 'comprehensive') => {
    if (!onEnhancePrompt) return;
    
    setLoading(prev => ({ ...prev, enhancement: true }));
    
    try {
      const result = await onEnhancePrompt(userPrompt, 'to-sub', level);
      setCurrentEnhancement(result);
    } catch (error) {
      console.error('Enhancement failed:', error);
    } finally {
      setLoading(prev => ({ ...prev, enhancement: false }));
    }
  };

  const StatCard: React.FC<{
    title: string;
    value: string | number;
    subtitle?: string;
    icon: React.ReactNode;
    color: string;
  }> = ({ title, value, subtitle, icon, color }) => (
    <div className="bg-white p-6 rounded-lg border border-gray-200">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className={`text-2xl font-bold ${color}`}>{value}</p>
          {subtitle && (
            <p className="text-sm text-gray-500">{subtitle}</p>
          )}
        </div>
        <div className={`${color} opacity-20`}>
          {icon}
        </div>
      </div>
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">Phase 3 Validation Dashboard</h2>
        <button
          onClick={fetchStats}
          disabled={loading.stats}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 flex items-center space-x-2"
        >
          <Clock className="h-4 w-4" />
          <span>Refresh Stats</span>
        </button>
      </div>

      {/* Stats Grid */}
      {validationStats && enhancementStats && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            title="Validation Success Rate"
            value={`${(validationStats.success_rate * 100).toFixed(1)}%`}
            subtitle={`${validationStats.passed}/${validationStats.total_validations} passed`}
            icon={<Shield className="h-8 w-8" />}
            color="text-green-600"
          />
          <StatCard
            title="Enhancement Score"
            value={enhancementStats.average_score.toFixed(3)}
            subtitle={`${enhancementStats.total_enhancements} total`}
            icon={<Zap className="h-8 w-8" />}
            color="text-blue-600"
          />
          <StatCard
            title="Avg Processing Time"
            value={`${enhancementStats.average_processing_time.toFixed(2)}s`}
            subtitle="Enhancement speed"
            icon={<Clock className="h-8 w-8" />}
            color="text-purple-600"
          />
          <StatCard
            title="Error Rate"
            value={`${(validationStats.average_error_rate * 100).toFixed(1)}%`}
            subtitle={`${validationStats.errors} errors`}
            icon={<AlertCircle className="h-8 w-8" />}
            color="text-orange-600"
          />
        </div>
      )}

      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          <button
            onClick={() => setActiveTab('validation')}
            className={`py-2 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'validation'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            <Shield className="inline h-4 w-4 mr-2" />
            External Validation
          </button>
          <button
            onClick={() => setActiveTab('enhancement')}
            className={`py-2 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'enhancement'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            <Zap className="inline h-4 w-4 mr-2" />
            Prompt Enhancement
          </button>
        </nav>
      </div>

      {/* Validation Tab */}
      {activeTab === 'validation' && (
        <div className="space-y-6">
          {/* Demo Code Section */}
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Code Validation</h3>
              <button
                onClick={handleRunValidation}
                disabled={loading.validation}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 flex items-center space-x-2"
              >
                <CheckCircle className="h-4 w-4" />
                <span>Run Validation</span>
              </button>
            </div>
            
            <div className="bg-gray-50 rounded-md p-4 font-mono text-sm overflow-x-auto">
              <textarea 
                className="w-full h-64 p-3 border rounded-md font-mono text-sm"
                placeholder="Enter your code here for validation..."
                value={userCode}
                onChange={(e) => setUserCode(e.target.value)}
              />
            </div>
          </div>

          {/* Validation Results */}
          <ValidationSummary
            verdict={currentValidation}
            loading={loading.validation}
            onRetry={handleRunValidation}
            onViewDetails={(result) => console.log('View details:', result)}
          />
        </div>
      )}

      {/* Enhancement Tab */}
      {activeTab === 'enhancement' && (
        <div className="space-y-6">
          {/* Prompt Input */}
          <div className="bg-card border rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Zap className="w-5 h-5" />
              Enter Prompt for Enhancement
            </h3>
            <textarea 
              className="w-full h-32 p-3 border rounded-md text-sm"
              placeholder="Enter your prompt here for enhancement..."
              value={userPrompt}
              onChange={(e) => setUserPrompt(e.target.value)}
            />
          </div>

          <PromptViewer
            originalPrompt={userPrompt}
            enhancementResult={currentEnhancement}
            loading={loading.enhancement}
            onEnhance={handleEnhancePrompt}
            onCopy={(text) => {
              navigator.clipboard.writeText(text);
              console.log('Copied to clipboard:', text.slice(0, 50) + '...');
            }}
            onDownload={() => {
              if (currentEnhancement) {
                const blob = new Blob([currentEnhancement.enhanced_prompt], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `enhanced_prompt_${currentEnhancement.request_id.slice(0, 8)}.txt`;
                a.click();
                URL.revokeObjectURL(url);
              }
            }}
          />
        </div>
      )}

      {/* Additional Stats */}
      {validationStats && enhancementStats && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Validation Trends */}
          <div className="bg-white p-6 rounded-lg border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Validation Metrics</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Total Validations:</span>
                <span className="font-medium">{validationStats.total_validations}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600">False Positive Rate:</span>
                <span className="font-medium">{(validationStats.average_false_positive_rate * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Success Rate:</span>
                <div className="flex items-center space-x-2">
                  <div className="w-20 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-green-500 h-2 rounded-full"
                      style={{ width: `${validationStats.success_rate * 100}%` }}
                    ></div>
                  </div>
                  <span className="font-medium">{(validationStats.success_rate * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>
          </div>

          {/* Enhancement Metrics */}
          <div className="bg-white p-6 rounded-lg border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Enhancement Metrics</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Total Enhancements:</span>
                <span className="font-medium">{enhancementStats.total_enhancements}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Cache Hit Rate:</span>
                <span className="font-medium">
                  {((enhancementStats.cache_hits / enhancementStats.total_enhancements) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Direction Split:</span>
                <div className="text-sm">
                  <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded mr-2">
                    To-Sub: {enhancementStats.direction_distribution['to-sub']}
                  </span>
                  <span className="bg-green-100 text-green-800 px-2 py-1 rounded">
                    From-Sub: {enhancementStats.direction_distribution['from-sub']}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};