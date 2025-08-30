import React, { useState } from 'react';
import { Eye, EyeOff, Copy, Download, RefreshCw, Zap, ArrowRight } from 'lucide-react';

interface ContextInjection {
  type: string;
  content: string;
  confidence: number;
  source: string;
  metadata: Record<string, any>;
}

interface PromptEnhancementResult {
  enhanced_prompt: string;
  enhancement_score: number;
  context_injections: ContextInjection[];
  validation_flags: string[];
  metadata: Record<string, any>;
  processing_time: number;
  request_id: string;
}

interface PromptViewerProps {
  originalPrompt: string;
  enhancementResult: PromptEnhancementResult | null;
  loading: boolean;
  onEnhance?: (level: 'basic' | 'enhanced' | 'comprehensive') => void;
  onCopy?: (text: string) => void;
  onDownload?: () => void;
}

export const PromptViewer: React.FC<PromptViewerProps> = ({
  originalPrompt,
  enhancementResult,
  loading,
  onEnhance,
  onCopy,
  onDownload
}) => {
  const [viewMode, setViewMode] = useState<'split' | 'original' | 'enhanced'>('split');
  const [showDetails, setShowDetails] = useState(false);

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600 bg-green-50';
    if (score >= 0.6) return 'text-yellow-600 bg-yellow-50';
    return 'text-red-600 bg-red-50';
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'bg-green-500';
    if (confidence >= 0.6) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  const formatPromptText = (text: string) => {
    return text.split('\n').map((line, index) => (
      <div key={index} className={line.trim() === '' ? 'h-4' : ''}>
        {line || '\u00A0'}
      </div>
    ));
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    if (onCopy) onCopy(text);
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Zap className="h-5 w-5 text-blue-500" />
            <h3 className="text-lg font-semibold text-gray-900">Prompt Enhancement</h3>
            {enhancementResult && (
              <span className={`px-2 py-1 text-xs font-medium rounded ${getScoreColor(enhancementResult.enhancement_score)}`}>
                Score: {enhancementResult.enhancement_score.toFixed(2)}
              </span>
            )}
          </div>
          
          <div className="flex items-center space-x-2">
            {/* View Mode Toggle */}
            <div className="flex rounded-md border border-gray-300">
              <button
                onClick={() => setViewMode('original')}
                className={`px-3 py-1 text-xs font-medium ${
                  viewMode === 'original' 
                    ? 'bg-blue-500 text-white' 
                    : 'bg-white text-gray-600 hover:bg-gray-50'
                }`}
              >
                Original
              </button>
              <button
                onClick={() => setViewMode('split')}
                className={`px-3 py-1 text-xs font-medium border-x border-gray-300 ${
                  viewMode === 'split' 
                    ? 'bg-blue-500 text-white' 
                    : 'bg-white text-gray-600 hover:bg-gray-50'
                }`}
              >
                Split
              </button>
              <button
                onClick={() => setViewMode('enhanced')}
                className={`px-3 py-1 text-xs font-medium ${
                  viewMode === 'enhanced' 
                    ? 'bg-blue-500 text-white' 
                    : 'bg-white text-gray-600 hover:bg-gray-50'
                }`}
                disabled={!enhancementResult}
              >
                Enhanced
              </button>
            </div>

            {/* Enhancement Controls */}
            {onEnhance && (
              <div className="flex space-x-1">
                <button
                  onClick={() => onEnhance('basic')}
                  disabled={loading}
                  className="px-3 py-1 text-xs bg-gray-100 text-gray-600 rounded hover:bg-gray-200 disabled:opacity-50"
                >
                  Basic
                </button>
                <button
                  onClick={() => onEnhance('enhanced')}
                  disabled={loading}
                  className="px-3 py-1 text-xs bg-blue-100 text-blue-600 rounded hover:bg-blue-200 disabled:opacity-50"
                >
                  Enhanced
                </button>
                <button
                  onClick={() => onEnhance('comprehensive')}
                  disabled={loading}
                  className="px-3 py-1 text-xs bg-purple-100 text-purple-600 rounded hover:bg-purple-200 disabled:opacity-50"
                >
                  Full
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Loading State */}
      {loading && (
        <div className="px-6 py-8">
          <div className="flex items-center justify-center space-x-3">
            <RefreshCw className="h-5 w-5 animate-spin text-blue-500" />
            <span className="text-gray-600">Enhancing prompt...</span>
          </div>
        </div>
      )}

      {/* Prompt Display */}
      {!loading && (
        <div className="flex-1">
          {/* Split View */}
          {viewMode === 'split' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 divide-y lg:divide-y-0 lg:divide-x divide-gray-200 min-h-96">
              <div className="p-6">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-sm font-medium text-gray-900">Original Prompt</h4>
                  <button
                    onClick={() => copyToClipboard(originalPrompt)}
                    className="p-1 text-gray-400 hover:text-gray-600 rounded"
                    title="Copy original prompt"
                  >
                    <Copy className="h-4 w-4" />
                  </button>
                </div>
                <div className="bg-gray-50 rounded-md p-4 text-sm text-gray-700 font-mono whitespace-pre-wrap max-h-80 overflow-y-auto">
                  {formatPromptText(originalPrompt)}
                </div>
              </div>

              <div className="p-6">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-sm font-medium text-gray-900">Enhanced Prompt</h4>
                  {enhancementResult && (
                    <button
                      onClick={() => copyToClipboard(enhancementResult.enhanced_prompt)}
                      className="p-1 text-gray-400 hover:text-gray-600 rounded"
                      title="Copy enhanced prompt"
                    >
                      <Copy className="h-4 w-4" />
                    </button>
                  )}
                </div>
                <div className="bg-blue-50 rounded-md p-4 text-sm text-gray-700 font-mono whitespace-pre-wrap max-h-80 overflow-y-auto">
                  {enhancementResult ? (
                    formatPromptText(enhancementResult.enhanced_prompt)
                  ) : (
                    <div className="text-gray-400 italic">No enhancement result available</div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Original Only */}
          {viewMode === 'original' && (
            <div className="p-6">
              <div className="flex items-center justify-between mb-3">
                <h4 className="text-sm font-medium text-gray-900">Original Prompt</h4>
                <button
                  onClick={() => copyToClipboard(originalPrompt)}
                  className="p-1 text-gray-400 hover:text-gray-600 rounded"
                >
                  <Copy className="h-4 w-4" />
                </button>
              </div>
              <div className="bg-gray-50 rounded-md p-4 text-sm text-gray-700 font-mono whitespace-pre-wrap max-h-96 overflow-y-auto">
                {formatPromptText(originalPrompt)}
              </div>
            </div>
          )}

          {/* Enhanced Only */}
          {viewMode === 'enhanced' && enhancementResult && (
            <div className="p-6">
              <div className="flex items-center justify-between mb-3">
                <h4 className="text-sm font-medium text-gray-900">Enhanced Prompt</h4>
                <button
                  onClick={() => copyToClipboard(enhancementResult.enhanced_prompt)}
                  className="p-1 text-gray-400 hover:text-gray-600 rounded"
                >
                  <Copy className="h-4 w-4" />
                </button>
              </div>
              <div className="bg-blue-50 rounded-md p-4 text-sm text-gray-700 font-mono whitespace-pre-wrap max-h-96 overflow-y-auto">
                {formatPromptText(enhancementResult.enhanced_prompt)}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Enhancement Details */}
      {enhancementResult && (
        <div className="border-t border-gray-200">
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="w-full px-6 py-3 text-left text-sm font-medium text-gray-700 hover:bg-gray-50 flex items-center justify-between"
          >
            <span>Enhancement Details</span>
            {showDetails ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
          </button>

          {showDetails && (
            <div className="px-6 py-4 bg-gray-50 border-t border-gray-200">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Metrics */}
                <div>
                  <h5 className="text-sm font-medium text-gray-900 mb-3">Metrics</h5>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Enhancement Score:</span>
                      <span className="font-medium">{enhancementResult.enhancement_score.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Processing Time:</span>
                      <span className="font-medium">{enhancementResult.processing_time.toFixed(3)}s</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Context Injections:</span>
                      <span className="font-medium">{enhancementResult.context_injections.length}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Request ID:</span>
                      <span className="font-mono text-xs">{enhancementResult.request_id.slice(0, 8)}</span>
                    </div>
                  </div>
                </div>

                {/* Context Injections */}
                <div>
                  <h5 className="text-sm font-medium text-gray-900 mb-3">Context Injections</h5>
                  <div className="space-y-2 max-h-40 overflow-y-auto">
                    {enhancementResult.context_injections.map((injection, index) => (
                      <div key={index} className="p-2 bg-white rounded border text-xs">
                        <div className="flex items-center justify-between mb-1">
                          <span className="font-medium capitalize">{injection.type}</span>
                          <div className="flex items-center space-x-1">
                            <div 
                              className={`w-2 h-2 rounded-full ${getConfidenceColor(injection.confidence)}`}
                              title={`Confidence: ${injection.confidence.toFixed(2)}`}
                            ></div>
                            <span className="text-gray-500">{injection.source}</span>
                          </div>
                        </div>
                        <div className="text-gray-600 line-clamp-2">
                          {injection.content.length > 100 
                            ? `${injection.content.slice(0, 100)}...` 
                            : injection.content}
                        </div>
                      </div>
                    ))}
                    {enhancementResult.context_injections.length === 0 && (
                      <div className="text-gray-500 text-xs italic">No context injections</div>
                    )}
                  </div>
                </div>
              </div>

              {/* Validation Flags */}
              {enhancementResult.validation_flags.length > 0 && (
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <h5 className="text-sm font-medium text-gray-900 mb-2">Validation Flags</h5>
                  <div className="flex flex-wrap gap-2">
                    {enhancementResult.validation_flags.map((flag, index) => (
                      <span 
                        key={index}
                        className="px-2 py-1 text-xs bg-yellow-100 text-yellow-800 rounded"
                      >
                        {flag.replace(/_/g, ' ')}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Footer Actions */}
      <div className="px-6 py-3 bg-gray-50 border-t border-gray-200 flex justify-between items-center">
        <div className="text-sm text-gray-500">
          {originalPrompt.split('\n').length} lines, {originalPrompt.length} characters
          {enhancementResult && (
            <span className="ml-2">
              <ArrowRight className="inline h-3 w-3 mx-1" />
              {enhancementResult.enhanced_prompt.split('\n').length} lines, {enhancementResult.enhanced_prompt.length} characters
            </span>
          )}
        </div>

        {onDownload && enhancementResult && (
          <button
            onClick={onDownload}
            className="flex items-center space-x-1 px-3 py-1 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded transition-colors"
          >
            <Download className="h-4 w-4" />
            <span>Download</span>
          </button>
        )}
      </div>
    </div>
  );
};