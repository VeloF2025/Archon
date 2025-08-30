import React, { useState } from 'react';

export const SimpleGraphExplorer: React.FC = () => {
  const [selectedEntity, setSelectedEntity] = useState<string | null>(null);
  
  // Mock data
  const mockEntities = [
    { id: 'func_auth_login', type: 'function', name: 'login', confidence: 95 },
    { id: 'class_user_manager', type: 'class', name: 'UserManager', confidence: 92 },
    { id: 'agent_code_implementer', type: 'agent', name: 'code-implementer', confidence: 97 },
    { id: 'concept_authentication', type: 'concept', name: 'Authentication', confidence: 88 }
  ];

  const mockRelationships = [
    { from: 'agent_code_implementer', to: 'func_auth_login', type: 'implements' },
    { from: 'func_auth_login', to: 'class_user_manager', type: 'calls' },
    { from: 'concept_authentication', to: 'func_auth_login', type: 'related_to' }
  ];

  const getEntityColor = (type: string) => {
    const colors: Record<string, string> = {
      function: 'bg-blue-100 text-blue-800 border-blue-300',
      class: 'bg-green-100 text-green-800 border-green-300',
      agent: 'bg-red-100 text-red-800 border-red-300',
      concept: 'bg-purple-100 text-purple-800 border-purple-300',
    };
    return colors[type] || 'bg-gray-100 text-gray-800 border-gray-300';
  };

  return (
    <div className="w-full h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <div className="bg-white border-b p-4 shadow-sm">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-bold text-gray-900 flex items-center space-x-2">
            <svg className="h-5 w-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16l2.879-2.879m0 0a3 3 0 104.243-4.242 3 3 0 00-4.243 4.242zM21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>Graphiti Explorer</span>
          </h1>
          <div className="text-sm text-gray-500">
            {mockEntities.length} entities • {mockRelationships.length} relationships
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex">
        {/* Graph View */}
        <div className="flex-1 p-6">
          <div className="bg-white rounded-lg shadow-lg h-full p-6">
            <h2 className="text-lg font-semibold mb-4">Knowledge Graph</h2>
            
            {/* Simple Graph Visualization */}
            <div className="grid grid-cols-2 gap-4 mb-6">
              {mockEntities.map((entity) => (
                <div
                  key={entity.id}
                  className={`p-4 rounded-lg border-2 cursor-pointer transition-all hover:shadow-md ${
                    selectedEntity === entity.id ? 'ring-2 ring-blue-500' : ''
                  } ${getEntityColor(entity.type)}`}
                  onClick={() => setSelectedEntity(entity.id)}
                >
                  <div className="font-semibold">{entity.name}</div>
                  <div className="text-xs uppercase tracking-wide mt-1">{entity.type}</div>
                  <div className="text-xs mt-2">Confidence: {entity.confidence}%</div>
                </div>
              ))}
            </div>

            {/* Relationships */}
            <div className="border-t pt-4">
              <h3 className="font-semibold mb-3">Relationships</h3>
              <div className="space-y-2">
                {mockRelationships.map((rel, index) => {
                  const fromEntity = mockEntities.find(e => e.id === rel.from);
                  const toEntity = mockEntities.find(e => e.id === rel.to);
                  return (
                    <div key={index} className="flex items-center space-x-2 text-sm">
                      <span className="font-medium">{fromEntity?.name}</span>
                      <span className="text-gray-500">→</span>
                      <span className="px-2 py-1 bg-gray-100 rounded text-xs">{rel.type}</span>
                      <span className="text-gray-500">→</span>
                      <span className="font-medium">{toEntity?.name}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {/* Side Panel */}
        <div className="w-80 bg-white border-l p-6">
          {selectedEntity ? (
            <div>
              <h2 className="text-lg font-semibold mb-4">Entity Details</h2>
              {(() => {
                const entity = mockEntities.find(e => e.id === selectedEntity);
                if (!entity) return null;
                return (
                  <div className="space-y-4">
                    <div>
                      <label className="text-sm font-medium text-gray-500">Name</label>
                      <div className="font-semibold">{entity.name}</div>
                    </div>
                    <div>
                      <label className="text-sm font-medium text-gray-500">Type</label>
                      <div className={`inline-block px-2 py-1 rounded text-sm ${getEntityColor(entity.type)}`}>
                        {entity.type}
                      </div>
                    </div>
                    <div>
                      <label className="text-sm font-medium text-gray-500">Confidence Score</label>
                      <div className="flex items-center space-x-2">
                        <div className="flex-1 bg-gray-200 rounded-full h-2">
                          <div 
                            className="bg-blue-600 h-2 rounded-full" 
                            style={{ width: `${entity.confidence}%` }}
                          />
                        </div>
                        <span className="text-sm">{entity.confidence}%</span>
                      </div>
                    </div>
                    <div>
                      <label className="text-sm font-medium text-gray-500">Related Entities</label>
                      <div className="space-y-1 mt-1">
                        {mockRelationships
                          .filter(rel => rel.from === selectedEntity || rel.to === selectedEntity)
                          .map((rel, index) => {
                            const otherEntityId = rel.from === selectedEntity ? rel.to : rel.from;
                            const otherEntity = mockEntities.find(e => e.id === otherEntityId);
                            return (
                              <div key={index} className="text-sm text-gray-600">
                                {rel.type} → {otherEntity?.name}
                              </div>
                            );
                          })}
                      </div>
                    </div>
                  </div>
                );
              })()}
            </div>
          ) : (
            <div className="text-center text-gray-500 mt-8">
              <div className="text-4xl mb-2">🔍</div>
              <p>Click on an entity to view details</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};