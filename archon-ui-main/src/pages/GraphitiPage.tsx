import React from 'react';
import { GraphExplorer } from '@/components/graphiti/GraphExplorer';

export const GraphitiPage: React.FC = () => {
  return (
    <div className="w-full h-screen">
      <GraphExplorer />
    </div>
  );
};