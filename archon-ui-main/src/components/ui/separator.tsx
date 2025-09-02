import React from 'react';

export const Separator: React.FC<{
  orientation?: 'horizontal' | 'vertical';
  className?: string;
}> = ({ orientation = 'horizontal', className = '' }) => {
  return (
    <div 
      className={`${
        orientation === 'horizontal' 
          ? 'h-px w-full bg-gray-200 dark:bg-gray-700' 
          : 'w-px h-full bg-gray-200 dark:bg-gray-700'
      } ${className}`}
    />
  );
};