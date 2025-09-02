import React from 'react';

export const Label: React.FC<{
  htmlFor?: string;
  children: React.ReactNode;
  className?: string;
}> = ({ htmlFor, children, className = '' }) => {
  return (
    <label 
      htmlFor={htmlFor}
      className={`text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 text-gray-700 dark:text-gray-300 ${className}`}
    >
      {children}
    </label>
  );
};