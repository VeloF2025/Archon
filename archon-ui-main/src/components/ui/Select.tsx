import React, { useState } from 'react';

// Legacy Select component
interface LegacySelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  accentColor?: 'purple' | 'green' | 'pink' | 'blue';
  label?: string;
  options: {
    value: string;
    label: string;
  }[];
}
export const LegacySelect: React.FC<LegacySelectProps> = ({
  accentColor = 'purple',
  label,
  options,
  className = '',
  ...props
}) => {
  const accentColorMap = {
    purple: 'focus-within:border-purple-500 focus-within:shadow-[0_0_15px_rgba(168,85,247,0.5)]',
    green: 'focus-within:border-emerald-500 focus-within:shadow-[0_0_15px_rgba(16,185,129,0.5)]',
    pink: 'focus-within:border-pink-500 focus-within:shadow-[0_0_15px_rgba(236,72,153,0.5)]',
    blue: 'focus-within:border-blue-500 focus-within:shadow-[0_0_15px_rgba(59,130,246,0.5)]'
  };
  return <div className="w-full">
      {label && <label className="block text-gray-600 dark:text-zinc-400 text-sm mb-1.5">
          {label}
        </label>}
      <div className={`
        relative backdrop-blur-md bg-gradient-to-b dark:from-white/10 dark:to-black/30 from-white/80 to-white/60
        border dark:border-zinc-800/80 border-gray-200 rounded-md
        transition-all duration-200 ${accentColorMap[accentColor]}
      `}>
        <select className={`
            w-full bg-transparent text-gray-800 dark:text-white appearance-none px-3 py-2
            focus:outline-none ${className}
          `} {...props}>
          {options.map(option => <option key={option.value} value={option.value} className="bg-white dark:bg-zinc-900">
              {option.label}
            </option>)}
        </select>
        <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-gray-500 dark:text-zinc-500">
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M2.5 4.5L6 8L9.5 4.5" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
      </div>
    </div>;
};

// Modern Select components for Shadcn compatibility
export const Select: React.FC<{
  value: string;
  onValueChange: (value: string) => void;
  children: React.ReactNode;
}> = ({ value, onValueChange, children }) => {
  const [isOpen, setIsOpen] = useState(false);
  
  return (
    <div className="relative">
      {React.Children.map(children, child => 
        React.isValidElement(child) 
          ? React.cloneElement(child, { value, onValueChange, isOpen, setIsOpen })
          : child
      )}
    </div>
  );
};

export const SelectTrigger: React.FC<{
  children: React.ReactNode;
  className?: string;
  value?: string;
  isOpen?: boolean;
  setIsOpen?: (open: boolean) => void;
}> = ({ children, className = '', value, isOpen, setIsOpen }) => {
  return (
    <button
      type="button"
      className={`flex h-10 w-full items-center justify-between rounded-md border border-gray-300 bg-white px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 dark:border-gray-600 dark:bg-gray-700 ${className}`}
      onClick={() => setIsOpen?.(!isOpen)}
    >
      {children}
      <svg width="12" height="12" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M2.5 4.5L6 8L9.5 4.5" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </button>
  );
};

export const SelectValue: React.FC<{
  placeholder?: string;
  value?: string;
}> = ({ placeholder, value }) => {
  return (
    <span>
      {value || placeholder}
    </span>
  );
};

export const SelectContent: React.FC<{
  children: React.ReactNode;
  isOpen?: boolean;
  setIsOpen?: (open: boolean) => void;
  onValueChange?: (value: string) => void;
}> = ({ children, isOpen, setIsOpen, onValueChange }) => {
  if (!isOpen) return null;
  
  return (
    <>
      <div 
        className="fixed inset-0 z-40" 
        onClick={() => setIsOpen?.(false)}
      />
      <div className="absolute top-full z-50 w-full mt-1 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md shadow-lg">
        {React.Children.map(children, child => 
          React.isValidElement(child) 
            ? React.cloneElement(child, { onValueChange, setIsOpen })
            : child
        )}
      </div>
    </>
  );
};

export const SelectItem: React.FC<{
  value: string;
  children: React.ReactNode;
  onValueChange?: (value: string) => void;
  setIsOpen?: (open: boolean) => void;
}> = ({ value, children, onValueChange, setIsOpen }) => {
  return (
    <div
      className="relative flex w-full cursor-default select-none items-center rounded-sm py-1.5 pl-8 pr-2 text-sm outline-none hover:bg-gray-100 dark:hover:bg-gray-600 focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50"
      onClick={() => {
        onValueChange?.(value);
        setIsOpen?.(false);
      }}
    >
      {children}
    </div>
  );
};