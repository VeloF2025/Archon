import React from 'react';

export const Tabs: React.FC<{ 
  defaultValue: string; 
  children: React.ReactNode; 
  className?: string;
}> = ({ defaultValue, children, className = '' }) => {
  const [activeTab, setActiveTab] = React.useState(defaultValue);
  
  return (
    <div className={`w-full ${className}`} data-active-tab={activeTab}>
      {React.Children.map(children, child => 
        React.isValidElement(child) 
          ? React.cloneElement(child, { activeTab, setActiveTab })
          : child
      )}
    </div>
  );
};

export const TabsList: React.FC<{ 
  children: React.ReactNode; 
  className?: string;
  activeTab?: string;
  setActiveTab?: (value: string) => void;
}> = ({ children, className = '', activeTab, setActiveTab }) => {
  return (
    <div className={`inline-flex h-10 items-center justify-center rounded-md bg-muted p-1 text-muted-foreground ${className}`}>
      {React.Children.map(children, child => 
        React.isValidElement(child) 
          ? React.cloneElement(child, { activeTab, setActiveTab })
          : child
      )}
    </div>
  );
};

export const TabsTrigger: React.FC<{ 
  value: string; 
  children: React.ReactNode;
  activeTab?: string;
  setActiveTab?: (value: string) => void;
}> = ({ value, children, activeTab, setActiveTab }) => {
  const isActive = activeTab === value;
  
  return (
    <button
      className={`inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 ${
        isActive 
          ? 'bg-background text-foreground shadow-sm' 
          : 'hover:bg-background/50'
      }`}
      onClick={() => setActiveTab?.(value)}
    >
      {children}
    </button>
  );
};

export const TabsContent: React.FC<{ 
  value: string; 
  children: React.ReactNode;
  className?: string;
  activeTab?: string;
}> = ({ value, children, className = '', activeTab }) => {
  if (activeTab !== value) return null;
  
  return (
    <div className={`mt-2 ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 ${className}`}>
      {children}
    </div>
  );
};