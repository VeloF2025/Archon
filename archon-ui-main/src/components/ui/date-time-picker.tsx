import React from 'react';

export const DateTimePicker: React.FC<{
  date: Date;
  setDate: (date: Date) => void;
  className?: string;
}> = ({ date, setDate, className = '' }) => {
  const handleDateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newDate = new Date(e.target.value);
    setDate(newDate);
  };

  const formatForInput = (date: Date): string => {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    
    return `${year}-${month}-${day}T${hours}:${minutes}`;
  };

  return (
    <input
      type="datetime-local"
      value={formatForInput(date)}
      onChange={handleDateChange}
      className={`
        w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm 
        focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500
        dark:border-gray-600 dark:bg-gray-700 dark:text-white
        ${className}
      `}
    />
  );
};