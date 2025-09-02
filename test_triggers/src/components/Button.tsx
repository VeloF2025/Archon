// Test React component for autonomous trigger testing
import React from 'react';

export const Button = ({ label, onClick }) => {
  return <button onClick={onClick}>{label}</button>;
};