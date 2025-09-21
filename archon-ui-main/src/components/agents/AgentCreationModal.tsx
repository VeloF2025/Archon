import React, { useState } from 'react';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '../ui/dialog';
import { Button } from '../ui/button';
import { Input } from '../ui/Input';
import { Textarea } from '../ui/textarea';
import { Label } from '../ui/label';
import { Select } from '../ui/Select';

export interface AgentCreationData {
  name: string;
  description: string;
  type: string;
  capabilities: string[];
  config: Record<string, any>;
}

interface AgentCreationModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCreateAgent: (agentData: AgentCreationData) => void;
}

const AGENT_TYPES = [
  { value: 'code-implementer', label: 'Code Implementer' },
  { value: 'system-architect', label: 'System Architect' },
  { value: 'test-coverage-validator', label: 'Test Coverage Validator' },
  { value: 'security-auditor', label: 'Security Auditor' },
  { value: 'performance-optimizer', label: 'Performance Optimizer' },
  { value: 'code-quality-reviewer', label: 'Code Quality Reviewer' },
  { value: 'ui-ux-optimizer', label: 'UI/UX Optimizer' },
  { value: 'database-architect', label: 'Database Architect' },
  { value: 'api-design-architect', label: 'API Design Architect' },
  { value: 'documentation-generator', label: 'Documentation Generator' },
];

export const AgentCreationModal: React.FC<AgentCreationModalProps> = ({
  isOpen,
  onClose,
  onCreateAgent,
}) => {
  const [formData, setFormData] = useState<AgentCreationData>({
    name: '',
    description: '',
    type: '',
    capabilities: [],
    config: {},
  });

  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!formData.name.trim() || !formData.type) {
      return;
    }

    setIsSubmitting(true);
    
    try {
      await onCreateAgent(formData);
      
      // Reset form
      setFormData({
        name: '',
        description: '',
        type: '',
        capabilities: [],
        config: {},
      });
      
      onClose();
    } catch (error) {
      console.error('Failed to create agent:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleInputChange = (field: keyof AgentCreationData) => (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    setFormData(prev => ({
      ...prev,
      [field]: e.target.value,
    }));
  };

  const handleSelectChange = (value: string) => {
    setFormData(prev => ({
      ...prev,
      type: value,
    }));
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Create New Agent</DialogTitle>
          <DialogDescription>
            Create a new AI agent with specific capabilities for your project.
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="space-y-4">
            <div>
              <Label htmlFor="agent-name">Agent Name</Label>
              <Input
                id="agent-name"
                value={formData.name}
                onChange={handleInputChange('name')}
                placeholder="Enter agent name"
                required
              />
            </div>

            <div>
              <Label htmlFor="agent-type">Agent Type</Label>
              <Select
                value={formData.type}
                onValueChange={handleSelectChange}
                placeholder="Select agent type"
              >
                {AGENT_TYPES.map(type => (
                  <option key={type.value} value={type.value}>
                    {type.label}
                  </option>
                ))}
              </Select>
            </div>

            <div>
              <Label htmlFor="agent-description">Description</Label>
              <Textarea
                id="agent-description"
                value={formData.description}
                onChange={handleInputChange('description')}
                placeholder="Describe what this agent will do"
                rows={3}
              />
            </div>
          </div>

          <div className="flex justify-end space-x-3">
            <Button
              type="button"
              variant="outline"
              onClick={onClose}
              disabled={isSubmitting}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={!formData.name.trim() || !formData.type || isSubmitting}
            >
              {isSubmitting ? 'Creating...' : 'Create Agent'}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default AgentCreationModal;