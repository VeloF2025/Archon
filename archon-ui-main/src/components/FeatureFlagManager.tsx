import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Switch } from './ui/switch';
import { Input } from './ui/Input';
import { Label } from './ui/label';
import { Textarea } from './ui/textarea';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from './ui/dialog';
import { Plus, Settings, BarChart3, Users, Flag, Trash2, Edit } from 'lucide-react';
import { toast } from 'sonner';

interface FeatureFlag {
  key: string;
  name: string;
  description: string;
  flag_type: string;
  status: string;
  default_value: any;
  enabled_for_all: boolean;
  percentage: number;
  user_ids: string[];
  variants: Record<string, any>;
  created_at: string;
  updated_at: string;
  tags: string[];
}

interface FeatureFlagVariant {
  id: string;
  flag_id: string;
  name: string;
  value: any;
  weight: number;
  is_control: boolean;
}

export const FeatureFlagManager: React.FC = () => {
  const [flags, setFlags] = useState<FeatureFlag[]>([]);
  const [variants, setVariants] = useState<Record<string, FeatureFlagVariant[]>>({});
  const [loading, setLoading] = useState(true);
  const [selectedFlag, setSelectedFlag] = useState<FeatureFlag | null>(null);
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [activeTab, setActiveTab] = useState('flags');

  // New flag form state
  const [newFlag, setNewFlag] = useState({
    name: '',
    description: '',
    is_enabled: false,
    rollout_percentage: 0,
    environment: 'production'
  });

  useEffect(() => {
    fetchFlags();
  }, []);

  const fetchFlags = async () => {
    try {
      const response = await fetch('/api/feature-flags/list');
      if (response.ok) {
        const data = await response.json();
        setFlags(data.flags || []);
        
        // Variants are already included in the flag data
        const variantsMap: Record<string, FeatureFlagVariant[]> = {};
        data.flags?.forEach((flag: FeatureFlag) => {
          if (flag.variants && Object.keys(flag.variants).length > 0) {
            variantsMap[flag.key] = Object.entries(flag.variants).map(([name, weight], index) => ({
              id: `${flag.key}_${name}`,
              flag_id: flag.key,
              name,
              value: { type: name },
              weight: typeof weight === 'number' ? weight : 0,
              is_control: index === 0
            }));
          }
        });
        setVariants(variantsMap);
      }
    } catch (error) {
      toast.error('Failed to fetch feature flags');
      console.error('Error fetching flags:', error);
    } finally {
      setLoading(false);
    }
  };

  const toggleFlag = async (flagKey: string, enabled: boolean) => {
    try {
      const endpoint = enabled ? `activate` : `deactivate`;
      const response = await fetch(`/api/feature-flags/${flagKey}/${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      if (response.ok) {
        setFlags(flags.map(flag => 
          flag.key === flagKey ? { ...flag, status: enabled ? 'active' : 'inactive' } : flag
        ));
        toast.success(`Feature flag ${enabled ? 'activated' : 'deactivated'}`);
      } else {
        toast.error('Failed to update feature flag');
      }
    } catch (error) {
      toast.error('Error updating feature flag');
      console.error('Error toggling flag:', error);
    }
  };

  const updateRolloutPercentage = async (flagId: string, percentage: number) => {
    try {
      const response = await fetch(`/api/feature-flags/${flagId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rollout_percentage: percentage })
      });

      if (response.ok) {
        setFlags(flags.map(flag => 
          flag.id === flagId ? { ...flag, rollout_percentage: percentage } : flag
        ));
        toast.success('Rollout percentage updated');
      } else {
        toast.error('Failed to update rollout percentage');
      }
    } catch (error) {
      toast.error('Error updating rollout percentage');
      console.error('Error updating rollout:', error);
    }
  };

  const createFlag = async () => {
    try {
      const flagData = {
        key: newFlag.name.toLowerCase().replace(/\s+/g, '_'),
        name: newFlag.name,
        description: newFlag.description,
        flag_type: 'boolean',
        default_value: newFlag.is_enabled,
        tags: ['ui-created']
      };

      const response = await fetch('/api/feature-flags/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(flagData)
      });

      if (response.ok) {
        await fetchFlags(); // Refresh the list
        setNewFlag({
          name: '',
          description: '',
          is_enabled: false,
          rollout_percentage: 0,
          environment: 'production'
        });
        setIsCreateDialogOpen(false);
        toast.success('Feature flag created successfully');
      } else {
        const error = await response.json();
        toast.error(error.detail || 'Failed to create feature flag');
      }
    } catch (error) {
      toast.error('Error creating feature flag');
      console.error('Error creating flag:', error);
    }
  };

  const deleteFlag = async (flagKey: string) => {
    if (!confirm('Are you sure you want to delete this feature flag?')) return;

    try {
      const response = await fetch(`/api/feature-flags/${flagKey}`, {
        method: 'DELETE'
      });

      if (response.ok) {
        setFlags(flags.filter(flag => flag.key !== flagKey));
        toast.success('Feature flag deleted');
      } else {
        toast.error('Failed to delete feature flag');
      }
    } catch (error) {
      toast.error('Error deleting feature flag');
      console.error('Error deleting flag:', error);
    }
  };

  const getRolloutColor = (percentage: number): string => {
    if (percentage === 0) return 'bg-gray-500';
    if (percentage < 25) return 'bg-red-500';
    if (percentage < 75) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Feature Flags
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Manage runtime feature toggles and gradual rollouts
          </p>
        </div>
        
        <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
          <DialogTrigger asChild>
            <Button className="flex items-center gap-2">
              <Plus className="w-4 h-4" />
              Create Flag
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create New Feature Flag</DialogTitle>
              <DialogDescription>
                Define a new feature flag for runtime control
              </DialogDescription>
            </DialogHeader>
            
            <div className="space-y-4">
              <div>
                <Label htmlFor="name">Name</Label>
                <Input
                  id="name"
                  placeholder="feature_name"
                  value={newFlag.name}
                  onChange={(e) => setNewFlag({...newFlag, name: e.target.value})}
                />
              </div>
              
              <div>
                <Label htmlFor="description">Description</Label>
                <Textarea
                  id="description"
                  placeholder="Describe what this feature flag controls..."
                  value={newFlag.description}
                  onChange={(e) => setNewFlag({...newFlag, description: e.target.value})}
                />
              </div>
              
              <div className="flex items-center space-x-2">
                <Switch
                  id="enabled"
                  checked={newFlag.is_enabled}
                  onCheckedChange={(checked) => setNewFlag({...newFlag, is_enabled: checked})}
                />
                <Label htmlFor="enabled">Start enabled</Label>
              </div>
              
              <div>
                <Label htmlFor="rollout">Initial rollout percentage</Label>
                <Input
                  id="rollout"
                  type="number"
                  min="0"
                  max="100"
                  value={newFlag.rollout_percentage}
                  onChange={(e) => setNewFlag({...newFlag, rollout_percentage: parseFloat(e.target.value)})}
                />
              </div>
              
              <div className="flex justify-end space-x-2">
                <Button 
                  variant="outline" 
                  onClick={() => setIsCreateDialogOpen(false)}
                >
                  Cancel
                </Button>
                <Button onClick={createFlag}>
                  Create Flag
                </Button>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="flags" className="flex items-center gap-2">
            <Flag className="w-4 h-4" />
            Flags ({flags.length})
          </TabsTrigger>
          <TabsTrigger value="analytics" className="flex items-center gap-2">
            <BarChart3 className="w-4 h-4" />
            Analytics
          </TabsTrigger>
        </TabsList>

        <TabsContent value="flags" className="space-y-4">
          <div className="grid gap-4">
            {flags.map((flag) => (
              <Card key={flag.key} className="p-4">
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <h3 className="font-semibold text-lg">{flag.name}</h3>
                      <Badge variant={flag.status === 'active' ? "default" : "secondary"}>
                        {flag.status === 'active' ? 'Active' : 'Inactive'}
                      </Badge>
                      <Badge variant="outline">{flag.flag_type}</Badge>
                      {flag.tags.map(tag => (
                        <Badge key={tag} variant="outline" className="text-xs">
                          {tag}
                        </Badge>
                      ))}
                    </div>
                    
                    <p className="text-gray-600 dark:text-gray-400 mb-3">
                      {flag.description}
                    </p>
                    
                    <div className="flex items-center gap-4">
                      <div className="flex items-center gap-2">
                        <Switch
                          checked={flag.status === 'active'}
                          onCheckedChange={(checked) => toggleFlag(flag.key, checked)}
                        />
                        <span className="text-sm">Active</span>
                      </div>
                      
                      {flag.flag_type === 'percentage' && (
                        <div className="flex items-center gap-2">
                          <span className="text-sm">Rollout:</span>
                          <div className="flex items-center gap-2">
                            <div className={`w-3 h-3 rounded-full ${getRolloutColor(flag.percentage)}`} />
                            <Input
                              type="number"
                              min="0"
                              max="100"
                              value={flag.percentage}
                              onChange={(e) => updateRolloutPercentage(flag.key, parseFloat(e.target.value))}
                              className="w-20 h-8"
                            />
                            <span className="text-sm">%</span>
                          </div>
                        </div>
                      )}
                      
                      {flag.flag_type === 'user_list' && flag.user_ids.length > 0 && (
                        <Badge variant="outline">
                          {flag.user_ids.length} user{flag.user_ids.length !== 1 ? 's' : ''}
                        </Badge>
                      )}
                      
                      {variants[flag.key] && variants[flag.key].length > 0 && (
                        <Badge variant="outline">
                          {variants[flag.key].length} variant{variants[flag.key].length !== 1 ? 's' : ''}
                        </Badge>
                      )}
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2 ml-4">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setSelectedFlag(flag)}
                    >
                      <Edit className="w-4 h-4" />
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => deleteFlag(flag.key)}
                      className="text-red-600 hover:text-red-700"
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
                
                <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
                  <div className="flex items-center justify-between text-sm text-gray-500">
                    <span>Created: {new Date(flag.created_at).toLocaleDateString()}</span>
                    <span>Updated: {new Date(flag.updated_at).toLocaleDateString()}</span>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="w-5 h-5" />
                Feature Flag Analytics
              </CardTitle>
              <CardDescription>
                Usage metrics and evaluation statistics
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">
                    {flags.filter(f => f.status === 'active').length}
                  </div>
                  <div className="text-sm text-gray-600">Active Flags</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">
                    {Math.round(flags.filter(f => f.flag_type === 'percentage').reduce((sum, f) => sum + f.percentage, 0) / Math.max(flags.filter(f => f.flag_type === 'percentage').length, 1)) || 0}%
                  </div>
                  <div className="text-sm text-gray-600">Avg Percentage Rollout</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-600">
                    {Object.values(variants).flat().length}
                  </div>
                  <div className="text-sm text-gray-600">Total Variants</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};