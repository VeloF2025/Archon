import React, { useState, useEffect } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import { Button } from '@/components/ui/Button';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select';
import { DateTimePicker } from '@/components/ui/date-time-picker';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Calendar, Clock, Zap, TrendingUp, Filter, Activity } from 'lucide-react';
import graphExplorerService from '@/services/graphExplorerService';

interface TemporalFilterProps {
  onApply: (filter: {
    start_time: number;
    end_time: number;
    granularity: 'hour' | 'day' | 'week';
    entity_type?: string;
    pattern?: 'evolution' | 'trending';
  } | null) => void;
  onClose: () => void;
  currentFilter?: {
    start_time: number;
    end_time: number;
    granularity: string;
    entity_type?: string;
    pattern?: string;
  } | null;
}

interface QuickFilter {
  label: string;
  icon: React.ReactNode;
  description: string;
  getValue: () => { start_time: number; end_time: number; granularity: 'hour' | 'day' | 'week'; pattern?: 'evolution' | 'trending' };
}

export const TemporalFilter: React.FC<TemporalFilterProps> = ({
  onApply,
  onClose,
  currentFilter
}) => {
  const [filterType, setFilterType] = useState<'quick' | 'custom'>('quick');
  const [startTime, setStartTime] = useState<Date>(new Date(Date.now() - 24 * 60 * 60 * 1000));
  const [endTime, setEndTime] = useState<Date>(new Date());
  const [granularity, setGranularity] = useState<'hour' | 'day' | 'week'>('hour');
  const [entityType, setEntityType] = useState<string>('');
  const [pattern, setPattern] = useState<'evolution' | 'trending' | ''>('');
  const [selectedQuick, setSelectedQuick] = useState<string>('');

  // Initialize with current filter if exists
  useEffect(() => {
    if (currentFilter) {
      setStartTime(new Date(currentFilter.start_time));
      setEndTime(new Date(currentFilter.end_time));
      setGranularity(currentFilter.granularity as 'hour' | 'day' | 'week');
      setEntityType(currentFilter.entity_type || '');
      setPattern(currentFilter.pattern as 'evolution' | 'trending' || '');
      setFilterType('custom');
    }
  }, [currentFilter]);

  // Quick filter options
  const quickFilters: QuickFilter[] = [
    {
      label: 'Last Hour',
      icon: <Clock className="h-4 w-4" />,
      description: 'Show entities from the last hour',
      getValue: () => ({
        start_time: Date.now() - 60 * 60 * 1000,
        end_time: Date.now(),
        granularity: 'hour'
      })
    },
    {
      label: 'Last 24 Hours',
      icon: <Calendar className="h-4 w-4" />,
      description: 'Show entities from the last day',
      getValue: () => ({
        start_time: Date.now() - 24 * 60 * 60 * 1000,
        end_time: Date.now(),
        granularity: 'hour'
      })
    },
    {
      label: 'Last Week',
      icon: <TrendingUp className="h-4 w-4" />,
      description: 'Show entities from the last 7 days',
      getValue: () => ({
        start_time: Date.now() - 7 * 24 * 60 * 60 * 1000,
        end_time: Date.now(),
        granularity: 'day'
      })
    },
    {
      label: 'Last Month',
      icon: <Zap className="h-4 w-4" />,
      description: 'Show entities from the last 30 days',
      getValue: () => ({
        start_time: Date.now() - 30 * 24 * 60 * 60 * 1000,
        end_time: Date.now(),
        granularity: 'day'
      })
    },
    {
      label: 'Trending (24h)',
      icon: <Activity className="h-4 w-4" />,
      description: 'Show trending entities from last 24 hours',
      getValue: () => ({
        start_time: Date.now() - 24 * 60 * 60 * 1000,
        end_time: Date.now(),
        granularity: 'hour',
        pattern: 'trending' as 'trending'
      })
    },
    {
      label: 'Evolution (Week)',
      icon: <Filter className="h-4 w-4" />,
      description: 'Show entity evolution over the last week',
      getValue: () => ({
        start_time: Date.now() - 7 * 24 * 60 * 60 * 1000,
        end_time: Date.now(),
        granularity: 'day',
        pattern: 'evolution' as 'evolution'
      })
    }
  ];

  // Handle quick filter selection
  const handleQuickFilter = (filterId: string) => {
    setSelectedQuick(filterId);
    const filter = quickFilters.find(f => f.label === filterId);
    if (filter) {
      const filterValue = filter.getValue();
      setStartTime(new Date(filterValue.start_time));
      setEndTime(new Date(filterValue.end_time));
      setGranularity(filterValue.granularity);
      setPattern(filterValue.pattern || '');
    }
  };

  // Apply filter
  const handleApply = () => {
    if (filterType === 'quick' && selectedQuick) {
      const filter = quickFilters.find(f => f.label === selectedQuick);
      if (filter) {
        const filterValue = filter.getValue();
        onApply({
          ...filterValue,
          entity_type: entityType || undefined,
        });
        return;
      }
    }
    
    // Custom filter
    onApply({
      start_time: startTime.getTime(),
      end_time: endTime.getTime(),
      granularity,
      entity_type: entityType || undefined,
      pattern: pattern || undefined,
    });
  };

  // Clear filter
  const handleClear = () => {
    onApply(null);
  };

  // Format duration for display
  const formatDuration = (start: Date, end: Date): string => {
    const duration = end.getTime() - start.getTime();
    const days = Math.floor(duration / (24 * 60 * 60 * 1000));
    const hours = Math.floor((duration % (24 * 60 * 60 * 1000)) / (60 * 60 * 1000));
    const minutes = Math.floor((duration % (60 * 60 * 1000)) / (60 * 1000));

    if (days > 0) return `${days} day${days !== 1 ? 's' : ''}`;
    if (hours > 0) return `${hours} hour${hours !== 1 ? 's' : ''}`;
    return `${minutes} minute${minutes !== 1 ? 's' : ''}`;
  };

  return (
    <Dialog open={true} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center space-x-2">
            <Clock className="h-5 w-5 text-blue-600" />
            <span>Temporal Graph Filter</span>
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-6">
          {/* Filter Type Selection */}
          <div className="flex space-x-4">
            <Button
              variant={filterType === 'quick' ? 'default' : 'outline'}
              onClick={() => setFilterType('quick')}
              className="flex-1"
            >
              Quick Filters
            </Button>
            <Button
              variant={filterType === 'custom' ? 'default' : 'outline'}
              onClick={() => setFilterType('custom')}
              className="flex-1"
            >
              Custom Range
            </Button>
          </div>

          {/* Quick Filters */}
          {filterType === 'quick' && (
            <div>
              <div className="grid grid-cols-2 gap-3 mb-4">
                {quickFilters.map((filter) => (
                  <div
                    key={filter.label}
                    className={`p-4 border rounded-lg cursor-pointer transition-all hover:bg-gray-50 ${
                      selectedQuick === filter.label
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200'
                    }`}
                    onClick={() => handleQuickFilter(filter.label)}
                  >
                    <div className="flex items-center space-x-2 mb-2">
                      {filter.icon}
                      <span className="font-medium">{filter.label}</span>
                    </div>
                    <p className="text-sm text-gray-600">{filter.description}</p>
                  </div>
                ))}
              </div>
              
              {/* Additional filters for quick mode */}
              <div className="space-y-3 border-t pt-4">
                <div className="space-y-2">
                  <Label htmlFor="entity-type-quick">Entity Type (Optional)</Label>
                  <Select value={entityType} onValueChange={setEntityType}>
                    <SelectTrigger>
                      <SelectValue placeholder="All entity types" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="">All Types</SelectItem>
                      {graphExplorerService.getEntityTypeOptions()
                        .filter(option => option.value !== 'all')
                        .map(option => (
                          <SelectItem key={option.value} value={option.value}>
                            {option.label}
                          </SelectItem>
                        ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>
          )}

          {/* Custom Range */}
          {filterType === 'custom' && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="start-time">Start Time</Label>
                  <DateTimePicker
                    date={startTime}
                    setDate={setStartTime}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="end-time">End Time</Label>
                  <DateTimePicker
                    date={endTime}
                    setDate={setEndTime}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="granularity">Time Granularity</Label>
                  <Select value={granularity} onValueChange={setGranularity}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="hour">Hour</SelectItem>
                      <SelectItem value="day">Day</SelectItem>
                      <SelectItem value="week">Week</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="pattern">Analysis Pattern</Label>
                  <Select value={pattern} onValueChange={setPattern}>
                    <SelectTrigger>
                      <SelectValue placeholder="No pattern" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="">No Pattern</SelectItem>
                      <SelectItem value="trending">Trending</SelectItem>
                      <SelectItem value="evolution">Evolution</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="entity-type">Entity Type Filter</Label>
                <Select value={entityType} onValueChange={setEntityType}>
                  <SelectTrigger>
                    <SelectValue placeholder="All entity types" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="">All Types</SelectItem>
                    {graphExplorerService.getEntityTypeOptions()
                      .filter(option => option.value !== 'all')
                      .map(option => (
                        <SelectItem key={option.value} value={option.value}>
                          {option.label}
                        </SelectItem>
                      ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Duration Display */}
              <div className="p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Duration:</span>
                  <Badge variant="secondary">
                    {formatDuration(startTime, endTime)}
                  </Badge>
                </div>
              </div>
            </div>
          )}

          {/* Current Filter Display */}
          {currentFilter && (
            <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-blue-900">Current Filter Active</span>
                <Badge variant="outline" className="text-blue-700 border-blue-300">
                  {formatDuration(
                    new Date(currentFilter.start_time),
                    new Date(currentFilter.end_time)
                  )}
                </Badge>
              </div>
              <p className="text-xs text-blue-700 mt-1">
                {new Date(currentFilter.start_time).toLocaleString()} → {' '}
                {new Date(currentFilter.end_time).toLocaleString()}
              </p>
            </div>
          )}

          {/* Filter Preview */}
          {(filterType === 'custom' || selectedQuick) && (
            <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
              <div className="flex items-center space-x-2 mb-2">
                <TrendingUp className="h-4 w-4 text-green-600" />
                <span className="text-sm font-medium text-green-900">Filter Preview</span>
              </div>
              <div className="text-xs text-green-700 space-y-1">
                <div>From: {startTime.toLocaleString()}</div>
                <div>To: {endTime.toLocaleString()}</div>
                <div>Duration: {formatDuration(startTime, endTime)}</div>
                <div>Granularity: {granularity}</div>
                {entityType && <div>Entity Type: {entityType}</div>}
                {pattern && <div>Pattern: {pattern}</div>}
              </div>
            </div>
          )}
        </div>

        <DialogFooter className="flex justify-between">
          <div>
            {currentFilter && (
              <Button variant="outline" onClick={handleClear}>
                Clear Filter
              </Button>
            )}
          </div>
          <div className="space-x-2">
            <Button variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button 
              onClick={handleApply}
              disabled={filterType === 'quick' && !selectedQuick}
            >
              Apply Filter
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};