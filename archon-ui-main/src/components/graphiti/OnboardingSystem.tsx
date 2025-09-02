/**
 * Graphiti Explorer Onboarding System
 * Interactive guided tour with tooltips and progressive disclosure
 */

import React, { useState, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/Button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/badge';
import { 
  ChevronLeft, 
  ChevronRight, 
  X, 
  Play, 
  HelpCircle,
  Lightbulb,
  Mouse,
  Search,
  Target,
  Zap,
  Eye,
  Users,
  ArrowRight
} from 'lucide-react';

// Onboarding step definition
interface OnboardingStep {
  id: string;
  title: string;
  description: string;
  targetSelector?: string;
  position: 'top' | 'bottom' | 'left' | 'right' | 'center';
  icon: React.ReactNode;
  actions?: Array<{
    type: 'click' | 'hover' | 'wait';
    selector?: string;
    duration?: number;
    description: string;
  }>;
}

// Predefined onboarding steps for Graphiti Explorer
const ONBOARDING_STEPS: OnboardingStep[] = [
  {
    id: 'welcome',
    title: 'Welcome to Graphiti Explorer!',
    description: 'Discover connections and explore knowledge graphs with powerful visualization tools. Let us guide you through the key features.',
    position: 'center',
    icon: <Users className="w-6 h-6" />
  },
  {
    id: 'graph-overview',
    title: 'Knowledge Graph Visualization',
    description: 'This interactive graph shows entities (nodes) and their relationships (connections). Each entity has a confidence score and type.',
    targetSelector: '.react-flow__viewport, .graph-container',
    position: 'top',
    icon: <Target className="w-6 h-6" />
  },
  {
    id: 'entity-cards',
    title: 'Entity Cards',
    description: 'Colored cards represent different entity types: People, Organizations, Concepts, and Events. Click any entity to explore its details.',
    targetSelector: '[data-testid="rf__node"], .react-flow__node',
    position: 'right',
    icon: <Eye className="w-6 h-6" />,
    actions: [
      {
        type: 'click',
        selector: '[data-testid="rf__node"]:first-of-type, .react-flow__node:first-of-type',
        description: 'Try clicking on an entity card'
      }
    ]
  },
  {
    id: 'search-functionality',
    title: 'Smart Search',
    description: 'Use the search box to quickly find entities by name, type, or attributes. Search results update in real-time.',
    targetSelector: 'input[placeholder*="search" i], input[placeholder*="Search" i]',
    position: 'bottom',
    icon: <Search className="w-6 h-6" />,
    actions: [
      {
        type: 'click',
        selector: 'input[placeholder*="search" i], input[placeholder*="Search" i]',
        description: 'Try typing "Entity" to see search in action'
      }
    ]
  },
  {
    id: 'navigation-controls',
    title: 'Graph Navigation',
    description: 'Use mouse wheel to zoom, drag to pan, and use the controls to reset view or access the minimap for quick navigation.',
    targetSelector: '.react-flow__controls',
    position: 'left',
    icon: <Mouse className="w-6 h-6" />,
    actions: [
      {
        type: 'hover',
        selector: '.react-flow__controls',
        description: 'Hover over controls to see options'
      }
    ]
  },
  {
    id: 'performance-features',
    title: 'Performance Optimizations',
    description: 'The graph uses viewport culling and smart rendering for smooth performance, even with large datasets. Look for performance stats if available.',
    targetSelector: '[class*="performance"], [class*="stats"], button:contains("Stats")',
    position: 'top',
    icon: <Zap className="w-6 h-6" />
  },
  {
    id: 'exploration-tips',
    title: 'Exploration Tips',
    description: 'Double-click to fit view, right-click for context menus, and watch for real-time updates as the knowledge graph evolves.',
    position: 'center',
    icon: <Lightbulb className="w-6 h-6" />
  }
];

// Tooltip component for individual steps
interface TooltipProps {
  step: OnboardingStep;
  isVisible: boolean;
  onNext: () => void;
  onPrevious: () => void;
  onSkip: () => void;
  currentStepIndex: number;
  totalSteps: number;
  targetElement?: HTMLElement | null;
}

const OnboardingTooltip: React.FC<TooltipProps> = ({
  step,
  isVisible,
  onNext,
  onPrevious,
  onSkip,
  currentStepIndex,
  totalSteps,
  targetElement
}) => {
  const [position, setPosition] = useState({ top: 0, left: 0 });
  const tooltipRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isVisible && targetElement && tooltipRef.current) {
      const targetRect = targetElement.getBoundingClientRect();
      const tooltipRect = tooltipRef.current.getBoundingClientRect();
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;

      let top = 0;
      let left = 0;

      switch (step.position) {
        case 'top':
          top = targetRect.top - tooltipRect.height - 10;
          left = targetRect.left + (targetRect.width - tooltipRect.width) / 2;
          break;
        case 'bottom':
          top = targetRect.bottom + 10;
          left = targetRect.left + (targetRect.width - tooltipRect.width) / 2;
          break;
        case 'left':
          top = targetRect.top + (targetRect.height - tooltipRect.height) / 2;
          left = targetRect.left - tooltipRect.width - 10;
          break;
        case 'right':
          top = targetRect.top + (targetRect.height - tooltipRect.height) / 2;
          left = targetRect.right + 10;
          break;
        case 'center':
        default:
          top = viewportHeight / 2 - tooltipRect.height / 2;
          left = viewportWidth / 2 - tooltipRect.width / 2;
          break;
      }

      // Keep tooltip within viewport bounds
      top = Math.max(10, Math.min(top, viewportHeight - tooltipRect.height - 10));
      left = Math.max(10, Math.min(left, viewportWidth - tooltipRect.width - 10));

      setPosition({ top, left });
    }
  }, [isVisible, targetElement, step.position]);

  if (!isVisible) return null;

  return (
    <>
      {/* Backdrop overlay */}
      <div className="fixed inset-0 bg-black/40 z-40" />
      
      {/* Target highlight */}
      {targetElement && (
        <div
          className="fixed border-2 border-blue-400 bg-blue-100/20 rounded-lg z-50 pointer-events-none"
          style={{
            top: targetElement.getBoundingClientRect().top - 4,
            left: targetElement.getBoundingClientRect().left - 4,
            width: targetElement.getBoundingClientRect().width + 8,
            height: targetElement.getBoundingClientRect().height + 8,
          }}
        />
      )}
      
      {/* Tooltip card */}
      <div
        ref={tooltipRef}
        className="fixed z-50 max-w-sm bg-white shadow-xl border border-gray-200 rounded-lg"
        style={{
          top: `${position.top}px`,
          left: `${position.left}px`,
        }}
      >
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-50 text-blue-600 rounded-lg">
                {step.icon}
              </div>
              <div>
                <CardTitle className="text-lg">{step.title}</CardTitle>
                <Badge variant="outline" className="mt-1">
                  Step {currentStepIndex + 1} of {totalSteps}
                </Badge>
              </div>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={onSkip}
              className="h-8 w-8 p-0"
            >
              <X className="w-4 h-4" />
            </Button>
          </div>
        </CardHeader>
        
        <CardContent>
          <p className="text-gray-700 mb-4 leading-relaxed">
            {step.description}
          </p>
          
          {step.actions && step.actions.length > 0 && (
            <div className="mb-4 p-3 bg-blue-50 rounded-lg">
              <h4 className="text-sm font-medium text-blue-900 mb-2">
                Interactive Action:
              </h4>
              <p className="text-sm text-blue-700">
                {step.actions[0].description}
              </p>
            </div>
          )}
          
          <div className="flex items-center justify-between">
            <Button
              variant="outline"
              size="sm"
              onClick={onPrevious}
              disabled={currentStepIndex === 0}
              className="flex items-center gap-2"
            >
              <ChevronLeft className="w-4 h-4" />
              Previous
            </Button>
            
            <div className="flex gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={onSkip}
              >
                Skip Tour
              </Button>
              <Button
                size="sm"
                onClick={onNext}
                className="flex items-center gap-2"
              >
                {currentStepIndex === totalSteps - 1 ? 'Finish' : 'Next'}
                <ChevronRight className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </CardContent>
      </div>
    </>
  );
};

// Main onboarding system component
interface OnboardingSystemProps {
  isActive: boolean;
  onComplete: () => void;
  onSkip: () => void;
}

export const OnboardingSystem: React.FC<OnboardingSystemProps> = ({
  isActive,
  onComplete,
  onSkip
}) => {
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [targetElement, setTargetElement] = useState<HTMLElement | null>(null);

  const currentStep = ONBOARDING_STEPS[currentStepIndex];

  // Find target element for current step
  useEffect(() => {
    if (isActive && currentStep?.targetSelector) {
      const findTarget = () => {
        const element = document.querySelector(currentStep.targetSelector!) as HTMLElement;
        setTargetElement(element);
        return element;
      };

      // Try immediately, then retry with delays for dynamic content
      let element = findTarget();
      if (!element) {
        const retryTimeouts = [100, 500, 1000];
        retryTimeouts.forEach(delay => {
          setTimeout(() => {
            if (!element) {
              element = findTarget();
            }
          }, delay);
        });
      }
    } else {
      setTargetElement(null);
    }
  }, [isActive, currentStep]);

  const handleNext = () => {
    if (currentStepIndex < ONBOARDING_STEPS.length - 1) {
      setCurrentStepIndex(currentStepIndex + 1);
    } else {
      onComplete();
    }
  };

  const handlePrevious = () => {
    if (currentStepIndex > 0) {
      setCurrentStepIndex(currentStepIndex - 1);
    }
  };

  const handleSkip = () => {
    onSkip();
  };

  if (!isActive) return null;

  return (
    <OnboardingTooltip
      step={currentStep}
      isVisible={isActive}
      onNext={handleNext}
      onPrevious={handlePrevious}
      onSkip={handleSkip}
      currentStepIndex={currentStepIndex}
      totalSteps={ONBOARDING_STEPS.length}
      targetElement={targetElement}
    />
  );
};

// Help button to restart onboarding
interface HelpButtonProps {
  onStartOnboarding: () => void;
  className?: string;
}

export const HelpButton: React.FC<HelpButtonProps> = ({ 
  onStartOnboarding, 
  className = "" 
}) => {
  return (
    <Button
      variant="outline"
      size="sm"
      onClick={onStartOnboarding}
      className={`flex items-center gap-2 ${className}`}
      title="Start guided tour"
    >
      <HelpCircle className="w-4 h-4" />
      Help
    </Button>
  );
};

// Quick tips tooltip system (always visible hints)
interface QuickTipProps {
  message: string;
  targetSelector: string;
  position?: 'top' | 'bottom' | 'left' | 'right';
  show?: boolean;
}

export const QuickTip: React.FC<QuickTipProps> = ({
  message,
  targetSelector,
  position = 'top',
  show = true
}) => {
  const [targetElement, setTargetElement] = useState<HTMLElement | null>(null);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const element = document.querySelector(targetSelector) as HTMLElement;
    setTargetElement(element);
  }, [targetSelector]);

  useEffect(() => {
    if (show && targetElement) {
      const showTooltip = () => setIsVisible(true);
      const hideTooltip = () => setIsVisible(false);

      targetElement.addEventListener('mouseenter', showTooltip);
      targetElement.addEventListener('mouseleave', hideTooltip);

      return () => {
        targetElement.removeEventListener('mouseenter', showTooltip);
        targetElement.removeEventListener('mouseleave', hideTooltip);
      };
    }
  }, [show, targetElement]);

  if (!isVisible || !targetElement) return null;

  const rect = targetElement.getBoundingClientRect();
  let top = 0;
  let left = 0;

  switch (position) {
    case 'top':
      top = rect.top - 40;
      left = rect.left + rect.width / 2;
      break;
    case 'bottom':
      top = rect.bottom + 10;
      left = rect.left + rect.width / 2;
      break;
    case 'left':
      top = rect.top + rect.height / 2;
      left = rect.left - 10;
      break;
    case 'right':
      top = rect.top + rect.height / 2;
      left = rect.right + 10;
      break;
  }

  return (
    <div
      className="fixed z-30 px-3 py-2 text-sm bg-gray-900 text-white rounded-lg shadow-lg pointer-events-none transform -translate-x-1/2"
      style={{ top, left }}
    >
      {message}
      <div className="absolute top-full left-1/2 transform -translate-x-1/2 border-4 border-transparent border-t-gray-900" />
    </div>
  );
};