/**
 * Virtualization Manager
 *
 * High-performance virtualization for large datasets with:
 * - Dynamic windowing and viewport calculation
 * - Variable height support
 * - Smooth scrolling and performance
 * - Memory optimization
 * - Intersection Observer integration
 */

import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';

// Virtualization configuration
export interface VirtualizationConfig {
  itemHeight: number;
  containerHeight: number;
  overscanCount: number;
  enableDynamicSizing: boolean;
  estimatedItemHeight: number;
  enableSmoothScrolling: boolean;
  scrollThreshold: number;
}

// Virtual item interface
export interface VirtualItem {
  index: number;
  start: number;
  end: number;
  size: number;
  data: any;
}

// Position cache for dynamic sizing
class PositionCache {
  private cache: Map<number, number> = new Map();
  private lastMeasuredIndex: number = -1;

  set(index: number, size: number): void {
    this.cache.set(index, size);
    this.lastMeasuredIndex = Math.max(this.lastMeasuredIndex, index);
  }

  get(index: number): number | undefined {
    return this.cache.get(index);
  }

  has(index: number): boolean {
    return this.cache.has(index);
  }

  clear(): void {
    this.cache.clear();
    this.lastMeasuredIndex = -1;
  }

  getLastMeasuredIndex(): number {
    return this.lastMeasuredIndex;
  }
}

// Size estimator for dynamic items
class SizeEstimator {
  private measurements: Map<number, number> = new Map();
  private defaultHeight: number;

  constructor(defaultHeight: number) {
    this.defaultHeight = defaultHeight;
  }

  estimate(index: number): number {
    return this.measurements.get(index) || this.defaultHeight;
  }

  update(index: number, height: number): void {
    this.measurements.set(index, height);
  }

  getAverageHeight(): number {
    if (this.measurements.size === 0) return this.defaultHeight;

    const total = Array.from(this.measurements.values()).reduce((sum, height) => sum + height, 0);
    return total / this.measurements.size;
  }
}

// Virtual list component
interface VirtualListProps<T> {
  items: T[];
  renderItem: (item: T, index: number, style: React.CSSProperties) => React.ReactNode;
  itemHeight?: number;
  containerHeight?: number;
  overscanCount?: number;
  enableDynamicSizing?: boolean;
  className?: string;
  style?: React.CSSProperties;
  onScroll?: (scrollTop: number) => void;
  onItemsRendered?: (visibleItems: VirtualItem[]) => void;
}

export const VirtualList = React.memo(<T extends any>({
  items,
  renderItem,
  itemHeight = 50,
  containerHeight = 600,
  overscanCount = 5,
  enableDynamicSizing = true,
  className = '',
  style = {},
  onScroll,
  onItemsRendered,
}: VirtualListProps<T>) => {
  const [scrollTop, setScrollTop] = useState(0);
  const [isScrolling, setIsScrolling] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const positionCache = useRef(new PositionCache());
  const sizeEstimator = useRef(new SizeEstimator(itemHeight));
  const scrollTimeoutRef = useRef<NodeJS.Timeout>();

  // Calculate total height
  const totalHeight = useMemo(() => {
    if (!enableDynamicSizing) {
      return items.length * itemHeight;
    }

    let height = 0;
    const lastMeasured = positionCache.current.getLastMeasuredIndex();

    if (lastMeasured >= 0) {
      height = positionCache.current.get(lastMeasured)!;
    }

    for (let i = lastMeasured + 1; i < items.length; i++) {
      height += sizeEstimator.current.estimate(i);
    }

    return height;
  }, [items.length, itemHeight, enableDynamicSizing]);

  // Calculate visible range
  const { startIndex, endIndex } = useMemo(() => {
    const start = Math.max(0, scrollTop - overscanCount * itemHeight);
    const end = Math.min(totalHeight, scrollTop + containerHeight + overscanCount * itemHeight);

    let startIndex = 0;
    let endIndex = 0;

    if (!enableDynamicSizing) {
      startIndex = Math.floor(start / itemHeight);
      endIndex = Math.min(items.length - 1, Math.floor(end / itemHeight));
    } else {
      // Binary search for start index
      let low = 0;
      let high = items.length - 1;

      while (low <= high) {
        const mid = Math.floor((low + high) / 2);
        const midPos = positionCache.current.get(mid) || mid * sizeEstimator.current.estimate(0);

        if (midPos < start) {
          low = mid + 1;
        } else {
          high = mid - 1;
        }
      }

      startIndex = Math.max(0, low);

      // Binary search for end index
      low = 0;
      high = items.length - 1;

      while (low <= high) {
        const mid = Math.floor((low + high) / 2);
        const midPos = positionCache.current.get(mid) || mid * sizeEstimator.current.estimate(0);

        if (midPos < end) {
          low = mid + 1;
        } else {
          high = mid - 1;
        }
      }

      endIndex = Math.min(items.length - 1, low);
    }

    return { startIndex, endIndex };
  }, [scrollTop, containerHeight, overscanCount, itemHeight, totalHeight, items.length, enableDynamicSizing]);

  // Calculate item positions
  const visibleItems = useMemo(() => {
    const items: VirtualItem[] = [];
    let currentTop = 0;

    for (let i = 0; i < items.length; i++) {
      const size = enableDynamicSizing
        ? positionCache.current.get(i) || sizeEstimator.current.estimate(i)
        : itemHeight;

      if (i >= startIndex && i <= endIndex) {
        items.push({
          index: i,
          start: currentTop,
          end: currentTop + size,
          size,
          data: items[i],
        });
      }

      currentTop += size;
    }

    return items;
  }, [items, startIndex, endIndex, enableDynamicSizing, itemHeight]);

  // Handle scroll events
  const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    const scrollTop = e.currentTarget.scrollTop;
    setScrollTop(scrollTop);
    setIsScrolling(true);

    onScroll?.(scrollTop);

    // Clear existing timeout
    if (scrollTimeoutRef.current) {
      clearTimeout(scrollTimeoutRef.current);
    }

    // Set new timeout
    scrollTimeoutRef.current = setTimeout(() => {
      setIsScrolling(false);
    }, 150);
  }, [onScroll]);

  // Update item size
  const updateItemSize = useCallback((index: number, size: number) => {
    if (!enableDynamicSizing) return;

    const oldSize = positionCache.current.get(index);
    if (oldSize !== size) {
      positionCache.current.set(index, size);
      sizeEstimator.current.update(index, size);
    }
  }, [enableDynamicSizing]);

  // Handle resize observer for dynamic sizing
  useEffect(() => {
    if (!enableDynamicSizing) return;

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const index = parseInt(entry.target.getAttribute('data-index') || '0', 10);
        const height = entry.contentRect.height;
        updateItemSize(index, height);
      }
    });

    // Observe all visible items
    visibleItems.forEach(item => {
      const element = document.querySelector(`[data-index="${item.index}"]`);
      if (element) {
        resizeObserver.observe(element);
      }
    });

    return () => {
      resizeObserver.disconnect();
    };
  }, [visibleItems, enableDynamicSizing, updateItemSize]);

  // Notify parent of rendered items
  useEffect(() => {
    onItemsRendered?.(visibleItems);
  }, [visibleItems, onItemsRendered]);

  return (
    <div
      ref={containerRef}
      className={`relative overflow-auto ${className}`}
      style={{
        height: containerHeight,
        ...style,
      }}
      onScroll={handleScroll}
    >
      {/* Spacer for total height */}
      <div style={{ height: totalHeight, position: 'relative' }}>
        {/* Visible items */}
        {visibleItems.map(item => (
          <div
            key={item.index}
            data-index={item.index}
            style={{
              position: 'absolute',
              top: item.start,
              left: 0,
              right: 0,
              height: item.size,
              willChange: isScrolling ? 'transform' : 'auto',
              opacity: isScrolling ? 0.8 : 1,
              transition: isScrolling ? 'opacity 0.15s ease' : 'none',
            }}
          >
            {renderItem(items[item.index], item.index, {
              height: item.size,
              opacity: isScrolling ? 0.8 : 1,
              transition: isScrolling ? 'opacity 0.15s ease' : 'none',
            })}
          </div>
        ))}
      </div>
    </div>
  );
});

// Virtual grid component
interface VirtualGridProps<T> {
  items: T[];
  renderItem: (item: T, index: number, style: React.CSSProperties) => React.ReactNode;
  columnCount: number;
  itemHeight?: number;
  itemWidth?: number;
  containerHeight?: number;
  overscanCount?: number;
  gap?: number;
  className?: string;
  style?: React.CSSProperties;
  onScroll?: (scrollTop: number) => void;
}

export const VirtualGrid = React.memo(<T extends any>({
  items,
  renderItem,
  columnCount,
  itemHeight = 200,
  itemWidth = 200,
  containerHeight = 600,
  overscanCount = 3,
  gap = 10,
  className = '',
  style = {},
  onScroll,
}: VirtualGridProps<T>) => {
  const [scrollTop, setScrollTop] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);

  // Calculate row height with gap
  const rowHeight = itemHeight + gap;
  const totalRows = Math.ceil(items.length / columnCount);

  // Calculate total height
  const totalHeight = totalRows * rowHeight;

  // Calculate visible range
  const { startRow, endRow } = useMemo(() => {
    const start = Math.max(0, scrollTop - overscanCount * rowHeight);
    const end = Math.min(totalHeight, scrollTop + containerHeight + overscanCount * rowHeight);

    const startRow = Math.floor(start / rowHeight);
    const endRow = Math.min(totalRows - 1, Math.floor(end / rowHeight));

    return { startRow, endRow };
  }, [scrollTop, containerHeight, overscanCount, rowHeight, totalHeight, totalRows]);

  // Calculate visible items
  const visibleItems = useMemo(() => {
    const items: { index: number; row: number; col: number; data: T }[] = [];

    for (let row = startRow; row <= endRow; row++) {
      for (let col = 0; col < columnCount; col++) {
        const index = row * columnCount + col;
        if (index < items.length) {
          items.push({ index, row, col, data: items[index] });
        }
      }
    }

    return items;
  }, [items, startRow, endRow, columnCount]);

  // Handle scroll events
  const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    const scrollTop = e.currentTarget.scrollTop;
    setScrollTop(scrollTop);
    onScroll?.(scrollTop);
  }, [onScroll]);

  return (
    <div
      ref={containerRef}
      className={`relative overflow-auto ${className}`}
      style={{
        height: containerHeight,
        ...style,
      }}
      onScroll={handleScroll}
    >
      {/* Spacer for total height */}
      <div style={{ height: totalHeight, position: 'relative' }}>
        {/* Visible items */}
        {visibleItems.map(({ index, row, col, data }) => (
          <div
            key={index}
            style={{
              position: 'absolute',
              top: row * rowHeight,
              left: col * (itemWidth + gap),
              width: itemWidth,
              height: itemHeight,
            }}
          >
            {renderItem(data, index, {
              width: itemWidth,
              height: itemHeight,
            })}
          </div>
        ))}
      </div>
    </div>
  );
});

// Virtualization Manager class
export class VirtualizationManager {
  private config: VirtualizationConfig;

  constructor(config: Partial<VirtualizationConfig> = {}) {
    this.config = {
      itemHeight: 50,
      containerHeight: 600,
      overscanCount: 5,
      enableDynamicSizing: true,
      estimatedItemHeight: 50,
      enableSmoothScrolling: true,
      scrollThreshold: 100,
      ...config,
    };
  }

  createVirtualList<T>(items: T[], renderItem: (item: T, index: number, style: React.CSSProperties) => React.ReactNode) {
    return (
      <VirtualList
        items={items}
        renderItem={renderItem}
        itemHeight={this.config.itemHeight}
        containerHeight={this.config.containerHeight}
        overscanCount={this.config.overscanCount}
        enableDynamicSizing={this.config.enableDynamicSizing}
        estimatedItemHeight={this.config.estimatedItemHeight}
      />
    );
  }

  createVirtualGrid<T>(items: T[], renderItem: (item: T, index: number, style: React.CSSProperties) => React.ReactNode, columnCount: number) {
    return (
      <VirtualGrid
        items={items}
        renderItem={renderItem}
        columnCount={columnCount}
        itemHeight={this.config.itemHeight}
        containerHeight={this.config.containerHeight}
        overscanCount={this.config.overscanCount}
      />
    );
  }

  // Performance utilities
  getOptimalOverscanCount(itemCount: number, viewportSize: number): number {
    if (itemCount < 100) return 3;
    if (itemCount < 1000) return 5;
    return Math.min(10, Math.floor(viewportSize / this.config.itemHeight));
  }

  calculateVirtualizationMetrics(itemCount: number, viewportSize: number): {
    virtualItemCount: number;
    memorySavings: number;
    estimatedRenderTime: number;
  } {
    const virtualItemCount = Math.ceil(viewportSize / this.config.itemHeight) + 2 * this.config.overscanCount;
    const memorySavings = ((itemCount - virtualItemCount) / itemCount) * 100;
    const estimatedRenderTime = virtualItemCount * 0.1; // 0.1ms per item

    return {
      virtualItemCount,
      memorySavings,
      estimatedRenderTime,
    };
  }

  updateConfig(newConfig: Partial<VirtualizationConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }
}

export default VirtualizationManager;