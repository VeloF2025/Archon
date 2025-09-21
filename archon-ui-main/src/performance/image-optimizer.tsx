/**
 * Image Optimizer
 *
 * Advanced image optimization with:
 * - Modern format conversion
 * - Responsive image loading
 * - Lazy loading strategies
 * - Quality optimization
 * - Placeholder generation
 */

import { useState, useRef, useEffect, useCallback } from 'react';

// Image optimization configuration
export interface ImageOptimizationConfig {
  enableWebP: boolean;
  enableAVIF: boolean;
  enableLazyLoading: boolean;
  enablePlaceholder: boolean;
  quality: number;
  maxWidth: number;
  formats: ('webp' | 'avif' | 'jpeg' | 'png')[];
  breakpoints: number[];
  placeholderQuality: number;
  blurAmount: number;
}

// Image source
export interface ImageSource {
  src: string;
  srcSet?: string;
  sizes?: string;
  type?: string;
}

// Optimized image props
export interface OptimizedImageProps extends React.ImgHTMLAttributes<HTMLImageElement> {
  src: string;
  alt: string;
  width?: number;
  height?: number;
  config?: Partial<ImageOptimizationConfig>;
  placeholder?: string;
  loadingStrategy?: 'lazy' | 'eager' | 'auto';
  decodingStrategy?: 'sync' | 'async' | 'auto';
  priority?: 'high' | 'medium' | 'low';
  onLoad?: () => void;
  onError?: () => void;
  onPlaceholderLoad?: () => void;
}

// Image optimization class
export class ImageOptimizer {
  private config: ImageOptimizationConfig;
  private cache: Map<string, string> = new Map();
  private observer: IntersectionObserver | null = null;
  private loadingImages: Set<string> = new Set();

  constructor(config: Partial<ImageOptimizationConfig> = {}) {
    this.config = {
      enableWebP: true,
      enableAVIF: true,
      enableLazyLoading: true,
      enablePlaceholder: true,
      quality: 85,
      maxWidth: 1920,
      formats: ['webp', 'jpeg'],
      breakpoints: [320, 640, 768, 1024, 1280, 1536, 1920],
      placeholderQuality: 20,
      blurAmount: 10,
      ...config,
    };

    this.initialize();
  }

  private initialize(): void {
    this.setupIntersectionObserver();
    this.detectBrowserSupport();
  }

  // Setup Intersection Observer for lazy loading
  private setupIntersectionObserver(): void {
    if (!('IntersectionObserver' in window)) return;

    this.observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const element = entry.target as HTMLImageElement;
            const src = element.dataset.src;
            if (src) {
              this.loadImage(element, src);
              this.observer?.unobserve(element);
            }
          }
        });
      },
      {
        rootMargin: '50px',
        threshold: 0.1,
      }
    );
  }

  // Detect browser support for modern formats
  private detectBrowserSupport(): void {
    // Check WebP support
    if (this.config.enableWebP) {
      this.config.enableWebP = this.checkWebPSupport();
    }

    // Check AVIF support
    if (this.config.enableAVIF) {
      this.config.enableAVIF = this.checkAVIFSupport();
    }
  }

  // Check WebP support
  private checkWebPSupport(): boolean {
    const canvas = document.createElement('canvas');
    return canvas.toDataURL('image/webp').indexOf('data:image/webp') === 0;
  }

  // Check AVIF support
  private checkAVIFSupport(): boolean {
    const canvas = document.createElement('canvas');
    try {
      return canvas.toDataURL('image/avif').indexOf('data:image/avif') === 0;
    } catch {
      return false;
    }
  }

  // Generate optimized image URL
  generateOptimizedUrl(
    src: string,
    options: {
      width?: number;
      height?: number;
      quality?: number;
      format?: string;
    } = {}
  ): string {
    const cacheKey = `${src}_${JSON.stringify(options)}`;
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey)!;
    }

    const url = new URL(src, window.location.origin);
    const params = new URLSearchParams(url.search);

    // Add optimization parameters
    if (options.width) params.set('w', options.width.toString());
    if (options.height) params.set('h', options.height.toString());
    if (options.quality) params.set('q', options.quality.toString());
    if (options.format) params.set('f', options.format);

    // Auto-format based on browser support
    if (!options.format) {
      if (this.config.enableAVIF) {
        params.set('f', 'avif');
      } else if (this.config.enableWebP) {
        params.set('f', 'webp');
      }
    }

    const optimizedUrl = `${url.origin}${url.pathname}?${params.toString()}`;
    this.cache.set(cacheKey, optimizedUrl);

    return optimizedUrl;
  }

  // Generate srcset for responsive images
  generateSrcSet(src: string): string {
    const srcsetEntries: string[] = [];

    this.config.breakpoints.forEach((width) => {
      const optimizedUrl = this.generateOptimizedUrl(src, { width });
      srcsetEntries.push(`${optimizedUrl} ${width}w`);
    });

    return srcsetEntries.join(', ');
  }

  // Generate sizes attribute
  generateSizes(breakpoints: { [key: string]: number }): string {
    const sizes: string[] = [];

    Object.entries(breakpoints).forEach(([breakpoint, width]) => {
      sizes.push(`(min-width: ${breakpoint}px) ${width}px`);
    });

    sizes.push('100vw');
    return sizes.join(', ');
  }

  // Generate placeholder image
  generatePlaceholder(src: string): Promise<string> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        if (!ctx) {
          reject(new Error('Canvas context not available'));
          return;
        }

        // Set small dimensions for placeholder
        canvas.width = 50;
        canvas.height = 50;

        // Draw scaled down image
        ctx.drawImage(img, 0, 0, 50, 50);

        // Apply blur effect
        ctx.filter = `blur(${this.config.blurAmount}px)`;
        ctx.drawImage(canvas, 0, 0);

        // Get low quality data URL
        resolve(canvas.toDataURL('image/jpeg', this.config.placeholderQuality / 100));
      };

      img.onerror = () => reject(new Error('Failed to load image'));
      img.src = src;
    });
  }

  // Load image with optimization
  loadImage(element: HTMLImageElement, src: string): void {
    if (this.loadingImages.has(src)) return;

    this.loadingImages.add(src);

    // Generate optimized URL
    const optimizedSrc = this.generateOptimizedUrl(src);

    // Generate srcset
    const srcset = this.generateSrcSet(src);

    // Generate sizes
    const sizes = this.generateSizes({
      '320': 320,
      '640': 640,
      '768': 768,
      '1024': 1024,
      '1280': 1280,
    });

    // Set attributes
    element.srcset = srcset;
    element.sizes = sizes;
    element.src = optimizedSrc;

    // Remove data-src attribute
    delete element.dataset.src;

    // Handle events
    element.onload = () => {
      this.loadingImages.delete(src);
      element.classList.add('loaded');
    };

    element.onerror = () => {
      this.loadingImages.delete(src);
      element.classList.add('error');
    };
  }

  // Preload image
  preloadImage(src: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve();
      img.onerror = () => reject(new Error('Failed to preload image'));
      img.src = this.generateOptimizedUrl(src);
    });
  }

  // Get image information
  async getImageInfo(src: string): Promise<{
    width: number;
    height: number;
    format: string;
    size: number;
  }> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        resolve({
          width: img.naturalWidth,
          height: img.naturalHeight,
          format: src.split('.').pop()?.toLowerCase() || 'unknown',
          size: 0, // Would need to fetch to get actual size
        });
      };
      img.onerror = () => reject(new Error('Failed to get image info'));
      img.src = src;
    });
  }

  // Get optimization recommendations
  getRecommendations(imageUrl: string): string[] {
    const recommendations: string[] = [];

    // Check if image is large
    const url = new URL(imageUrl);
    const width = parseInt(url.searchParams.get('w') || '0');
    const height = parseInt(url.searchParams.get('h') || '0');

    if (width > 1920 || height > 1080) {
      recommendations.push('Consider resizing large images for better performance');
    }

    // Check format
    const format = url.searchParams.get('f') || imageUrl.split('.').pop();
    if (format === 'png' && !imageUrl.includes('.png?')) {
      recommendations.push('Consider converting PNG to WebP for better compression');
    }

    // Check quality
    const quality = parseInt(url.searchParams.get('q') || '0');
    if (quality > 90) {
      recommendations.push('Consider reducing image quality for better compression');
    }

    return recommendations;
  }

  // Clear cache
  clearCache(): void {
    this.cache.clear();
  }

  // Destroy optimizer
  destroy(): void {
    if (this.observer) {
      this.observer.disconnect();
    }
    this.clearCache();
  }
}

// Optimized Image component
export const OptimizedImage: React.FC<OptimizedImageProps> = ({
  src,
  alt,
  width,
  height,
  config,
  placeholder,
  loadingStrategy = 'lazy',
  decodingStrategy = 'async',
  priority = 'medium',
  onLoad,
  onError,
  onPlaceholderLoad,
  className = '',
  style = {},
  ...props
}) => {
  const [imageState, setImageState] = useState<'placeholder' | 'loading' | 'loaded' | 'error'>('placeholder');
  const [placeholderUrl, setPlaceholderUrl] = useState<string | null>(null);
  const [optimizedSources, setOptimizedSources] = useState<ImageSource[]>([]);
  const imageRef = useRef<HTMLImageElement>(null);
  const optimizer = useRef<ImageOptimizer | null>(null);

  // Initialize optimizer
  useEffect(() => {
    optimizer.current = new ImageOptimizer(config);

    // Generate optimized sources
    const sources: ImageSource[] = [];

    if (optimizer.current.config.enableAVIF) {
      sources.push({
        src: optimizer.current.generateOptimizedUrl(src, { format: 'avif' }),
        type: 'image/avif',
      });
    }

    if (optimizer.current.config.enableWebP) {
      sources.push({
        src: optimizer.current.generateOptimizedUrl(src, { format: 'webp' }),
        type: 'image/webp',
      });
    }

    sources.push({
      src: optimizer.current.generateOptimizedUrl(src),
    });

    setOptimizedSources(sources);

    return () => {
      optimizer.current?.destroy();
    };
  }, [src, config]);

  // Generate placeholder
  useEffect(() => {
    if (placeholder || !optimizer.current?.config.enablePlaceholder) {
      setPlaceholderUrl(placeholder || null);
      return;
    }

    optimizer.current?.generatePlaceholder(src).then(setPlaceholderUrl).catch(() => {
      setPlaceholderUrl(null);
    });
  }, [src, placeholder]);

  // Setup lazy loading
  useEffect(() => {
    if (!imageRef.current || !optimizer.current || loadingStrategy !== 'lazy') return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            loadImage();
            observer.disconnect();
          }
        });
      },
      {
        rootMargin: '50px',
        threshold: 0.1,
      }
    );

    observer.observe(imageRef.current);

    return () => {
      observer.disconnect();
    };
  }, [loadingStrategy]);

  // Load image
  const loadImage = useCallback(() => {
    if (!imageRef.current || imageState !== 'placeholder') return;

    setImageState('loading');

    const img = new Image();
    img.onload = () => {
      setImageState('loaded');
      onLoad?.();
    };

    img.onerror = () => {
      setImageState('error');
      onError?.();
    };

    // Use optimized URL
    img.src = optimizedSources[optimizedSources.length - 1]?.src || src;
  }, [imageState, optimizedSources, src, onLoad, onError]);

  // Handle load immediately if not lazy
  useEffect(() => {
    if (loadingStrategy !== 'lazy' && imageState === 'placeholder') {
      loadImage();
    }
  }, [loadingStrategy, imageState, loadImage]);

  return (
    <div className={`optimized-image-container ${className}`} style={style}>
      {/* Placeholder */}
      {imageState === 'placeholder' && placeholderUrl && (
        <img
          src={placeholderUrl}
          alt=""
          className="absolute inset-0 w-full h-full object-cover transition-opacity duration-300"
          onLoad={onPlaceholderLoad}
        />
      )}

      {/* Loading state */}
      {imageState === 'loading' && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-100 dark:bg-gray-800">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        </div>
      )}

      {/* Error state */}
      {imageState === 'error' && (
        <div className="absolute inset-0 flex items-center justify-center bg-red-50 dark:bg-red-900/20">
          <div className="text-center">
            <div className="text-red-500 text-4xl mb-2">⚠️</div>
            <p className="text-red-600 dark:text-red-400">Failed to load image</p>
          </div>
        </div>
      )}

      {/* Actual image */}
      <picture>
        {optimizedSources.slice(0, -1).map((source, index) => (
          <source
            key={index}
            srcSet={source.srcSet || source.src}
            type={source.type}
          />
        ))}
        <img
          ref={imageRef}
          src={optimizedSources[optimizedSources.length - 1]?.src || src}
          alt={alt}
          width={width}
          height={height}
          data-src={src}
          loading={loadingStrategy}
          decoding={decodingStrategy}
          className={`w-full h-full object-cover transition-opacity duration-300 ${
            imageState === 'loaded' ? 'opacity-100' : 'opacity-0'
          }`}
          {...props}
        />
      </picture>
    </div>
  );
};

// Image gallery component
interface OptimizedImageGalleryProps {
  images: string[];
  imagesPerRow?: number;
  gap?: string;
  config?: Partial<ImageOptimizationConfig>;
  className?: string;
}

export const OptimizedImageGallery: React.FC<OptimizedImageGalleryProps> = ({
  images,
  imagesPerRow = 3,
  gap = '1rem',
  config,
  className = '',
}) => {
  return (
    <div
      className={`optimized-image-gallery ${className}`}
      style={{
        display: 'grid',
        gridTemplateColumns: `repeat(${imagesPerRow}, 1fr)`,
        gap,
      }}
    >
      {images.map((image, index) => (
        <div key={index} className="relative aspect-square overflow-hidden rounded-lg">
          <OptimizedImage
            src={image}
            alt={`Gallery image ${index + 1}`}
            config={config}
            className="w-full h-full"
          />
        </div>
      ))}
    </div>
  );
};

// React hook for image optimization
export function useImageOptimizer(config?: Partial<ImageOptimizationConfig>) {
  const optimizer = React.useMemo(() => new ImageOptimizer(config), [config]);

  React.useEffect(() => {
    return () => {
      optimizer.destroy();
    };
  }, [optimizer]);

  return optimizer;
}

// Image utilities
export const imageUtils = {
  // Generate color palette from image
  generateColorPalette: async (imageUrl: string, colorCount: number = 5): Promise<string[]> => {
    return new Promise((resolve) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        if (!ctx) {
          resolve([]);
          return;
        }

        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);

        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const pixels = imageData.data;

        const colorMap: { [key: string]: number } = {};

        // Sample every 10th pixel for performance
        for (let i = 0; i < pixels.length; i += 40) {
          const r = pixels[i];
          const g = pixels[i + 1];
          const b = pixels[i + 2];
          const alpha = pixels[i + 3];

          if (alpha > 128) {
            const key = `${Math.floor(r / 32) * 32},${Math.floor(g / 32) * 32},${Math.floor(b / 32) * 32}`;
            colorMap[key] = (colorMap[key] || 0) + 1;
          }
        }

        // Get top colors
        const sortedColors = Object.entries(colorMap)
          .sort(([, a], [, b]) => b - a)
          .slice(0, colorCount)
          .map(([color]) => `rgb(${color})`);

        resolve(sortedColors);
      };

      img.onerror = () => resolve([]);
      img.src = imageUrl;
    });
  },

  // Calculate aspect ratio
  getAspectRatio: (width: number, height: number): string => {
    const gcd = (a: number, b: number): number => b === 0 ? a : gcd(b, a % b);
    const divisor = gcd(width, height);
    return `${width / divisor}:${height / divisor}`;
  },

  // Calculate optimal image dimensions
  getOptimalDimensions: (
    originalWidth: number,
    originalHeight: number,
    maxWidth: number,
    maxHeight: number
  ): { width: number; height: number } => {
    const aspectRatio = originalWidth / originalHeight;

    if (originalWidth > maxWidth) {
      return {
        width: maxWidth,
        height: maxWidth / aspectRatio,
      };
    }

    if (originalHeight > maxHeight) {
      return {
        width: maxHeight * aspectRatio,
        height: maxHeight,
      };
    }

    return { width: originalWidth, height: originalHeight };
  },
};

export default OptimizedImage;