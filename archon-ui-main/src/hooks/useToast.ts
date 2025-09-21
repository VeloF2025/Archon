import { toast as sonnerToast, ToasterProps } from 'sonner';

/**
 * Custom toast hook that provides a consistent interface for notifications
 * Uses sonner under the hood for modern toast notifications
 */
export interface ToastOptions {
  title?: string;
  description?: string;
  variant?: 'default' | 'destructive' | 'success' | 'warning';
  duration?: number;
}

export function useToast() {
  const toast = ({
    title,
    description,
    variant = 'default',
    duration = 4000,
  }: ToastOptions) => {
    const message = title || description || '';
    const fullMessage = title && description ? `${title}: ${description}` : message;

    switch (variant) {
      case 'success':
        return sonnerToast.success(fullMessage, { duration });
      case 'destructive':
      case 'error':
        return sonnerToast.error(fullMessage, { duration });
      case 'warning':
        return sonnerToast.warning(fullMessage, { duration });
      default:
        return sonnerToast(fullMessage, { duration });
    }
  };

  // Provide direct access to sonner methods for advanced usage
  const toastMethods = {
    success: (message: string, options?: { duration?: number }) => 
      sonnerToast.success(message, options),
    error: (message: string, options?: { duration?: number }) => 
      sonnerToast.error(message, options),
    warning: (message: string, options?: { duration?: number }) => 
      sonnerToast.warning(message, options),
    info: (message: string, options?: { duration?: number }) => 
      sonnerToast.info(message, options),
    loading: (message: string) => 
      sonnerToast.loading(message),
    promise: <T>(
      promise: Promise<T>, 
      messages: { loading: string; success: string; error: string }
    ) => sonnerToast.promise(promise, messages),
    dismiss: (toastId?: string | number) => 
      sonnerToast.dismiss(toastId),
  };

  return {
    toast,
    ...toastMethods
  };
}

// For backward compatibility with components expecting { toast } destructuring
export default useToast;