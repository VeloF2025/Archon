import { useState, useEffect } from 'react';

interface MigrationStatus {
  migrationRequired: boolean;
  message?: string;
  loading: boolean;
}

export const useMigrationStatus = (): MigrationStatus => {
  const [status, setStatus] = useState<MigrationStatus>({
    migrationRequired: false,
    loading: true,
  });

  useEffect(() => {
    const checkMigrationStatus = async () => {
      try {
        const response = await fetch('/api/health', {
          signal: AbortSignal.timeout(15000) // 15 second timeout
        });
        const healthData = await response.json();
        
        if (healthData.status === 'migration_required') {
          setStatus({
            migrationRequired: true,
            message: healthData.message,
            loading: false,
          });
        } else {
          setStatus({
            migrationRequired: false,
            loading: false,
          });
        }
      } catch (error) {
        console.error('Failed to check migration status:', error);
        setStatus({
          migrationRequired: false,
          loading: false,
        });
      }
    };

    // Add small delay to stagger with other API calls on page load
    setTimeout(checkMigrationStatus, 1000);
    
    // Check periodically (every 60 seconds) to reduce server load
    const interval = setInterval(checkMigrationStatus, 60000);
    
    return () => clearInterval(interval);
  }, []);

  return status;
};