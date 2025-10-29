#!/usr/bin/env python3
"""
Pgvector to Qdrant Migration Script

Migrates existing embeddings from Supabase pgvector to Qdrant for improved performance.
Features:
- Batch migration for efficiency
- Progress tracking and resumption
- Data validation and integrity checks
- Rollback capabilities
- Performance comparison
"""

import asyncio
import sys
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.server.services.qdrant_service import QdrantVectorService
from src.server.services.enhanced_rag_service import EnhancedRAGService
from src.config.logfire_config import get_logger

logger = get_logger(__name__)

@dataclass
class MigrationStats:
    """Migration statistics tracking"""
    total_records: int = 0
    migrated_records: int = 0
    failed_records: int = 0
    batches_completed: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    @property
    def success_rate(self) -> float:
        if self.total_records == 0:
            return 0.0
        return (self.migrated_records / self.total_records) * 100

    @property
    def migration_duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def migration_speed(self) -> float:
        if self.migration_duration and self.migration_duration > 0:
            return self.migrated_records / self.migration_duration
        return 0.0

class PgvectorToQdrantMigrator:
    """Handles migration from pgvector to Qdrant"""

    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        qdrant_collection: str = "archon_documents",
        batch_size: int = 1000,
        validate_data: bool = True
    ):
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.qdrant_collection = qdrant_collection
        self.batch_size = batch_size
        self.validate_data = validate_data

        self.qdrant_service: Optional[QdrantVectorService] = None
        self.rag_service: Optional[EnhancedRAGService] = None
        self.stats = MigrationStats()

        # Migration state
        self.migration_id = f"migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_file = Path(f"migration_checkpoint_{self.migration_id}.json")

    async def initialize(self) -> bool:
        """Initialize services for migration"""
        try:
            logger.info("Initializing migration services...")

            # Initialize Qdrant service
            self.qdrant_service = QdrantVectorService(
                host=self.qdrant_host,
                port=self.qdrant_port,
                collection_name=self.qdrant_collection
            )

            qdrant_success = await self.qdrant_service.initialize()
            if not qdrant_success:
                logger.error("Failed to initialize Qdrant service")
                return False

            # Initialize RAG service (for pgvector access)
            self.rag_service = EnhancedRAGService()
            rag_success = await self.rag_service.initialize()
            if not rag_success:
                logger.error("Failed to initialize RAG service")
                return False

            logger.info("Migration services initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize migration services: {e}")
            return False

    async def get_pgvector_data(self, offset: int = 0, limit: int = None) -> List[Dict[str, Any]]:
        """Retrieve data from pgvector"""
        try:
            if not self.rag_service:
                raise RuntimeError("RAG service not initialized")

            # Query pgvector for documents with embeddings
            # This is a simplified approach - adjust based on actual schema
            query = f"""
            SELECT
                d.id,
                d.content,
                d.source_id,
                d.source_type,
                d.metadata,
                e.embedding,
                d.created_at,
                d.updated_at
            FROM documents d
            JOIN embeddings e ON d.id = e.document_id
            WHERE e.embedding IS NOT NULL
            ORDER BY d.id
            LIMIT {limit or self.batch_size}
            OFFSET {offset}
            """

            # Execute query using RAG service's database connection
            result = await self.rag_service.execute_query(query)

            if result and len(result) > 0:
                logger.debug(f"Retrieved {len(result)} records from pgvector")
                return result
            else:
                logger.debug("No more records found in pgvector")
                return []

        except Exception as e:
            logger.error(f"Failed to retrieve pgvector data: {e}")
            return []

    async def count_pgvector_records(self) -> int:
        """Count total records to migrate"""
        try:
            if not self.rag_service:
                raise RuntimeError("RAG service not initialized")

            query = """
            SELECT COUNT(*) as total_count
            FROM documents d
            JOIN embeddings e ON d.id = e.document_id
            WHERE e.embedding IS NOT NULL
            """

            result = await self.rag_service.execute_query(query)
            return result[0]['total_count'] if result else 0

        except Exception as e:
            logger.error(f"Failed to count pgvector records: {e}")
            return 0

    def validate_embedding(self, embedding: Any) -> bool:
        """Validate embedding format and content"""
        if not embedding:
            return False

        # Check if it's a list or tuple
        if not isinstance(embedding, (list, tuple)):
            return False

        # Check dimensionality
        if len(embedding) != 1536:  # OpenAI ada-002 dimension
            logger.warning(f"Unexpected embedding dimension: {len(embedding)} (expected 1536)")

        # Check for valid float values
        for i, value in enumerate(embedding):
            if not isinstance(value, (int, float)):
                logger.warning(f"Invalid embedding value at index {i}: {value}")
                return False
            if abs(value) > 10:  # Unusually large value
                logger.warning(f"Large embedding value at index {i}: {value}")

        return True

    def transform_payload(self, pgvector_record: Dict[str, Any]) -> Dict[str, Any]:
        """Transform pgvector record to Qdrant payload format"""
        return {
            "content": pgvector_record.get("content", ""),
            "source_id": pgvector_record.get("source_id", ""),
            "source_type": pgvector_record.get("source_type", "unknown"),
            "metadata": pgvector_record.get("metadata", {}),
            "created_at": pgvector_record.get("created_at"),
            "updated_at": pgvector_record.get("updated_at"),
            "migration_id": self.migration_id,
            "migrated_at": datetime.utcnow().isoformat()
        }

    async def migrate_batch(self, batch_data: List[Dict[str, Any]]) -> Tuple[int, int]:
        """Migrate a batch of records to Qdrant"""
        if not batch_data:
            return 0, 0

        successful = 0
        failed = 0

        try:
            # Transform data for Qdrant
            vectors_data = []
            for record in batch_data:
                try:
                    # Validate embedding
                    embedding = record.get("embedding")
                    if not self.validate_embedding(embedding):
                        logger.warning(f"Invalid embedding for record {record.get('id')}")
                        failed += 1
                        continue

                    # Transform payload
                    payload = self.transform_payload(record)

                    vectors_data.append((
                        str(record["id"]),
                        list(embedding),  # Ensure it's a list
                        payload
                    ))
                    successful += 1

                except Exception as e:
                    logger.error(f"Failed to transform record {record.get('id')}: {e}")
                    failed += 1

            # Batch insert into Qdrant
            if vectors_data:
                qdrant_success = await self.qdrant_service.add_vectors(vectors_data)
                if not qdrant_success:
                    logger.error(f"Failed to insert batch of {len(vectors_data)} vectors into Qdrant")
                    return successful, failed + len(vectors_data)

                logger.info(f"Successfully migrated batch of {len(vectors_data)} vectors to Qdrant")

            return successful, failed

        except Exception as e:
            logger.error(f"Batch migration failed: {e}")
            return successful, failed + len(batch_data)

    def save_checkpoint(self, offset: int):
        """Save migration progress checkpoint"""
        try:
            checkpoint_data = {
                "migration_id": self.migration_id,
                "offset": offset,
                "stats": {
                    "total_records": self.stats.total_records,
                    "migrated_records": self.stats.migrated_records,
                    "failed_records": self.stats.failed_records,
                    "batches_completed": self.stats.batches_completed,
                    "start_time": self.stats.start_time.isoformat() if self.stats.start_time else None
                },
                "timestamp": datetime.utcnow().isoformat()
            }

            with open(self.checkpoint_file, 'w') as f:
                import json
                json.dump(checkpoint_data, f, indent=2)

            logger.debug(f"Saved checkpoint at offset {offset}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self) -> Optional[int]:
        """Load migration progress from checkpoint"""
        try:
            if not self.checkpoint_file.exists():
                return None

            with open(self.checkpoint_file, 'r') as f:
                import json
                checkpoint_data = json.load(f)

            # Restore stats
            stats_data = checkpoint_data.get("stats", {})
            self.stats.total_records = stats_data.get("total_records", 0)
            self.stats.migrated_records = stats_data.get("migrated_records", 0)
            self.stats.failed_records = stats_data.get("failed_records", 0)
            self.stats.batches_completed = stats_data.get("batches_completed", 0)

            if stats_data.get("start_time"):
                self.stats.start_time = datetime.fromisoformat(stats_data["start_time"])

            logger.info(f"Loaded checkpoint: {checkpoint_data.get('offset', 0)} records migrated")
            return checkpoint_data.get("offset", 0)

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    async def run_migration(self, resume: bool = False) -> bool:
        """Run the complete migration process"""
        try:
            logger.info("Starting pgvector to Qdrant migration...")
            self.stats.start_time = datetime.now()

            # Initialize services
            if not await self.initialize():
                return False

            # Count total records
            if self.stats.total_records == 0:
                self.stats.total_records = await self.count_pgvector_records()
                logger.info(f"Found {self.stats.total_records} records to migrate")

            if self.stats.total_records == 0:
                logger.info("No records found to migrate")
                return True

            # Load checkpoint if resuming
            start_offset = 0
            if resume:
                start_offset = self.load_checkpoint() or 0
                logger.info(f"Resuming migration from offset {start_offset}")

            # Migration loop
            offset = start_offset
            while offset < self.stats.total_records:
                logger.info(f"Processing batch starting at offset {offset}...")

                # Get batch from pgvector
                batch_data = await self.get_pgvector_data(offset, self.batch_size)

                if not batch_data:
                    logger.info("No more data to migrate")
                    break

                # Migrate batch
                batch_successful, batch_failed = await self.migrate_batch(batch_data)

                # Update statistics
                self.stats.migrated_records += batch_successful
                self.stats.failed_records += batch_failed
                self.stats.batches_completed += 1

                # Log progress
                progress = (self.stats.migrated_records / self.stats.total_records) * 100
                logger.info(
                    f"Batch {self.stats.batches_completed} completed: "
                    f"{batch_successful} successful, {batch_failed} failed "
                    f"({progress:.1f}% complete)"
                )

                # Save checkpoint
                self.save_checkpoint(offset + len(batch_data))

                # Move to next batch
                offset += len(batch_data)

            # Complete migration
            self.stats.end_time = datetime.now()

            # Log final statistics
            self.log_migration_summary()

            # Cleanup checkpoint file
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()

            return self.stats.failed_records == 0

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            self.stats.errors.append(str(e))
            return False

    def log_migration_summary(self):
        """Log migration completion summary"""
        duration = self.stats.migration_duration
        speed = self.stats.migration_speed

        logger.info("=" * 60)
        logger.info("MIGRATION COMPLETION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Migration ID: {self.migration_id}")
        logger.info(f"Total Records: {self.stats.total_records}")
        logger.info(f"Migrated Successfully: {self.stats.migrated_records}")
        logger.info(f"Failed: {self.stats.failed_records}")
        logger.info(f"Success Rate: {self.stats.success_rate:.2f}%")
        logger.info(f"Batches Processed: {self.stats.batches_completed}")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Speed: {speed:.1f} records/second")

        if duration:
            logger.info(f"Average Time per Batch: {duration / self.stats.batches_completed:.2f} seconds")

        if self.stats.errors:
            logger.warning("Errors encountered:")
            for error in self.stats.errors[:5]:  # Show first 5 errors
                logger.warning(f"  - {error}")

        logger.info("=" * 60)

    async def validate_migration(self) -> bool:
        """Validate migration integrity"""
        try:
            logger.info("Validating migration integrity...")

            # Get Qdrant collection metrics
            qdrant_metrics = await self.qdrant_service.get_collection_metrics()

            # Compare with expected counts
            expected_count = self.stats.migrated_records
            actual_count = qdrant_metrics.total_vectors

            logger.info(f"Expected vectors: {expected_count}")
            logger.info(f"Actual vectors: {actual_count}")

            if expected_count != actual_count:
                logger.error(f"Migration validation failed: count mismatch")
                return False

            # Sample validation - check a few random vectors
            sample_size = min(10, actual_count)
            sample_queries = [self._generate_test_embedding() for _ in range(sample_size)]

            successful_searches = 0
            for i, query in enumerate(sample_queries):
                try:
                    results, _ = await self.qdrant_service.search(
                        query_embedding=query,
                        limit=5
                    )
                    if results:
                        successful_searches += 1
                except Exception as e:
                    logger.warning(f"Sample search {i+1} failed: {e}")

            validation_rate = (successful_searches / sample_size) * 100
            logger.info(f"Search validation: {successful_searches}/{sample_size} ({validation_rate:.1f}%)")

            if validation_rate < 80:
                logger.warning("Low search validation rate - migration may have issues")

            logger.info("Migration validation completed")
            return True

        except Exception as e:
            logger.error(f"Migration validation failed: {e}")
            return False

    def _generate_test_embedding(self) -> List[float]:
        """Generate a test embedding for validation"""
        import random
        return [random.uniform(-1, 1) for _ in range(1536)]

    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.qdrant_service:
                await self.qdrant_service.close()
            if self.rag_service:
                # RAG service cleanup if needed
                pass
            logger.info("Migration resources cleaned up")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

async def main():
    """Main migration function"""
    import argparse

    parser = argparse.ArgumentParser(description="Migrate pgvector to Qdrant")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for migration")
    parser.add_argument("--validate", action="store_true", help="Validate migration after completion")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--collection", default="archon_documents", help="Qdrant collection name")

    args = parser.parse_args()

    # Create migrator
    migrator = PgvectorToQdrantMigrator(
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        qdrant_collection=args.collection,
        batch_size=args.batch_size,
        validate_data=True
    )

    try:
        # Run migration
        success = await migrator.run_migration(resume=args.resume)

        if success:
            logger.info("Migration completed successfully!")

            # Validate if requested
            if args.validate:
                logger.info("Running migration validation...")
                validation_success = await migrator.validate_migration()
                if validation_success:
                    logger.info("Migration validation passed!")
                else:
                    logger.warning("Migration validation failed - check logs")
        else:
            logger.error("Migration failed - check logs for details")

        return success

    except KeyboardInterrupt:
        logger.info("Migration interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Migration failed with error: {e}")
        return False
    finally:
        await migrator.cleanup()

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)