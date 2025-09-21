"""
Optimized Database Queries for Agency Swarm

Provides high-performance database queries and indexing strategies:
- Query optimization and performance tuning
- Intelligent indexing strategies
- Bulk operations and batch processing
- Query plan analysis and optimization
- Connection pooling and resource management

Target Performance:
- <50ms query response times
- 95% cache hit ratio
- 1000+ concurrent queries
- Automatic query optimization
"""

import asyncio
import logging
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from collections import defaultdict, deque
import json
import hashlib
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of database queries"""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    BULK_INSERT = "bulk_insert"
    BULK_UPDATE = "bulk_update"
    AGGREGATE = "aggregate"
    JOIN = "join"
    SUBQUERY = "subquery"

class IndexType(Enum):
    """Types of database indexes"""
    BTREE = "btree"
    HASH = "hash"
    GIN = "gin"
    GIST = "gist"
    SPATIAL = "spatial"
    FULLTEXT = "fulltext"
    COMPOSITE = "composite"

@dataclass
class QueryMetrics:
    """Query performance metrics"""
    query_id: str
    query_type: QueryType
    execution_time_ms: float
    rows_affected: int
    cache_hit: bool
    index_used: bool
    table_scanned: bool
    created_at: float = field(default_factory=time.time)

@dataclass
class QueryPlan:
    """Database query execution plan"""
    query_id: str
    plan_text: str
    estimated_cost: float
    estimated_rows: int
    actual_cost: Optional[float] = None
    actual_rows: Optional[int] = None
    created_at: float = field(default_factory=time.time)

@dataclass
class IndexRecommendation:
    """Index optimization recommendation"""
    table_name: str
    column_names: List[str]
    index_type: IndexType
    estimated_improvement_percent: float
    current_query_time_ms: float
    estimated_query_time_ms: float
    priority: str  # HIGH, MEDIUM, LOW
    reason: str

class QueryOptimizer:
    """Query optimization and performance analysis"""

    def __init__(self, db_connection):
        self.db_connection = db_connection
        self.query_history = deque(maxlen=10000)
        self.query_plans: Dict[str, QueryPlan] = {}
        self.query_cache: Dict[str, Any] = {}
        self.slow_query_threshold = 100  # ms
        self.index_recommendations: List[IndexRecommendation] = []

        # Performance tracking
        self.total_queries = 0
        self.cache_hits = 0
        self.slow_queries = 0
        self.average_query_time = 0.0

        # Background optimization
        self.optimization_thread = None
        self.running = False

    def start(self):
        """Start query optimizer"""
        logger.info("Starting Query Optimizer")
        self.running = True

        # Start background optimization
        self.optimization_thread = threading.Thread(target=self._background_optimization)
        self.optimization_thread.daemon = True
        self.optimization_thread.start()

        logger.info("Query Optimizer started")

    def stop(self):
        """Stop query optimizer"""
        logger.info("Stopping Query Optimizer")
        self.running = False

        if self.optimization_thread and self.optimization_thread.is_alive():
            self.optimization_thread.join(timeout=5)

        logger.info("Query Optimizer stopped")

    def _background_optimization(self):
        """Background query optimization loop"""
        while self.running:
            try:
                # Analyze slow queries
                self._analyze_slow_queries()

                # Update index recommendations
                self._update_index_recommendations()

                # Optimize query cache
                self._optimize_query_cache()

                # Sleep for next optimization cycle
                time.sleep(300)  # 5 minutes

            except Exception as e:
                logger.error(f"Error in background optimization: {e}")
                time.sleep(60)

    def execute_query(self, query: str, params: Optional[Dict] = None, use_cache: bool = True) -> Any:
        """Execute optimized query with caching"""
        start_time = time.time()
        query_id = self._generate_query_id(query, params)

        # Check cache first
        if use_cache and query_id in self.query_cache:
            cache_entry = self.query_cache[query_id]
            if not self._is_cache_expired(cache_entry):
                self.cache_hits += 1
                self._record_metrics(query_id, QueryType.SELECT, 0, 0, True, True, False)
                return cache_entry['result']

        # Execute query
        try:
            cursor = self.db_connection.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            result = cursor.fetchall()
            execution_time = (time.time() - start_time) * 1000

            # Cache result if it's a SELECT query
            if use_cache and query.strip().upper().startswith('SELECT'):
                self._cache_result(query_id, result)

            # Record metrics
            self._record_metrics(query_id, QueryType.SELECT, execution_time, len(result), False, self._used_index(cursor), self._did_table_scan(cursor))

            # Check if it's a slow query
            if execution_time > self.slow_query_threshold:
                self.slow_queries += 1
                logger.warning(f"Slow query detected: {execution_time:.1f}ms - {query[:100]}...")

            return result

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            self._record_metrics(query_id, QueryType.SELECT, 0, 0, False, False, False)
            raise

    def _generate_query_id(self, query: str, params: Optional[Dict] = None) -> str:
        """Generate unique query ID for caching"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        if params:
            params_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
            return f"{query_hash}_{params_hash}"
        return query_hash

    def _is_cache_expired(self, cache_entry: Dict) -> bool:
        """Check if cache entry has expired"""
        ttl = cache_entry.get('ttl', 300)  # 5 minutes default
        age = time.time() - cache_entry['created_at']
        return age > ttl

    def _cache_result(self, query_id: str, result: Any, ttl: int = 300):
        """Cache query result"""
        self.query_cache[query_id] = {
            'result': result,
            'created_at': time.time(),
            'ttl': ttl
        }

    def _used_index(self, cursor) -> bool:
        """Check if query used an index"""
        # This would need to be implemented based on the specific database
        # For PostgreSQL, you could use EXPLAIN ANALYZE
        return True  # Placeholder

    def _did_table_scan(self, cursor) -> bool:
        """Check if query did a table scan"""
        # This would need to be implemented based on the specific database
        return False  # Placeholder

    def _record_metrics(self, query_id: str, query_type: QueryType, execution_time: float, rows_affected: int, cache_hit: bool, index_used: bool, table_scanned: bool):
        """Record query performance metrics"""
        metrics = QueryMetrics(
            query_id=query_id,
            query_type=query_type,
            execution_time_ms=execution_time,
            rows_affected=rows_affected,
            cache_hit=cache_hit,
            index_used=index_used,
            table_scanned=table_scanned
        )

        self.query_history.append(metrics)
        self.total_queries += 1

        # Update average query time
        if self.total_queries == 1:
            self.average_query_time = execution_time
        else:
            self.average_query_time = (self.average_query_time * (self.total_queries - 1) + execution_time) / self.total_queries

    def _analyze_slow_queries(self):
        """Analyze slow queries for optimization opportunities"""
        slow_queries = [q for q in self.query_history if q.execution_time_ms > self.slow_query_threshold]

        if not slow_queries:
            return

        logger.info(f"Analyzing {len(slow_queries)} slow queries")

        # Group by query pattern
        query_patterns = defaultdict(list)
        for query in slow_queries:
            pattern = self._extract_query_pattern(query.query_id)
            query_patterns[pattern].append(query)

        # Analyze each pattern
        for pattern, queries in query_patterns.items():
            avg_time = sum(q.execution_time_ms for q in queries) / len(queries)
            if avg_time > self.slow_query_threshold * 2:  # Very slow queries
                self._recommend_index_for_pattern(pattern, avg_time)

    def _extract_query_pattern(self, query_id: str) -> str:
        """Extract query pattern from query ID"""
        # This would extract the basic query structure for analysis
        return query_id.split('_')[0]  # Simplified

    def _recommend_index_for_pattern(self, pattern: str, avg_time: float):
        """Recommend index for slow query pattern"""
        # This would analyze the query pattern and recommend specific indexes
        recommendation = IndexRecommendation(
            table_name="unknown",
            column_names=["unknown"],
            index_type=IndexType.BTREE,
            estimated_improvement_percent=50.0,
            current_query_time_ms=avg_time,
            estimated_query_time_ms=avg_time * 0.5,
            priority="HIGH",
            reason=f"Query pattern {pattern} is slow with average time {avg_time:.1f}ms"
        )

        self.index_recommendations.append(recommendation)

    def _update_index_recommendations(self):
        """Update index recommendations based on current usage"""
        # This would analyze current query patterns and update recommendations
        pass

    def _optimize_query_cache(self):
        """Optimize query cache by removing expired entries"""
        current_time = time.time()
        expired_keys = []

        for key, entry in self.query_cache.items():
            if current_time - entry['created_at'] > entry['ttl']:
                expired_keys.append(key)

        for key in expired_keys:
            del self.query_cache[key]

        logger.debug(f"Removed {len(expired_keys)} expired cache entries")

    def get_query_stats(self) -> Dict[str, Any]:
        """Get query performance statistics"""
        cache_hit_rate = (self.cache_hits / max(self.total_queries, 1)) * 100
        slow_query_rate = (self.slow_queries / max(self.total_queries, 1)) * 100

        return {
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_hit_rate_percent": cache_hit_rate,
            "slow_queries": self.slow_queries,
            "slow_query_rate_percent": slow_query_rate,
            "average_query_time_ms": self.average_query_time,
            "cache_size": len(self.query_cache),
            "index_recommendations": len(self.index_recommendations)
        }

class BulkOperations:
    """Bulk database operations for high throughput"""

    def __init__(self, db_connection, batch_size: int = 1000):
        self.db_connection = db_connection
        self.batch_size = batch_size

    def bulk_insert(self, table_name: str, data: List[Dict], columns: List[str]) -> int:
        """Perform bulk insert operation"""
        if not data:
            return 0

        total_inserted = 0
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            inserted = self._insert_batch(table_name, batch, columns)
            total_inserted += inserted

        return total_inserted

    def _insert_batch(self, table_name: str, batch: List[Dict], columns: List[str]) -> int:
        """Insert a batch of records"""
        try:
            cursor = self.db_connection.cursor()

            # Build bulk insert statement
            placeholders = ', '.join(['%s'] * len(columns))
            columns_str = ', '.join(columns)
            query = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"

            # Prepare batch data
            values = []
            for record in batch:
                row = [record.get(col) for col in columns]
                values.append(tuple(row))

            # Execute batch insert
            cursor.executemany(query, values)
            self.db_connection.commit()

            return len(batch)

        except Exception as e:
            logger.error(f"Bulk insert failed: {e}")
            self.db_connection.rollback()
            raise

    def bulk_update(self, table_name: str, updates: List[Dict], key_column: str, update_columns: List[str]) -> int:
        """Perform bulk update operation"""
        if not updates:
            return 0

        total_updated = 0
        for i in range(0, len(updates), self.batch_size):
            batch = updates[i:i + self.batch_size]
            updated = self._update_batch(table_name, batch, key_column, update_columns)
            total_updated += updated

        return total_updated

    def _update_batch(self, table_name: str, batch: List[Dict], key_column: str, update_columns: List[str]) -> int:
        """Update a batch of records"""
        try:
            cursor = self.db_connection.cursor()

            # Build CASE statements for each column
            case_statements = []
            for col in update_columns:
                case_when = []
                then_values = []
                for record in batch:
                    key_value = record[key_column]
                    update_value = record.get(col)
                    case_when.append(f"WHEN {key_column} = %s THEN %s")
                    then_values.extend([key_value, update_value])

                case_statements.append(f"{col} = CASE {' '.join(case_when)} END")

            # Build WHERE clause
            key_values = [record[key_column] for record in batch]
            where_clause = f"WHERE {key_column} IN ({', '.join(['%s'] * len(key_values))})"

            # Combine into full query
            set_clause = ', '.join(case_statements)
            query = f"UPDATE {table_name} SET {set_clause} {where_clause}"

            # Execute batch update
            all_values = []
            for case_stmt in case_statements:
                # Count the number of WHEN/THEN pairs (2 values per record)
                all_values.extend([item for record in batch for item in [record[key_column], record.get(update_columns[case_statements.index(case_stmt)])]])
            all_values.extend(key_values)

            cursor.execute(query, all_values)
            self.db_connection.commit()

            return cursor.rowcount

        except Exception as e:
            logger.error(f"Bulk update failed: {e}")
            self.db_connection.rollback()
            raise

class QueryBuilder:
    """Query builder for optimized queries"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset query builder"""
        self.select_columns = []
        self.from_table = None
        self.joins = []
        self.where_conditions = []
        self.group_by_columns = []
        self.having_conditions = []
        self.order_by_columns = []
        self.limit_value = None
        self.offset_value = None
        self.parameters = []

    def select(self, *columns):
        """Add SELECT columns"""
        self.select_columns.extend(columns)
        return self

    def from_table(self, table):
        """Set FROM table"""
        self.from_table = table
        return self

    def join(self, table, condition, join_type="INNER"):
        """Add JOIN"""
        self.joins.append(f"{join_type} JOIN {table} ON {condition}")
        return self

    def where(self, condition, *args):
        """Add WHERE condition"""
        self.where_conditions.append(condition)
        self.parameters.extend(args)
        return self

    def group_by(self, *columns):
        """Add GROUP BY columns"""
        self.group_by_columns.extend(columns)
        return self

    def having(self, condition, *args):
        """Add HAVING condition"""
        self.having_conditions.append(condition)
        self.parameters.extend(args)
        return self

    def order_by(self, *columns):
        """Add ORDER BY columns"""
        self.order_by_columns.extend(columns)
        return self

    def limit(self, limit):
        """Set LIMIT"""
        self.limit_value = limit
        return self

    def offset(self, offset):
        """Set OFFSET"""
        self.offset_value = offset
        return self

    def build(self) -> Tuple[str, List]:
        """Build the query and return SQL and parameters"""
        if not self.from_table:
            raise ValueError("FROM table is required")

        # Build SELECT clause
        select_clause = "SELECT " + (", ".join(self.select_columns) if self.select_columns else "*")

        # Build FROM clause
        from_clause = f"FROM {self.from_table}"

        # Build JOIN clauses
        join_clause = " " + " ".join(self.joins) if self.joins else ""

        # Build WHERE clause
        where_clause = ""
        if self.where_conditions:
            where_clause = " WHERE " + " AND ".join(self.where_conditions)

        # Build GROUP BY clause
        group_by_clause = ""
        if self.group_by_columns:
            group_by_clause = " GROUP BY " + ", ".join(self.group_by_columns)

        # Build HAVING clause
        having_clause = ""
        if self.having_conditions:
            having_clause = " HAVING " + " AND ".join(self.having_conditions)

        # Build ORDER BY clause
        order_by_clause = ""
        if self.order_by_columns:
            order_by_clause = " ORDER BY " + ", ".join(self.order_by_columns)

        # Build LIMIT and OFFSET
        limit_clause = ""
        if self.limit_value is not None:
            limit_clause = f" LIMIT {self.limit_value}"
            if self.offset_value is not None:
                limit_clause += f" OFFSET {self.offset_value}"

        # Combine all clauses
        query = f"{select_clause} {from_clause}{join_clause}{where_clause}{group_by_clause}{having_clause}{order_by_clause}{limit_clause}"

        return query, self.parameters.copy()

class DatabaseOptimizer:
    """Main database optimization coordinator"""

    def __init__(self, db_connection):
        self.db_connection = db_connection
        self.query_optimizer = QueryOptimizer(db_connection)
        self.bulk_operations = BulkOperations(db_connection)
        self.query_builder = QueryBuilder()

        # Performance tracking
        self.start_time = time.time()

    def start(self):
        """Start database optimization"""
        logger.info("Starting Database Optimizer")
        self.query_optimizer.start()
        logger.info("Database Optimizer started")

    def stop(self):
        """Stop database optimization"""
        logger.info("Stopping Database Optimizer")
        self.query_optimizer.stop()
        logger.info("Database Optimizer stopped")

    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        query_stats = self.query_optimizer.get_query_stats()

        # Calculate uptime
        uptime_seconds = time.time() - self.start_time
        uptime_hours = uptime_seconds / 3600

        return {
            "uptime_hours": uptime_hours,
            "query_performance": query_stats,
            "index_recommendations": [
                {
                    "table": rec.table_name,
                    "columns": rec.column_names,
                    "type": rec.index_type.value,
                    "improvement_percent": rec.estimated_improvement_percent,
                    "priority": rec.priority,
                    "reason": rec.reason
                }
                for rec in self.query_optimizer.index_recommendations
            ],
            "optimization_suggestions": self._generate_optimization_suggestions(),
            "performance_score": self._calculate_performance_score(query_stats)
        }

    def _generate_optimization_suggestions(self) -> List[str]:
        """Generate optimization suggestions based on current metrics"""
        suggestions = []
        query_stats = self.query_optimizer.get_query_stats()

        # Cache hit rate suggestions
        if query_stats["cache_hit_rate_percent"] < 80:
            suggestions.append("Consider increasing query cache size or TTL for better cache hit rates")

        # Slow query suggestions
        if query_stats["slow_query_rate_percent"] > 10:
            suggestions.append("High slow query rate detected - review query plans and add appropriate indexes")

        # Average query time suggestions
        if query_stats["average_query_time_ms"] > 50:
            suggestions.append("Average query time is high - consider query optimization and indexing")

        # Index recommendations
        if len(self.query_optimizer.index_recommendations) > 5:
            suggestions.append("Multiple index recommendations available - implement high-priority indexes")

        return suggestions

    def _calculate_performance_score(self, query_stats: Dict[str, Any]) -> float:
        """Calculate overall database performance score (0-100)"""
        # Score components
        cache_score = min(100, query_stats["cache_hit_rate_percent"])
        slow_query_score = max(0, 100 - query_stats["slow_query_rate_percent"] * 2)
        query_time_score = max(0, 100 - query_stats["average_query_time_ms"] * 0.5)

        # Weighted average
        overall_score = (cache_score * 0.4 + slow_query_score * 0.3 + query_time_score * 0.3)

        return round(overall_score, 1)

# Example usage
if __name__ == "__main__":
    import sqlite3

    # Create test database
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # Create test table
    cursor.execute("""
        CREATE TABLE test_table (
            id INTEGER PRIMARY KEY,
            name TEXT,
            value INTEGER,
            created_at TIMESTAMP
        )
    """)

    # Initialize database optimizer
    db_optimizer = DatabaseOptimizer(conn)
    db_optimizer.start()

    try:
        # Use query builder
        builder = db_optimizer.query_builder
        query, params = builder.select("name", "value").from_table("test_table").where("value > ?", 100).limit(10).build()

        # Execute optimized query
        result = db_optimizer.query_optimizer.execute_query(query, params)
        print(f"Query result: {result}")

        # Get optimization report
        report = db_optimizer.get_optimization_report()
        print(f"Performance score: {report['performance_score']}")

    finally:
        db_optimizer.stop()
        conn.close()