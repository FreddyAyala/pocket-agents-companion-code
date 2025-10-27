#!/usr/bin/env python3
"""
Chapter 11: Edge Management - Database Performance Tuning

This module demonstrates optimized database settings, WAL mode,
connection pooling, and batch operations for edge RAG systems.

Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence
"""

import sqlite3
import threading
import queue
import time
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import json

@dataclass
class DatabaseConfig:
    """Configuration for edge database optimization"""
    db_path: str
    max_connections: int = 10
    cache_size_mb: int = 64
    page_size: int = 4096
    journal_mode: str = "WAL"
    synchronous: str = "NORMAL"
    temp_store: str = "MEMORY"
    mmap_size_mb: int = 256

class EdgeDatabaseOptimizer:
    """
    Optimized database manager for edge RAG systems.
    
    Implements WAL mode, connection pooling, and batch operations
    for high-performance edge deployment.
    """
    
    def __init__(self, config: DatabaseConfig):
        """Initialize edge database optimizer"""
        self.config = config
        self.connection_pool = queue.Queue(maxsize=config.max_connections)
        self.lock = threading.Lock()
        self.connection_count = 0
        
        # Initialize connection pool
        self._initialize_connection_pool()
        
        # Create tables if they don't exist
        self._create_tables()
    
    def _initialize_connection_pool(self):
        """Initialize connection pool with optimized settings"""
        
        for _ in range(min(5, self.config.max_connections)):  # Start with 5 connections
            conn = self._create_optimized_connection()
            self.connection_pool.put(conn)
            self.connection_count += 1
        
        print(f"🗄️ Database connection pool initialized with {self.connection_count} connections")
    
    def _create_optimized_connection(self) -> sqlite3.Connection:
        """Create optimized SQLite connection"""
        
        conn = sqlite3.connect(
            self.config.db_path,
            check_same_thread=False,
            timeout=30.0
        )
        
        # Apply optimizations
        self._apply_optimizations(conn)
        
        return conn
    
    def _apply_optimizations(self, conn: sqlite3.Connection):
        """Apply SQLite optimizations for edge deployment"""
        
        # PRAGMA settings for performance
        pragmas = [
            f"PRAGMA journal_mode = {self.config.journal_mode}",
            f"PRAGMA synchronous = {self.config.synchronous}",
            f"PRAGMA cache_size = -{self.config.cache_size_mb * 1024}",  # Negative for KB
            f"PRAGMA temp_store = {self.config.temp_store}",
            f"PRAGMA mmap_size = {self.config.mmap_size_mb * 1024 * 1024}",
            f"PRAGMA page_size = {self.config.page_size}",
            "PRAGMA auto_vacuum = INCREMENTAL",
            "PRAGMA optimize",
        ]
        
        for pragma in pragmas:
            try:
                conn.execute(pragma)
            except sqlite3.Error as e:
                print(f"⚠️ Failed to apply PRAGMA {pragma}: {e}")
        
        # Create indexes for common queries
        self._create_indexes(conn)
    
    def _create_indexes(self, conn: sqlite3.Connection):
        """Create indexes for common queries"""
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON embeddings(vector_id)",
            "CREATE INDEX IF NOT EXISTS idx_embeddings_metadata ON embeddings(metadata)",
            "CREATE INDEX IF NOT EXISTS idx_documents_timestamp ON documents(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(type)",
            "CREATE INDEX IF NOT EXISTS idx_documents_content ON documents(content)",
        ]
        
        for index_sql in indexes:
            try:
                conn.execute(index_sql)
            except sqlite3.Error as e:
                print(f"⚠️ Failed to create index: {e}")
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        
        with self.get_connection() as conn:
            # Embeddings table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vector_id TEXT UNIQUE NOT NULL,
                    vector_data BLOB,
                    metadata TEXT,
                    timestamp REAL DEFAULT (julianday('now'))
                )
            """)
            
            # Documents table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT,
                    content TEXT,
                    type TEXT,
                    metadata TEXT,
                    timestamp REAL DEFAULT (julianday('now'))
                )
            """)
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool"""
        
        conn = None
        try:
            # Get connection from pool
            conn = self.connection_pool.get(timeout=10)
            yield conn
        finally:
            # Return connection to pool
            if conn:
                self.connection_pool.put(conn)
    
    def execute_query(self, query: str, params: Tuple = None) -> List[Tuple]:
        """Execute query with connection from pool"""
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            return cursor.fetchall()
    
    def execute_many(self, query: str, params_list: List[Tuple]) -> None:
        """Execute query multiple times with different parameters"""
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
    
    def batch_insert_embeddings(self, embeddings: List[Dict[str, Any]], batch_size: int = 1000) -> int:
        """Insert embeddings in batches for better performance"""
        
        if not embeddings:
            return 0
        
        total_inserted = 0
        
        # Process in batches
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i+batch_size]
            
            # Prepare batch data
            batch_data = []
            for embedding in batch:
                batch_data.append((
                    embedding.get('vector_id', ''),
                    embedding.get('vector_data', b''),
                    json.dumps(embedding.get('metadata', {})),
                    time.time()
                ))
            
            # Insert batch
            query = """
                INSERT OR REPLACE INTO embeddings (vector_id, vector_data, metadata, timestamp)
                VALUES (?, ?, ?, ?)
            """
            
            self.execute_many(query, batch_data)
            total_inserted += len(batch_data)
        
        return total_inserted
    
    def batch_insert_documents(self, documents: List[Dict[str, Any]], batch_size: int = 1000) -> int:
        """Insert documents in batches for better performance"""
        
        if not documents:
            return 0
        
        total_inserted = 0
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            
            # Prepare batch data
            batch_data = []
            for doc in batch:
                batch_data.append((
                    doc.get('title', ''),
                    doc.get('content', ''),
                    doc.get('type', 'text'),
                    json.dumps(doc.get('metadata', {})),
                    time.time()
                ))
            
            # Insert batch
            query = """
                INSERT OR REPLACE INTO documents (title, content, type, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """
            
            self.execute_many(query, batch_data)
            total_inserted += len(batch_data)
        
        return total_inserted
    
    def search_embeddings(self, vector_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search embeddings by vector ID"""
        
        query = """
            SELECT vector_id, vector_data, metadata, timestamp
            FROM embeddings
            WHERE vector_id LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """
        
        results = self.execute_query(query, (f"%{vector_id}%", limit))
        
        return [
            {
                'vector_id': row[0],
                'vector_data': row[1],
                'metadata': json.loads(row[2]) if row[2] else {},
                'timestamp': row[3]
            }
            for row in results
        ]
    
    def search_documents(self, query_text: str, doc_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Search documents by content"""
        
        if doc_type:
            query = """
                SELECT id, title, content, type, metadata, timestamp
                FROM documents
                WHERE content LIKE ? AND type = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            params = (f"%{query_text}%", doc_type, limit)
        else:
            query = """
                SELECT id, title, content, type, metadata, timestamp
                FROM documents
                WHERE content LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            params = (f"%{query_text}%", limit)
        
        results = self.execute_query(query, params)
        
        return [
            {
                'id': row[0],
                'title': row[1],
                'content': row[2],
                'type': row[3],
                'metadata': json.loads(row[4]) if row[4] else {},
                'timestamp': row[5]
            }
            for row in results
        ]
    
    def optimize_database(self) -> None:
        """Optimize database for better performance"""
        
        with self.get_connection() as conn:
            # Analyze tables
            conn.execute("ANALYZE")
            
            # Incremental vacuum
            conn.execute("PRAGMA incremental_vacuum")
            
            # Optimize
            conn.execute("PRAGMA optimize")
            
            print("✅ Database optimization completed")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        
        stats = {}
        
        with self.get_connection() as conn:
            # Get table sizes
            cursor = conn.cursor()
            
            # Count records in each table
            cursor.execute("SELECT COUNT(*) FROM embeddings")
            stats['embeddings_count'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM documents")
            stats['documents_count'] = cursor.fetchone()[0]
            
            # Get database size
            cursor.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            
            cursor.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]
            
            stats['database_size_bytes'] = page_count * page_size
            stats['database_size_mb'] = stats['database_size_bytes'] / (1024 * 1024)
            
            # Get cache size
            cursor.execute("PRAGMA cache_size")
            cache_size = cursor.fetchone()[0]
            stats['cache_size_kb'] = abs(cache_size)
        
        return stats

class BatchProcessor:
    """
    Batch processor for large dataset operations.
    
    Handles bulk inserts and updates with optimized
    transaction management for edge devices.
    """
    
    def __init__(self, db_optimizer: EdgeDatabaseOptimizer, batch_size: int = 1000):
        """Initialize batch processor"""
        self.db_optimizer = db_optimizer
        self.batch_size = batch_size
        self.processing_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.workers = []
        self.running = False
    
    def start_workers(self, num_workers: int = 4):
        """Start worker threads for batch processing"""
        
        self.running = True
        
        for i in range(num_workers):
            worker = threading.Thread(target=self._worker_thread, name=f"BatchWorker-{i}")
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
        
        print(f"🔄 Batch processor started with {num_workers} workers")
    
    def stop_workers(self):
        """Stop worker threads"""
        
        self.running = False
        
        # Send shutdown signals
        for _ in self.workers:
            self.processing_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join()
        
        self.workers.clear()
        print("⏹️ Batch processor stopped")
    
    def _worker_thread(self):
        """Worker thread for processing batches"""
        
        while self.running:
            try:
                # Get batch from queue
                batch = self.processing_queue.get(timeout=1)
                
                if batch is None:  # Shutdown signal
                    break
                
                # Process batch
                result = self._process_batch(batch)
                
                # Put result in queue
                self.result_queue.put(result)
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Batch worker error: {e}")
    
    def _process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single batch"""
        
        batch_type = batch['type']
        data = batch['data']
        
        if batch_type == 'embeddings':
            return self._process_embedding_batch(data)
        elif batch_type == 'documents':
            return self._process_document_batch(data)
        else:
            raise ValueError(f"Unknown batch type: {batch_type}")
    
    def _process_embedding_batch(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process embedding batch"""
        
        # Insert embeddings in batch
        inserted_count = self.db_optimizer.batch_insert_embeddings(data, self.batch_size)
        
        return {
            'type': 'embeddings',
            'processed': len(data),
            'inserted': inserted_count,
            'status': 'success'
        }
    
    def _process_document_batch(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process document batch"""
        
        # Insert documents in batch
        inserted_count = self.db_optimizer.batch_insert_documents(data, self.batch_size)
        
        return {
            'type': 'documents',
            'processed': len(data),
            'inserted': inserted_count,
            'status': 'success'
        }
    
    def submit_batch(self, batch_type: str, data: List[Dict[str, Any]]) -> None:
        """Submit batch for processing"""
        
        batch = {
            'type': batch_type,
            'data': data
        }
        
        self.processing_queue.put(batch)
    
    def get_results(self, timeout: float = None) -> List[Dict[str, Any]]:
        """Get results from processing"""
        
        results = []
        
        while True:
            try:
                result = self.result_queue.get(timeout=timeout)
                results.append(result)
            except queue.Empty:
                break
        
        return results
    
    def process_large_dataset(self, data: List[Dict[str, Any]], batch_type: str, progress_callback=None) -> List[Dict[str, Any]]:
        """Process large dataset in batches"""
        
        total_batches = (len(data) + self.batch_size - 1) // self.batch_size
        processed_batches = 0
        
        # Submit all batches
        for i in range(0, len(data), self.batch_size):
            batch_data = data[i:i+self.batch_size]
            self.submit_batch(batch_type, batch_data)
        
        # Collect results
        results = []
        while processed_batches < total_batches:
            batch_results = self.get_results(timeout=10)
            results.extend(batch_results)
            processed_batches += len(batch_results)
            
            if progress_callback:
                progress_callback(processed_batches, total_batches)
        
        return results

def demonstrate_database_optimization():
    """Demonstrate database optimization capabilities"""
    
    print("=" * 70)
    print("Chapter 11: Database Performance Tuning Demo")
    print("Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence")
    print("=" * 70)
    
    # Create database config
    config = DatabaseConfig(
        db_path="edge_rag_demo.db",
        max_connections=5,
        cache_size_mb=32,
        journal_mode="WAL",
        synchronous="NORMAL"
    )
    
    # Initialize database optimizer
    db_optimizer = EdgeDatabaseOptimizer(config)
    
    # Demo 1: Batch Operations
    print("\n📊 Demo 1: Batch Operations")
    print("-" * 40)
    
    # Generate sample data
    sample_embeddings = [
        {
            'vector_id': f'embed_{i:04d}',
            'vector_data': f'embedding_data_{i}'.encode(),
            'metadata': {'source': f'doc_{i}', 'dimension': 768}
        }
        for i in range(100)
    ]
    
    sample_documents = [
        {
            'title': f'Document {i}',
            'content': f'This is the content of document {i} with some important information.',
            'type': 'text',
            'metadata': {'category': 'demo', 'priority': i % 3}
        }
        for i in range(50)
    ]
    
    # Batch insert embeddings
    print("Inserting embeddings in batches...")
    start_time = time.time()
    embeddings_inserted = db_optimizer.batch_insert_embeddings(sample_embeddings, batch_size=20)
    embedding_time = time.time() - start_time
    
    print(f"  ✅ Inserted {embeddings_inserted} embeddings in {embedding_time:.2f}s")
    
    # Batch insert documents
    print("Inserting documents in batches...")
    start_time = time.time()
    documents_inserted = db_optimizer.batch_insert_documents(sample_documents, batch_size=10)
    document_time = time.time() - start_time
    
    print(f"  ✅ Inserted {documents_inserted} documents in {document_time:.2f}s")
    
    # Demo 2: Search Operations
    print("\n🔍 Demo 2: Search Operations")
    print("-" * 40)
    
    # Search embeddings
    print("Searching embeddings...")
    embedding_results = db_optimizer.search_embeddings('embed_', limit=5)
    print(f"  Found {len(embedding_results)} embedding results")
    
    # Search documents
    print("Searching documents...")
    document_results = db_optimizer.search_documents('document', limit=5)
    print(f"  Found {len(document_results)} document results")
    
    for result in document_results[:3]:
        print(f"    {result['title']}: {result['content'][:50]}...")
    
    # Demo 3: Batch Processor
    print("\n⚡ Demo 3: Batch Processor")
    print("-" * 40)
    
    batch_processor = BatchProcessor(db_optimizer, batch_size=25)
    batch_processor.start_workers(num_workers=2)
    
    # Process large dataset
    large_dataset = [
        {'vector_id': f'large_{i}', 'vector_data': f'data_{i}'.encode(), 'metadata': {}}
        for i in range(200)
    ]
    
    print("Processing large dataset with batch processor...")
    start_time = time.time()
    
    # Submit batches
    for i in range(0, len(large_dataset), 50):
        batch_data = large_dataset[i:i+50]
        batch_processor.submit_batch('embeddings', batch_data)
    
    # Collect results
    results = []
    while len(results) < 4:  # Expect 4 batches
        batch_results = batch_processor.get_results(timeout=5)
        results.extend(batch_results)
    
    processing_time = time.time() - start_time
    total_processed = sum(r['processed'] for r in results)
    
    print(f"  ✅ Processed {total_processed} items in {processing_time:.2f}s")
    print(f"  Throughput: {total_processed/processing_time:.1f} items/second")
    
    batch_processor.stop_workers()
    
    # Demo 4: Database Statistics
    print("\n📈 Demo 4: Database Statistics")
    print("-" * 40)
    
    stats = db_optimizer.get_database_stats()
    print(f"Database Statistics:")
    print(f"  Embeddings: {stats['embeddings_count']} records")
    print(f"  Documents: {stats['documents_count']} records")
    print(f"  Database Size: {stats['database_size_mb']:.2f} MB")
    print(f"  Cache Size: {stats['cache_size_kb']} KB")
    
    # Optimize database
    print("\nOptimizing database...")
    db_optimizer.optimize_database()
    
    # Clean up
    if os.path.exists(config.db_path):
        os.remove(config.db_path)
        print(f"🧹 Cleaned up demo database: {config.db_path}")
    
    print("\n💡 Key Takeaways:")
    print("  • WAL mode enables concurrent read/write operations")
    print("  • Connection pooling eliminates connection overhead")
    print("  • Batch operations provide order-of-magnitude speedup")
    print("  • Proper indexing optimizes query performance")
    print("=" * 70)

if __name__ == "__main__":
    demonstrate_database_optimization()
