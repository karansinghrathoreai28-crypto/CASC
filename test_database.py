"""
Azure Cosmos DB Connection Test
Tests database connectivity, read/write operations, and queries
"""

import yaml
import sys
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from datetime import datetime
import uuid
import json

def load_config():
    """Load configuration file"""
    try:
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        print("✓ Configuration file loaded")
        return config
    except Exception as e:
        print(f"✗ Failed to load configuration: {str(e)}")
        return None

def test_client_connection(config):
    """Test client connection"""
    print("\n--- Test 1: Client Connection ---")
    try:
        endpoint = config['cosmos_db']['endpoint']
        key = config['cosmos_db']['key']
        
        if not endpoint or endpoint == "YOUR_COSMOS_DB_ENDPOINT":
            print("✗ Cosmos DB endpoint not configured")
            return None
        
        if not key or key == "YOUR_COSMOS_DB_KEY":
            print("✗ Cosmos DB key not configured")
            return None
        
        print(f"  Endpoint: {endpoint}")
        
        client = CosmosClient(endpoint, key)
        print("✓ Client connection successful")
        return client
    except Exception as e:
        print(f"✗ Client connection failed: {str(e)}")
        return None

def test_database_operations(client, config):
    """Test database creation and access"""
    print("\n--- Test 2: Database Operations ---")
    try:
        database_name = config['cosmos_db']['database_name']
        print(f"  Database name: {database_name}")
        
        # Create or get database
        database = client.create_database_if_not_exists(id=database_name)
        print(f"✓ Database '{database_name}' accessible")
        
        # List databases
        databases = list(client.list_databases())
        print(f"✓ Total databases in account: {len(databases)}")
        
        return database
    except Exception as e:
        print(f"✗ Database operations failed: {str(e)}")
        return None

def test_container_operations(database, config):
    """Test container creation and access"""
    print("\n--- Test 3: Container Operations ---")
    try:
        container_name = config['cosmos_db']['container_name']
        print(f"  Container name: {container_name}")
        
        # Create or get container (without throughput for serverless)
        try:
            # First try without throughput (for serverless accounts)
            container = database.create_container_if_not_exists(
                id=container_name,
                partition_key=PartitionKey(path="/event_id")
            )
            print(f"✓ Container '{container_name}' created/accessed (serverless mode)")
        except exceptions.CosmosHttpResponseError as e:
            if "throughput" in str(e).lower():
                # If throughput error, try again without it
                container = database.create_container_if_not_exists(
                    id=container_name,
                    partition_key=PartitionKey(path="/event_id")
                )
                print(f"✓ Container '{container_name}' created/accessed (serverless mode - retry)")
            else:
                raise e
        
        return container
    except Exception as e:
        print(f"✗ Container operations failed: {str(e)}")
        return None

def test_write_operations(container):
    """Test writing documents"""
    print("\n--- Test 4: Write Operations ---")
    test_docs = []
    
    try:
        # Test 1: Simple document
        doc1 = {
            'id': 'test-write-1',
            'event_id': 'test-write-1',
            'type': 'test',
            'message': 'Simple write test',
            'timestamp': datetime.utcnow().isoformat()
        }
        container.upsert_item(body=doc1)
        test_docs.append(doc1['id'])
        print("✓ Simple document write successful")
        
        # Test 2: Complex document (like real event)
        doc2 = {
            'id': 'test-write-2',
            'event_id': 'test-write-2',
            'type': 'security_event',
            'timestamp': datetime.utcnow().isoformat(),
            'motion_level': 15000,
            'vision_analysis': {
                'description': 'test person detected',
                'confidence': 0.95,
                'tags': ['person', 'indoor'],
                'objects': [{'name': 'person', 'confidence': 0.95}],
                'faces': 1
            },
            'ai_summary': 'Test security event with person detection',
            'conversations': []
        }
        container.upsert_item(body=doc2)
        test_docs.append(doc2['id'])
        print("✓ Complex document write successful")
        
        # Test 3: Update existing document
        doc2['conversations'].append({
            'timestamp': datetime.utcnow().isoformat(),
            'question': 'Test question?',
            'answer': 'Test answer'
        })
        container.replace_item(item=doc2['id'], body=doc2)
        test_docs.append(doc2['id'])
        print("✓ Document update successful")
        
        return test_docs
    except Exception as e:
        print(f"✗ Write operations failed: {str(e)}")
        return test_docs

def test_read_operations(container, test_doc_ids):
    """Test reading documents"""
    print("\n--- Test 5: Read Operations ---")
    try:
        for doc_id in test_doc_ids[:2]:  # Read first 2 docs
            doc = container.read_item(
                item=doc_id,
                partition_key=doc_id
            )
            print(f"✓ Read document '{doc_id}': {doc.get('message', doc.get('type'))}")
        
        return True
    except Exception as e:
        print(f"✗ Read operations failed: {str(e)}")
        return False

def test_query_operations(container):
    """Test querying documents"""
    print("\n--- Test 6: Query Operations ---")
    try:
        # Query 1: Get all test documents
        query1 = "SELECT * FROM c WHERE c.type = 'test'"
        items1 = list(container.query_items(
            query=query1,
            enable_cross_partition_query=True
        ))
        print(f"✓ Query test documents: Found {len(items1)} items")
        
        # Query 2: Get security events
        query2 = "SELECT * FROM c WHERE c.type = 'security_event'"
        items2 = list(container.query_items(
            query=query2,
            enable_cross_partition_query=True
        ))
        print(f"✓ Query security events: Found {len(items2)} items")
        
        # Query 3: Get recent events (sorted by timestamp)
        query3 = "SELECT * FROM c WHERE c.type = 'security_event' ORDER BY c.timestamp DESC OFFSET 0 LIMIT 5"
        items3 = list(container.query_items(
            query=query3,
            enable_cross_partition_query=True
        ))
        print(f"✓ Query recent events: Found {len(items3)} items")
        
        return True
    except Exception as e:
        print(f"✗ Query operations failed: {str(e)}")
        return False

def test_delete_operations(container, test_doc_ids):
    """Test deleting documents"""
    print("\n--- Test 7: Delete Operations ---")
    deleted_count = 0
    
    for doc_id in test_doc_ids:
        try:
            container.delete_item(
                item=doc_id,
                partition_key=doc_id
            )
            deleted_count += 1
        except Exception as e:
            print(f"  ⚠ Could not delete {doc_id}: {str(e)}")
    
    print(f"✓ Deleted {deleted_count}/{len(test_doc_ids)} test documents")
    return deleted_count > 0

def test_performance(container):
    """Test performance with multiple operations"""
    print("\n--- Test 8: Performance Test ---")
    try:
        import time
        
        # Write test
        start = time.time()
        batch_size = 10
        batch_ids = []
        
        for i in range(batch_size):
            doc_id = f"perf-test-{uuid.uuid4()}"
            doc = {
                'id': doc_id,
                'event_id': doc_id,
                'type': 'performance_test',
                'index': i,
                'timestamp': datetime.utcnow().isoformat()
            }
            container.create_item(body=doc)
            batch_ids.append(doc_id)
        
        write_time = time.time() - start
        print(f"✓ Wrote {batch_size} documents in {write_time:.2f}s ({batch_size/write_time:.1f} docs/sec)")
        
        # Read test
        start = time.time()
        for doc_id in batch_ids:
            container.read_item(item=doc_id, partition_key=doc_id)
        read_time = time.time() - start
        print(f"✓ Read {batch_size} documents in {read_time:.2f}s ({batch_size/read_time:.1f} docs/sec)")
        
        # Cleanup
        for doc_id in batch_ids:
            try:
                container.delete_item(item=doc_id, partition_key=doc_id)
            except:
                pass
        print(f"✓ Cleanup completed")
        
        return True
    except Exception as e:
        print(f"✗ Performance test failed: {str(e)}")
        return False

def main():
    print("="*70)
    print("Azure Cosmos DB Connection Tester")
    print("="*70)
    
    # Load config
    config = load_config()
    if not config:
        sys.exit(1)
    
    # Run tests
    results = {}
    
    # Test 1: Client connection
    client = test_client_connection(config)
    results['client_connection'] = client is not None
    
    if not client:
        print("\n✗ Cannot proceed without client connection")
        sys.exit(1)
    
    # Test 2: Database operations
    database = test_database_operations(client, config)
    results['database_operations'] = database is not None
    
    if not database:
        print("\n✗ Cannot proceed without database access")
        sys.exit(1)
    
    # Test 3: Container operations
    container = test_container_operations(database, config)
    results['container_operations'] = container is not None
    
    if not container:
        print("\n✗ Cannot proceed without container access")
        sys.exit(1)
    
    # Test 4-8: CRUD operations
    test_doc_ids = test_write_operations(container)
    results['write_operations'] = len(test_doc_ids) > 0
    
    results['read_operations'] = test_read_operations(container, test_doc_ids)
    results['query_operations'] = test_query_operations(container)
    results['delete_operations'] = test_delete_operations(container, test_doc_ids)
    results['performance'] = test_performance(container)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL DATABASE TESTS PASSED")
        print("  Azure Cosmos DB is fully operational!")
    else:
        print("✗ SOME TESTS FAILED")
        print("  Check connection settings and firewall rules")
    print("="*70)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
