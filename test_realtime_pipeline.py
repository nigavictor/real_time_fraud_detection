#!/usr/bin/env python3
"""
Real-Time Fraud Detection Pipeline Test
======================================

This script tests the complete real-time fraud detection pipeline end-to-end,
validating data flow from transaction generation through fraud detection to alerts.

Test Coverage:
- Transaction generation and Kafka publishing
- Feature engineering and model inference  
- Fraud alert generation and storage
- Dashboard API endpoints
- Performance benchmarks
"""

import os
import sys
import json
import time
import asyncio
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any
import subprocess

import redis
import psycopg2
from psycopg2.extras import RealDictCursor
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError

class RealTimePipelineTest:
    """Test suite for real-time fraud detection pipeline."""
    
    def __init__(self):
        self.test_results = {
            'start_time': datetime.now(),
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'errors': []
        }
        
        # Connection configs
        self.kafka_servers = 'localhost:9092'
        self.redis_host = 'localhost'
        self.postgres_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'fraud_detection',
            'user': 'fraud_user',
            'password': 'fraud_password'
        }
        self.dashboard_url = 'http://localhost:8080'
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete test suite."""
        print("🧪 REAL-TIME FRAUD DETECTION PIPELINE TESTS")
        print("="*60)
        print(f"Test started at: {self.test_results['start_time']}")
        print()
        
        # Test sequence
        tests = [
            ("Infrastructure Connectivity", self.test_infrastructure_connectivity),
            ("Kafka Topics and Messages", self.test_kafka_functionality),
            ("Redis Cache Operations", self.test_redis_functionality),
            ("PostgreSQL Database", self.test_postgres_functionality),
            ("Dashboard API Endpoints", self.test_dashboard_api),
            ("End-to-End Transaction Flow", self.test_end_to_end_flow),
            ("Performance Benchmarks", self.test_performance_benchmarks),
            ("System Health Monitoring", self.test_health_monitoring)
        ]
        
        for test_name, test_func in tests:
            self._run_test(test_name, test_func)
        
        return self._generate_test_report()
    
    def _run_test(self, test_name: str, test_func):
        """Run individual test with error handling."""
        print(f"🔍 Testing: {test_name}")
        
        try:
            self.test_results['tests_run'] += 1
            result = test_func()
            
            if result:
                print(f"✅ {test_name}: PASSED")
                self.test_results['tests_passed'] += 1
            else:
                print(f"❌ {test_name}: FAILED")
                self.test_results['tests_failed'] += 1
                
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {str(e)}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append({
                'test': test_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
        
        print()
    
    def test_infrastructure_connectivity(self) -> bool:
        """Test connectivity to all infrastructure components."""
        success = True
        
        # Test Docker containers are running
        try:
            result = subprocess.run(['docker-compose', 'ps'], 
                                  capture_output=True, text=True, check=True)
            
            required_services = ['kafka', 'zookeeper', 'redis', 'postgres', 
                               'fraud-detector', 'transaction-generator', 'dashboard']
            
            running_services = []
            for line in result.stdout.split('\n'):
                for service in required_services:
                    if service in line and 'Up' in line:
                        running_services.append(service)
            
            print(f"  Running services: {len(set(running_services))}/{len(required_services)}")
            
            if len(set(running_services)) < len(required_services):
                print(f"  Missing services: {set(required_services) - set(running_services)}")
                success = False
                
        except subprocess.CalledProcessError as e:
            print(f"  Docker compose error: {e.stderr}")
            success = False
        
        return success
    
    def test_kafka_functionality(self) -> bool:
        """Test Kafka producer and consumer functionality."""
        try:
            # Test producer
            producer = KafkaProducer(
                bootstrap_servers=self.kafka_servers,
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                request_timeout_ms=10000
            )
            
            test_message = {
                'test_id': f'test_{int(time.time())}',
                'timestamp': datetime.now().isoformat(),
                'message': 'Kafka connectivity test'
            }
            
            # Send test message
            future = producer.send('fraud-transactions', value=test_message)
            record_metadata = future.get(timeout=10)
            
            print(f"  Message sent to {record_metadata.topic}[{record_metadata.partition}]")
            
            producer.close()
            
            # Test consumer
            consumer = KafkaConsumer(
                'fraud-transactions',
                bootstrap_servers=self.kafka_servers,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                consumer_timeout_ms=5000,
                auto_offset_reset='latest'
            )
            
            # Check if we can connect
            partitions = consumer.partitions_for_topic('fraud-transactions')
            print(f"  Topic partitions: {len(partitions) if partitions else 0}")
            
            consumer.close()
            
            return True
            
        except Exception as e:
            print(f"  Kafka test failed: {e}")
            return False
    
    def test_redis_functionality(self) -> bool:
        """Test Redis cache operations."""
        try:
            r = redis.Redis(host=self.redis_host, port=6379, decode_responses=True)
            
            # Test connection
            r.ping()
            print("  Redis connection: OK")
            
            # Test basic operations
            test_key = f'test_key_{int(time.time())}'
            test_value = 'test_value'
            
            r.set(test_key, test_value, ex=30)  # 30 second expiry
            retrieved_value = r.get(test_key)
            
            if retrieved_value == test_value:
                print("  Redis read/write: OK")
            else:
                print("  Redis read/write: FAILED")
                return False
            
            # Test hash operations (used by feature engineering)
            hash_key = f'test_hash_{int(time.time())}'
            r.hset(hash_key, mapping={'field1': 'value1', 'field2': 'value2'})
            hash_data = r.hgetall(hash_key)
            
            if len(hash_data) == 2:
                print("  Redis hash operations: OK")
            else:
                print("  Redis hash operations: FAILED")
                return False
            
            # Cleanup
            r.delete(test_key, hash_key)
            
            return True
            
        except Exception as e:
            print(f"  Redis test failed: {e}")
            return False
    
    def test_postgres_functionality(self) -> bool:
        """Test PostgreSQL database operations."""
        try:
            conn = psycopg2.connect(**self.postgres_config)
            
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Test connection
                cursor.execute('SELECT 1 as test')
                result = cursor.fetchone()
                
                if result['test'] == 1:
                    print("  PostgreSQL connection: OK")
                else:
                    return False
                
                # Test required tables exist
                cursor.execute("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """)
                
                tables = [row['table_name'] for row in cursor.fetchall()]
                required_tables = ['fraud_alerts', 'real_time_metrics', 'model_performance']
                
                missing_tables = set(required_tables) - set(tables)
                if missing_tables:
                    print(f"  Missing tables: {missing_tables}")
                    return False
                else:
                    print(f"  Required tables: OK ({len(required_tables)} found)")
                
                # Test insert operation
                cursor.execute("""
                    INSERT INTO real_time_metrics (metric_name, metric_value, timestamp)
                    VALUES ('test_metric', 1.0, NOW())
                    RETURNING id
                """)
                
                result = cursor.fetchone()
                if result and result['id']:
                    print("  Database write operations: OK")
                else:
                    print("  Database write operations: FAILED")
                    return False
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            print(f"  PostgreSQL test failed: {e}")
            return False
    
    def test_dashboard_api(self) -> bool:
        """Test dashboard API endpoints."""
        try:
            base_url = self.dashboard_url
            
            # Test endpoints
            endpoints = [
                '/api/metrics/overview',
                '/api/alerts/recent?limit=5',
                '/api/performance/models',
                '/api/system/health'
            ]
            
            successful_endpoints = 0
            
            for endpoint in endpoints:
                try:
                    response = requests.get(f"{base_url}{endpoint}", timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        print(f"  {endpoint}: OK (status: {response.status_code})")
                        successful_endpoints += 1
                    else:
                        print(f"  {endpoint}: FAILED (status: {response.status_code})")
                        
                except requests.exceptions.RequestException as e:
                    print(f"  {endpoint}: ERROR ({str(e)})")
            
            success_rate = successful_endpoints / len(endpoints)
            print(f"  API endpoints success rate: {success_rate*100:.0f}%")
            
            return success_rate >= 0.75  # At least 75% of endpoints should work
            
        except Exception as e:
            print(f"  Dashboard API test failed: {e}")
            return False
    
    def test_end_to_end_flow(self) -> bool:
        """Test complete end-to-end transaction processing."""
        try:
            print("  Testing complete transaction processing flow...")
            
            # Generate a test transaction
            test_transaction = {
                'transaction_id': f'test_txn_{int(time.time())}',
                'user_id': 'test_user_001',
                'timestamp': datetime.now().isoformat(),
                'amount': 150.00,
                'merchant_category': 'online',
                'transaction_type': 'card_not_present',
                'is_weekend': False,
                'hour': 14,
                'day_of_week': 2,
                'month': datetime.now().month,
                'is_night': False,
                'latitude': 40.7128,
                'longitude': -74.0060,
                'city': 'New York',
                'country': 'US',
                'is_fraud': 0,
                'fraud_type': None
            }
            
            # Send transaction to Kafka
            producer = KafkaProducer(
                bootstrap_servers=self.kafka_servers,
                value_serializer=lambda x: json.dumps(x).encode('utf-8')
            )
            
            producer.send('fraud-transactions', value=test_transaction)
            producer.flush()
            producer.close()
            
            print(f"  Test transaction sent: {test_transaction['transaction_id']}")
            
            # Wait for processing
            time.sleep(5)
            
            # Check if transaction was processed (check Redis for any activity)
            r = redis.Redis(host=self.redis_host, decode_responses=True)
            prediction_count = int(r.get('fraud_detector:prediction_count') or 0)
            
            if prediction_count > 0:
                print(f"  Transaction processing: OK ({prediction_count} total predictions)")
                return True
            else:
                print("  Transaction processing: No evidence of processing")
                return False
                
        except Exception as e:
            print(f"  End-to-end test failed: {e}")
            return False
    
    def test_performance_benchmarks(self) -> bool:
        """Test system performance benchmarks."""
        try:
            print("  Running performance benchmarks...")
            
            # Get performance metrics from Redis
            r = redis.Redis(host=self.redis_host, decode_responses=True)
            
            avg_inference_time = float(r.get('fraud_detector:avg_inference_time') or 0.0)
            total_predictions = int(r.get('fraud_detector:prediction_count') or 0)
            
            print(f"  Average inference time: {avg_inference_time:.2f}ms")
            print(f"  Total predictions processed: {total_predictions}")
            
            # Performance targets
            max_inference_time_ms = 50.0  # Target <50ms (allowing for network overhead)
            min_predictions_for_test = 1    # At least 1 prediction to validate
            
            performance_ok = True
            
            if avg_inference_time > max_inference_time_ms:
                print(f"  WARNING: Inference time ({avg_inference_time:.2f}ms) > target ({max_inference_time_ms}ms)")
                performance_ok = False
            
            if total_predictions < min_predictions_for_test:
                print(f"  WARNING: Low prediction count ({total_predictions}) - system may not be processing")
                performance_ok = False
            
            # Test dashboard response time
            start_time = time.time()
            response = requests.get(f"{self.dashboard_url}/api/system/health", timeout=5)
            dashboard_response_time = (time.time() - start_time) * 1000
            
            print(f"  Dashboard response time: {dashboard_response_time:.2f}ms")
            
            if dashboard_response_time > 2000:  # 2 second max
                print(f"  WARNING: Dashboard response time too slow")
                performance_ok = False
            
            return performance_ok
            
        except Exception as e:
            print(f"  Performance benchmark failed: {e}")
            return False
    
    def test_health_monitoring(self) -> bool:
        """Test system health monitoring capabilities."""
        try:
            # Test dashboard health endpoint
            response = requests.get(f"{self.dashboard_url}/api/system/health", timeout=10)
            
            if response.status_code != 200:
                print(f"  Health endpoint failed: {response.status_code}")
                return False
                
            health_data = response.json()
            
            overall_status = health_data.get('overall_status', 'unknown')
            components = health_data.get('components', {})
            
            print(f"  Overall system status: {overall_status}")
            print(f"  Component health checks: {len(components)}")
            
            # Check individual components
            healthy_components = 0
            for component, status in components.items():
                component_status = status.get('status', 'unknown')
                print(f"    {component}: {component_status}")
                
                if component_status == 'healthy':
                    healthy_components += 1
            
            # At least 2/3 of components should be healthy
            health_threshold = len(components) * 0.67
            
            if healthy_components >= health_threshold:
                print(f"  Health monitoring: OK ({healthy_components}/{len(components)} healthy)")
                return True
            else:
                print(f"  Health monitoring: DEGRADED ({healthy_components}/{len(components)} healthy)")
                return False
                
        except Exception as e:
            print(f"  Health monitoring test failed: {e}")
            return False
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate final test report."""
        end_time = datetime.now()
        duration = end_time - self.test_results['start_time']
        
        self.test_results.update({
            'end_time': end_time,
            'duration_seconds': duration.total_seconds(),
            'success_rate': self.test_results['tests_passed'] / self.test_results['tests_run'] * 100
        })
        
        print("="*60)
        print("📊 TEST RESULTS SUMMARY")
        print("="*60)
        
        print(f"Tests Run: {self.test_results['tests_run']}")
        print(f"Tests Passed: {self.test_results['tests_passed']} ✅")
        print(f"Tests Failed: {self.test_results['tests_failed']} ❌")
        print(f"Success Rate: {self.test_results['success_rate']:.1f}%")
        print(f"Duration: {duration.total_seconds():.1f} seconds")
        
        if self.test_results['errors']:
            print(f"\n❌ ERRORS ENCOUNTERED:")
            for error in self.test_results['errors']:
                print(f"  - {error['test']}: {error['error']}")
        
        if self.test_results['success_rate'] >= 80:
            print("\n🎉 OVERALL RESULT: SYSTEM HEALTHY")
            print("The real-time fraud detection system is working correctly!")
        elif self.test_results['success_rate'] >= 60:
            print("\n⚠️  OVERALL RESULT: SYSTEM DEGRADED")
            print("Some components have issues but core functionality works.")
        else:
            print("\n❌ OVERALL RESULT: SYSTEM UNHEALTHY")
            print("Multiple critical issues detected. Check system configuration.")
        
        print(f"\nSystem Status: http://localhost:8080/api/system/health")
        print(f"Dashboard: http://localhost:8080")
        
        return self.test_results

def main():
    """Main test function."""
    print("🧪 Real-Time Fraud Detection System Test Suite")
    print("="*60)
    
    # Check if system is running
    try:
        result = subprocess.run(['docker-compose', 'ps'], 
                              capture_output=True, text=True, check=True)
        
        if 'Up' not in result.stdout:
            print("❌ System doesn't appear to be running!")
            print("Please start the system first:")
            print("  python start_realtime_system.py")
            sys.exit(1)
            
    except subprocess.CalledProcessError:
        print("❌ Could not check system status")
        print("Make sure Docker Compose is available and system is started")
        sys.exit(1)
    
    # Run tests
    tester = RealTimePipelineTest()
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    if results['success_rate'] >= 80:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure

if __name__ == "__main__":
    main()