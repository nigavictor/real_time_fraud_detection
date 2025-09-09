#!/usr/bin/env python3
"""
Simple Real-Time Fraud Detection Demo
====================================

This script demonstrates real-time fraud detection by:
1. Connecting to our running Kafka/Redis infrastructure
2. Generating sample transactions
3. Processing them through our trained ML models
4. Showing fraud detection results in real-time

This is a simplified version that runs without Docker containers.
"""

import sys
import os
sys.path.append('src')

import json
import time
import pandas as pd
import numpy as np
import joblib
import redis
from datetime import datetime
from kafka import KafkaProducer, KafkaConsumer
import threading
import queue

# Simple feature engineering (subset of full pipeline)
class SimpleFeatureEngineering:
    def __init__(self):
        self.user_cache = {}
    
    def engineer_features(self, transaction):
        """Create basic features for fraud detection."""
        features = {}
        
        # Basic features
        features['amount'] = float(transaction['amount'])
        features['amount_log'] = np.log1p(features['amount'])
        features['hour'] = int(transaction['hour'])
        features['day_of_week'] = int(transaction.get('day_of_week', 1))
        features['month'] = int(transaction.get('month', 1))
        features['is_weekend'] = bool(transaction.get('is_weekend', False))
        features['is_night'] = bool(transaction.get('is_night', False))
        features['is_business_hours'] = 9 <= features['hour'] <= 17
        
        # User behavior features (simplified)
        user_id = transaction['user_id']
        if user_id not in self.user_cache:
            self.user_cache[user_id] = {
                'avg_amount': features['amount'],
                'total_txns': 1,
                'max_amount': features['amount']
            }
        else:
            cache = self.user_cache[user_id]
            cache['total_txns'] += 1
            cache['avg_amount'] = (cache['avg_amount'] * (cache['total_txns'] - 1) + features['amount']) / cache['total_txns']
            cache['max_amount'] = max(cache['max_amount'], features['amount'])
        
        user_stats = self.user_cache[user_id]
        features['user_avg_amount'] = user_stats['avg_amount']
        features['user_std_amount'] = user_stats['avg_amount'] * 0.5  # Approximation
        features['user_total_transactions'] = float(user_stats['total_txns'])
        features['amount_deviation_from_user_avg'] = (features['amount'] - user_stats['avg_amount']) / (user_stats['avg_amount'] + 1)
        features['amount_vs_user_max'] = features['amount'] / (user_stats['max_amount'] + 1)
        
        # Velocity features (simplified)
        features['user_txn_count_1h'] = min(5.0, float(user_stats['total_txns']) / 24)
        features['user_txn_count_24h'] = float(user_stats['total_txns'])
        features['user_amount_sum_1h'] = features['amount'] * features['user_txn_count_1h']
        features['user_amount_sum_24h'] = features['amount'] * features['user_txn_count_24h']
        
        # Categorical features (simplified)
        category_map = {
            'grocery': 1, 'gas_station': 2, 'restaurant': 3, 'retail': 4, 'online': 5,
            'entertainment': 6, 'pharmacy': 7, 'transport': 8, 'utilities': 9, 'other': 10
        }
        features['merchant_category_encoded'] = category_map.get(transaction.get('merchant_category', 'other'), 10)
        features['transaction_type_encoded'] = 1 if transaction.get('transaction_type') == 'card_present' else 2
        features['country_encoded'] = 1  # Simplified
        features['city_encoded'] = 1  # Simplified
        features['user_top_category_encoded'] = features['merchant_category_encoded']
        features['user_common_txn_type_encoded'] = features['transaction_type_encoded']
        
        # Location features (simplified)
        features['distance_from_prev_km'] = 0.0
        features['time_since_prev_hours'] = 1.0
        features['travel_velocity_kmh'] = 0.0
        features['is_impossible_travel'] = False
        features['is_unusual_category'] = False
        features['is_unusual_txn_type'] = False
        
        return features

class SimpleFraudDetector:
    """Simplified fraud detector for demo."""
    
    def __init__(self):
        self.models = {}
        self.preprocessor = None
        self.feature_engineering = SimpleFeatureEngineering()
        self.load_models()
    
    def load_models(self):
        """Load trained models."""
        try:
            # Find latest model directory
            model_dirs = [d for d in os.listdir('data/models') if d.startswith('trained_models_')]
            if not model_dirs:
                raise Exception("No trained models found!")
            
            latest_model_dir = sorted(model_dirs)[-1]
            model_path = f"data/models/{latest_model_dir}"
            
            # Load models
            model_files = {
                'xgboost': 'xgboost_model.joblib',
                'random_forest': 'random_forest_model.joblib',
                'ensemble': 'ensemble_model.joblib'
            }
            
            for model_name, filename in model_files.items():
                full_path = f"{model_path}/{filename}"
                if os.path.exists(full_path):
                    self.models[model_name] = joblib.load(full_path)
                    print(f"✓ Loaded {model_name} model")
            
            # Load preprocessor
            preprocessor_files = [f for f in os.listdir('data/models') if f.startswith('preprocessor_')]
            if preprocessor_files:
                latest_preprocessor = sorted(preprocessor_files)[-1]
                self.preprocessor = joblib.load(f"data/models/{latest_preprocessor}")
                print(f"✓ Loaded preprocessor")
            
            print(f"✅ Loaded {len(self.models)} models for fraud detection")
            
        except Exception as e:
            print(f"❌ Failed to load models: {e}")
            raise
    
    def predict_fraud(self, transaction):
        """Predict if transaction is fraudulent."""
        start_time = time.time()
        
        try:
            # Feature engineering
            features = self.feature_engineering.engineer_features(transaction)
            
            # Convert to DataFrame
            feature_df = pd.DataFrame([features])
            
            # Prepare features (ensure all 29 required features)
            required_features = [
                'amount', 'amount_log', 'hour', 'day_of_week', 'month', 'is_weekend',
                'is_night', 'is_business_hours', 'user_avg_amount', 'user_std_amount',
                'user_total_transactions', 'amount_deviation_from_user_avg', 'amount_vs_user_max',
                'user_txn_count_1h', 'user_txn_count_24h', 'user_amount_sum_1h', 
                'user_amount_sum_24h', 'is_unusual_category', 'is_unusual_txn_type',
                'distance_from_prev_km', 'time_since_prev_hours', 'travel_velocity_kmh',
                'is_impossible_travel', 'merchant_category_encoded', 'country_encoded',
                'city_encoded', 'transaction_type_encoded', 'user_top_category_encoded',
                'user_common_txn_type_encoded'
            ]
            
            # Add missing features
            for feature in required_features:
                if feature not in feature_df.columns:
                    feature_df[feature] = 0
            
            # Select and order features
            feature_df = feature_df[required_features]
            
            # Apply preprocessing if available
            if self.preprocessor:
                try:
                    processed_features = self.preprocessor.transform(feature_df)
                except:
                    processed_features = feature_df.values
            else:
                processed_features = feature_df.values
            
            # Run predictions
            predictions = {}
            
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(processed_features)[0]
                        fraud_prob = prob[1] if len(prob) > 1 else prob[0]
                    else:
                        fraud_prob = model.predict(processed_features)[0]
                    predictions[model_name] = float(fraud_prob)
                except Exception as e:
                    predictions[model_name] = 0.0
            
            # Ensemble prediction (weighted average)
            model_weights = {'xgboost': 0.4, 'ensemble': 0.3, 'random_forest': 0.3}
            
            weighted_prob = sum(
                predictions.get(model, 0.0) * model_weights.get(model, 0.1)
                for model in predictions.keys()
            )
            total_weight = sum(model_weights.get(model, 0.1) for model in predictions.keys())
            final_probability = weighted_prob / total_weight if total_weight > 0 else 0.0
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                'transaction_id': transaction.get('transaction_id'),
                'user_id': transaction.get('user_id'),
                'amount': transaction.get('amount'),
                'fraud_probability': final_probability,
                'is_fraud_alert': final_probability > 0.5,
                'confidence': 'high' if final_probability > 0.8 else ('medium' if final_probability > 0.5 else 'low'),
                'model_predictions': predictions,
                'processing_time_ms': processing_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'transaction_id': transaction.get('transaction_id'),
                'error': str(e),
                'fraud_probability': 0.0,
                'processing_time_ms': (time.time() - start_time) * 1000
            }

def generate_sample_transaction():
    """Generate a sample transaction for testing."""
    import random
    
    user_id = f"demo_user_{random.randint(1, 100):03d}"
    
    # Generate realistic transaction
    transaction = {
        'transaction_id': f"demo_txn_{int(time.time() * 1000)}",
        'user_id': user_id,
        'timestamp': datetime.now().isoformat(),
        'amount': round(np.random.lognormal(4.0, 1.2), 2),  # Log-normal distribution
        'merchant_category': random.choice(['grocery', 'gas_station', 'restaurant', 'retail', 'online']),
        'transaction_type': random.choice(['card_present', 'online', 'contactless']),
        'hour': datetime.now().hour,
        'day_of_week': datetime.now().weekday(),
        'month': datetime.now().month,
        'is_weekend': datetime.now().weekday() >= 5,
        'is_night': datetime.now().hour < 6 or datetime.now().hour > 22,
        'city': 'Demo City',
        'country': 'Demo Country'
    }
    
    # Occasionally generate suspicious transactions
    if random.random() < 0.15:  # 15% suspicious
        transaction['amount'] = random.uniform(500, 2000)  # High amount
        transaction['merchant_category'] = 'online'
        transaction['hour'] = random.choice([2, 3, 4, 23, 0, 1])  # Unusual hours
    
    return transaction

def demo_fraud_detection():
    """Run the fraud detection demo."""
    print("🛡️ REAL-TIME FRAUD DETECTION DEMO")
    print("="*50)
    
    # Initialize fraud detector
    print("🤖 Loading ML models...")
    detector = SimpleFraudDetector()
    
    # Test Redis connection
    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        print("✓ Connected to Redis cache")
    except Exception as e:
        print(f"⚠️  Redis not available: {e}")
        r = None
    
    # Test Kafka connection
    try:
        producer = KafkaProducer(
            bootstrap_servers='localhost:9092',
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        print("✓ Connected to Kafka stream")
    except Exception as e:
        print(f"⚠️  Kafka not available: {e}")
        producer = None
    
    print("\n🚀 Starting real-time fraud detection demo...")
    print("Press Ctrl+C to stop\n")
    
    transaction_count = 0
    fraud_alerts = 0
    total_processing_time = 0
    
    try:
        while True:
            # Generate sample transaction
            transaction = generate_sample_transaction()
            
            # Process through fraud detection
            result = detector.predict_fraud(transaction)
            
            # Update statistics
            transaction_count += 1
            total_processing_time += result.get('processing_time_ms', 0)
            
            if result.get('is_fraud_alert', False):
                fraud_alerts += 1
                print(f"🚨 FRAUD ALERT #{fraud_alerts}")
            else:
                print(f"✅ Transaction #{transaction_count}")
            
            print(f"   User: {result['user_id']}")
            print(f"   Amount: ${result['amount']}")
            print(f"   Fraud Probability: {result['fraud_probability']:.3f}")
            print(f"   Confidence: {result['confidence']}")
            print(f"   Processing Time: {result['processing_time_ms']:.1f}ms")
            
            if result.get('model_predictions'):
                print(f"   Model Scores:")
                for model, score in result['model_predictions'].items():
                    print(f"     - {model}: {score:.3f}")
            
            # Send to Kafka if available
            if producer and result.get('is_fraud_alert'):
                try:
                    producer.send('fraud-alerts', value=result)
                    print(f"   📡 Alert sent to Kafka")
                except Exception as e:
                    pass
            
            # Cache user data in Redis if available
            if r:
                try:
                    r.hset(f"demo_user:{result['user_id']}", mapping={
                        'last_amount': result['amount'],
                        'last_transaction': result['timestamp'],
                        'total_processed': transaction_count
                    })
                except Exception as e:
                    pass
            
            print()
            
            # Statistics every 10 transactions
            if transaction_count % 10 == 0:
                avg_processing_time = total_processing_time / transaction_count
                fraud_rate = (fraud_alerts / transaction_count) * 100
                
                print(f"📊 STATISTICS (Last {transaction_count} transactions)")
                print(f"   Fraud Detection Rate: {fraud_rate:.1f}%")
                print(f"   Average Processing Time: {avg_processing_time:.1f}ms")
                print(f"   Fraud Alerts Generated: {fraud_alerts}")
                print("="*50)
                print()
            
            # Wait before next transaction
            time.sleep(2)  # 2 seconds between transactions for demo
    
    except KeyboardInterrupt:
        print(f"\n🛑 Demo stopped by user")
        
        # Final statistics
        if transaction_count > 0:
            avg_processing_time = total_processing_time / transaction_count
            fraud_rate = (fraud_alerts / transaction_count) * 100
            
            print(f"\n📊 FINAL STATISTICS")
            print(f"   Total Transactions: {transaction_count}")
            print(f"   Fraud Alerts: {fraud_alerts} ({fraud_rate:.1f}%)")
            print(f"   Average Processing Time: {avg_processing_time:.1f}ms")
            print(f"   Models Used: {len(detector.models)}")
            print(f"   System Performance: {'✅ Real-time ready' if avg_processing_time < 50 else '⚠️  Needs optimization'}")
    
    finally:
        if producer:
            producer.close()

if __name__ == "__main__":
    demo_fraud_detection()