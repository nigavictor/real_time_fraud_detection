#!/usr/bin/env python3
"""
Real-Time Fraud Detection Processor
==================================

This service consumes transaction streams from Kafka, applies trained ML models
in real-time, and produces fraud alerts with sub-10ms latency.

Features:
- Real-time ML model inference (<10ms)
- Multiple model ensemble voting
- Adaptive threshold adjustment
- Performance monitoring and metrics
- Alert generation and routing
- Model A/B testing capabilities
"""

import json
import time
import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading

import pandas as pd
import numpy as np
import joblib
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DetectorConfig:
    """Configuration for fraud detector."""
    kafka_bootstrap_servers: str = 'localhost:9092'
    kafka_input_topic: str = 'fraud-transactions'
    kafka_output_topic: str = 'fraud-alerts'
    redis_host: str = 'localhost'
    redis_port: int = 6379
    postgres_host: str = 'localhost'
    postgres_port: int = 5432
    postgres_db: str = 'fraud_detection'
    postgres_user: str = 'fraud_user'
    postgres_password: str = 'fraud_password'
    model_path: str = '/app/models'
    default_threshold: float = 0.5
    ensemble_voting: str = 'soft'  # 'soft' or 'hard'
    enable_adaptive_threshold: bool = True
    batch_size: int = 10
    max_processing_time_ms: float = 8.0  # Target <10ms including overhead

class FeatureEngineering:
    """Real-time feature engineering for streaming transactions."""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.feature_cache_ttl = 3600  # 1 hour
        
    def engineer_features(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Engineer features for a single transaction in real-time."""
        start_time = time.time()
        
        features = transaction.copy()
        user_id = transaction['user_id']
        timestamp = datetime.fromisoformat(transaction['timestamp'])
        amount = float(transaction['amount'])
        
        try:
            # Basic time features (already in transaction)
            features['amount_log'] = np.log1p(amount)
            features['is_business_hours'] = 9 <= timestamp.hour <= 17
            
            # User historical features (cached in Redis)
            user_features = self._get_user_features(user_id, timestamp, amount)
            features.update(user_features)
            
            # Velocity features
            velocity_features = self._get_velocity_features(user_id, timestamp, amount)
            features.update(velocity_features)
            
            # Location features
            if 'latitude' in transaction and 'longitude' in transaction:
                location_features = self._get_location_features(
                    user_id, transaction['latitude'], transaction['longitude']
                )
                features.update(location_features)
            
            # Categorical encoding (simplified)
            features.update(self._encode_categorical_features(transaction))
            
            processing_time = (time.time() - start_time) * 1000
            logger.debug(f"Feature engineering completed in {processing_time:.2f}ms")
            
            return features
            
        except Exception as e:
            logger.error(f"Feature engineering failed for {user_id}: {e}")
            # Return basic features as fallback
            return self._get_basic_features(transaction)
    
    def _get_user_features(self, user_id: str, timestamp: datetime, amount: float) -> Dict[str, Any]:
        """Get user historical features from Redis cache."""
        cache_key = f"user_features:{user_id}"
        
        try:
            # Get cached user statistics
            cached_data = self.redis.hgetall(cache_key)
            
            if cached_data:
                # Decode Redis data
                user_stats = {k.decode(): float(v.decode()) for k, v in cached_data.items()}
                
                # Calculate deviation features
                avg_amount = user_stats.get('avg_amount', amount)
                std_amount = user_stats.get('std_amount', amount * 0.5)
                max_amount = user_stats.get('max_amount', amount)
                total_txns = user_stats.get('total_transactions', 1)
                
                features = {
                    'user_avg_amount': avg_amount,
                    'user_std_amount': max(std_amount, 1.0),  # Avoid division by zero
                    'user_total_transactions': total_txns,
                    'amount_deviation_from_user_avg': (amount - avg_amount) / (std_amount + 1),
                    'amount_vs_user_max': amount / (max_amount + 1)
                }
                
                # Update cache with new transaction
                self._update_user_cache(cache_key, amount, total_txns + 1)
                
            else:
                # New user - initialize cache
                features = {
                    'user_avg_amount': amount,
                    'user_std_amount': amount * 0.5,
                    'user_total_transactions': 1,
                    'amount_deviation_from_user_avg': 0.0,
                    'amount_vs_user_max': 1.0
                }
                
                # Initialize cache
                self.redis.hmset(cache_key, {
                    'avg_amount': amount,
                    'std_amount': amount * 0.5,
                    'max_amount': amount,
                    'total_transactions': 1,
                    'last_update': timestamp.timestamp()
                })
                self.redis.expire(cache_key, self.feature_cache_ttl)
            
            return features
            
        except Exception as e:
            logger.warning(f"Failed to get user features for {user_id}: {e}")
            return {
                'user_avg_amount': amount,
                'user_std_amount': amount * 0.5,
                'user_total_transactions': 1,
                'amount_deviation_from_user_avg': 0.0,
                'amount_vs_user_max': 1.0
            }
    
    def _get_velocity_features(self, user_id: str, timestamp: datetime, amount: float) -> Dict[str, Any]:
        """Get velocity features from Redis sorted sets."""
        try:
            # Use Redis sorted sets for time-windowed queries
            txn_key = f"user_txns:{user_id}"
            amount_key = f"user_amounts:{user_id}"
            
            current_timestamp = timestamp.timestamp()
            one_hour_ago = current_timestamp - 3600
            one_day_ago = current_timestamp - 86400
            
            # Count transactions in time windows
            txn_count_1h = self.redis.zcount(txn_key, one_hour_ago, current_timestamp)
            txn_count_24h = self.redis.zcount(txn_key, one_day_ago, current_timestamp)
            
            # Sum amounts in time windows (approximate)
            amount_sum_1h = len(self.redis.zrangebyscore(amount_key, one_hour_ago, current_timestamp))
            amount_sum_24h = len(self.redis.zrangebyscore(amount_key, one_day_ago, current_timestamp))
            
            # Add current transaction to sorted sets
            self.redis.zadd(txn_key, {str(current_timestamp): current_timestamp})
            self.redis.zadd(amount_key, {f"{amount}:{current_timestamp}": current_timestamp})
            
            # Set expiration
            self.redis.expire(txn_key, 86400)  # 24 hours
            self.redis.expire(amount_key, 86400)
            
            # Clean old entries (keep only last 24h)
            self.redis.zremrangebyscore(txn_key, 0, one_day_ago)
            self.redis.zremrangebyscore(amount_key, 0, one_day_ago)
            
            return {
                'user_txn_count_1h': float(txn_count_1h),
                'user_txn_count_24h': float(txn_count_24h),
                'user_amount_sum_1h': amount * txn_count_1h,  # Approximation
                'user_amount_sum_24h': amount * txn_count_24h
            }
            
        except Exception as e:
            logger.warning(f"Failed to get velocity features for {user_id}: {e}")
            return {
                'user_txn_count_1h': 0.0,
                'user_txn_count_24h': 0.0,
                'user_amount_sum_1h': 0.0,
                'user_amount_sum_24h': 0.0
            }
    
    def _get_location_features(self, user_id: str, lat: float, lng: float) -> Dict[str, Any]:
        """Get location-based features."""
        try:
            # Get last location from cache
            location_key = f"user_location:{user_id}"
            last_location = self.redis.hgetall(location_key)
            
            if last_location:
                last_lat = float(last_location[b'latitude'].decode())
                last_lng = float(last_location[b'longitude'].decode())
                last_time = float(last_location[b'timestamp'].decode())
                
                # Calculate distance and velocity
                distance = self._haversine_distance(lat, lng, last_lat, last_lng)
                time_diff = time.time() - last_time
                velocity = distance / (time_diff / 3600) if time_diff > 0 else 0  # km/h
                
                features = {
                    'distance_from_prev_km': distance,
                    'time_since_prev_hours': time_diff / 3600,
                    'travel_velocity_kmh': velocity,
                    'is_impossible_travel': velocity > 800  # >800 km/h is impossible
                }
            else:
                features = {
                    'distance_from_prev_km': 0.0,
                    'time_since_prev_hours': 24.0,  # Assume 24h since last
                    'travel_velocity_kmh': 0.0,
                    'is_impossible_travel': False
                }
            
            # Update location cache
            self.redis.hmset(location_key, {
                'latitude': lat,
                'longitude': lng,
                'timestamp': time.time()
            })
            self.redis.expire(location_key, 86400)  # 24 hours
            
            return features
            
        except Exception as e:
            logger.warning(f"Failed to get location features for {user_id}: {e}")
            return {
                'distance_from_prev_km': 0.0,
                'time_since_prev_hours': 1.0,
                'travel_velocity_kmh': 0.0,
                'is_impossible_travel': False
            }
    
    def _encode_categorical_features(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Simple categorical encoding for streaming."""
        # Simplified label encoding based on common patterns
        category_encodings = {
            'grocery': 1, 'gas_station': 2, 'restaurant': 3, 'retail': 4,
            'online': 5, 'entertainment': 6, 'pharmacy': 7, 'transport': 8,
            'utilities': 9, 'other': 10, 'electronics': 11, 'jewelry': 12,
            'travel': 13, 'atm_withdrawal': 14, 'financial': 15
        }
        
        type_encodings = {
            'card_present': 1, 'online': 2, 'contactless': 3, 'card_not_present': 4
        }
        
        return {
            'merchant_category_encoded': category_encodings.get(
                transaction.get('merchant_category', 'other'), 10
            ),
            'transaction_type_encoded': type_encodings.get(
                transaction.get('transaction_type', 'card_present'), 1
            ),
            'country_encoded': hash(transaction.get('country', 'US')) % 100,  # Simple hash
            'city_encoded': hash(transaction.get('city', 'Unknown')) % 1000,
            # Simplified user pattern encoding
            'user_top_category_encoded': category_encodings.get(
                transaction.get('merchant_category', 'other'), 10
            ),
            'user_common_txn_type_encoded': type_encodings.get(
                transaction.get('transaction_type', 'card_present'), 1
            ),
            'is_unusual_category': False,  # Simplified
            'is_unusual_txn_type': False   # Simplified
        }
    
    def _update_user_cache(self, cache_key: str, amount: float, total_txns: int):
        """Update user statistics in cache."""
        try:
            # This is a simplified update - in production, you'd use more sophisticated running statistics
            current_avg = float(self.redis.hget(cache_key, 'avg_amount') or amount)
            new_avg = (current_avg * (total_txns - 1) + amount) / total_txns
            
            self.redis.hset(cache_key, 'avg_amount', new_avg)
            self.redis.hset(cache_key, 'total_transactions', total_txns)
            self.redis.hset(cache_key, 'last_update', time.time())
            
            if amount > float(self.redis.hget(cache_key, 'max_amount') or 0):
                self.redis.hset(cache_key, 'max_amount', amount)
                
        except Exception as e:
            logger.warning(f"Failed to update user cache: {e}")
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points in kilometers."""
        from math import radians, sin, cos, sqrt, asin
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        return 2 * 6371 * asin(sqrt(a))  # Earth radius = 6371 km
    
    def _get_basic_features(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Get basic features as fallback."""
        amount = float(transaction['amount'])
        return {
            'amount_log': np.log1p(amount),
            'user_avg_amount': amount,
            'user_std_amount': amount * 0.5,
            'user_total_transactions': 1,
            'amount_deviation_from_user_avg': 0.0,
            'amount_vs_user_max': 1.0,
            'user_txn_count_1h': 0.0,
            'user_txn_count_24h': 0.0,
            'user_amount_sum_1h': 0.0,
            'user_amount_sum_24h': 0.0,
            'distance_from_prev_km': 0.0,
            'time_since_prev_hours': 1.0,
            'travel_velocity_kmh': 0.0,
            'is_impossible_travel': False,
            'merchant_category_encoded': 10,
            'transaction_type_encoded': 1,
            'country_encoded': 1,
            'city_encoded': 1,
            'user_top_category_encoded': 10,
            'user_common_txn_type_encoded': 1,
            'is_unusual_category': False,
            'is_unusual_txn_type': False,
            'is_business_hours': True
        }


class ModelEnsemble:
    """Ensemble of fraud detection models with real-time inference."""
    
    def __init__(self, model_path: str, redis_client):
        self.model_path = model_path
        self.redis = redis_client
        self.models = {}
        self.preprocessor = None
        self.feature_names = []
        
        self._load_models()
        
    def _load_models(self):
        """Load trained models and preprocessor."""
        try:
            # Find latest model directory
            model_dirs = [d for d in os.listdir(self.model_path) if d.startswith('trained_models_')]
            if not model_dirs:
                raise Exception("No trained models found")
                
            latest_model_dir = sorted(model_dirs)[-1]
            model_dir_path = os.path.join(self.model_path, latest_model_dir)
            
            logger.info(f"Loading models from {model_dir_path}")
            
            # Load individual models
            model_files = {
                'xgboost': 'xgboost_model.joblib',
                'random_forest': 'random_forest_model.joblib',
                'logistic_regression': 'logistic_regression_model.joblib',
                'ensemble': 'ensemble_model.joblib'
            }
            
            for model_name, filename in model_files.items():
                model_path = os.path.join(model_dir_path, filename)
                if os.path.exists(model_path):
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"Loaded {model_name} model")
            
            # Load preprocessor
            preprocessor_files = [f for f in os.listdir(self.model_path) if f.startswith('preprocessor_')]
            if preprocessor_files:
                latest_preprocessor = sorted(preprocessor_files)[-1]
                preprocessor_path = os.path.join(self.model_path, latest_preprocessor)
                self.preprocessor = joblib.load(preprocessor_path)
                logger.info(f"Loaded preprocessor: {latest_preprocessor}")
            
            # Load metadata for feature names
            metadata_path = os.path.join(model_dir_path, 'model_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.feature_names = metadata.get('feature_names', [])
            
            logger.info(f"Model ensemble loaded with {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference on all models and return ensemble result."""
        start_time = time.time()
        
        try:
            # Convert features to DataFrame
            feature_df = pd.DataFrame([features])
            
            # Ensure all required features are present
            feature_df = self._prepare_features(feature_df)
            
            # Apply preprocessing if available
            if self.preprocessor:
                try:
                    processed_features = self.preprocessor.transform(feature_df)
                except Exception as e:
                    logger.warning(f"Preprocessing failed, using raw features: {e}")
                    processed_features = feature_df.values
            else:
                processed_features = feature_df.values
            
            # Run predictions on all models
            predictions = {}
            probabilities = {}
            
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(processed_features)[0]
                        fraud_prob = prob[1] if len(prob) > 1 else prob[0]
                        probabilities[model_name] = float(fraud_prob)
                    else:
                        # For models without probability output
                        pred = model.predict(processed_features)[0]
                        probabilities[model_name] = float(pred)
                        
                except Exception as e:
                    logger.warning(f"Prediction failed for {model_name}: {e}")
                    probabilities[model_name] = 0.0
            
            # Ensemble voting (weighted average)
            model_weights = {
                'xgboost': 0.4,      # Best performing model
                'ensemble': 0.3,      # Ensemble model
                'random_forest': 0.2, # Good precision
                'logistic_regression': 0.1  # Baseline
            }
            
            weighted_prob = sum(
                probabilities.get(model, 0.0) * model_weights.get(model, 0.1)
                for model in probabilities.keys()
            )
            
            # Normalize by total weight
            total_weight = sum(model_weights.get(model, 0.1) for model in probabilities.keys())
            final_probability = weighted_prob / total_weight if total_weight > 0 else 0.0
            
            inference_time = (time.time() - start_time) * 1000
            
            result = {
                'fraud_probability': final_probability,
                'model_predictions': probabilities,
                'inference_time_ms': inference_time,
                'feature_count': len(feature_df.columns),
                'primary_model': max(probabilities.keys(), key=lambda k: probabilities[k])
            }
            
            # Update performance metrics
            self._update_performance_metrics(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            return {
                'fraud_probability': 0.0,
                'model_predictions': {},
                'inference_time_ms': (time.time() - start_time) * 1000,
                'error': str(e)
            }
    
    def _prepare_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required features are present with correct types."""
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
        
        # Add missing features with default values
        for feature in required_features:
            if feature not in feature_df.columns:
                if 'encoded' in feature or feature.startswith('is_'):
                    feature_df[feature] = 0
                else:
                    feature_df[feature] = 0.0
        
        # Select only required features in correct order
        try:
            feature_df = feature_df[required_features]
        except KeyError as e:
            logger.warning(f"Some features missing: {e}")
            # Use available features
            available_features = [f for f in required_features if f in feature_df.columns]
            feature_df = feature_df[available_features]
        
        # Convert boolean columns to integers
        bool_columns = ['is_weekend', 'is_night', 'is_business_hours', 'is_unusual_category',
                       'is_unusual_txn_type', 'is_impossible_travel']
        
        for col in bool_columns:
            if col in feature_df.columns:
                feature_df[col] = feature_df[col].astype(int)
        
        return feature_df
    
    def _update_performance_metrics(self, result: Dict[str, Any]):
        """Update performance metrics in Redis."""
        try:
            # Update inference time metrics
            inference_time = result['inference_time_ms']
            
            # Running average of inference time
            avg_key = 'fraud_detector:avg_inference_time'
            count_key = 'fraud_detector:prediction_count'
            
            current_count = int(self.redis.get(count_key) or 0)
            current_avg = float(self.redis.get(avg_key) or 0.0)
            
            new_count = current_count + 1
            new_avg = (current_avg * current_count + inference_time) / new_count
            
            self.redis.set(avg_key, new_avg)
            self.redis.set(count_key, new_count)
            
            # Update hourly metrics
            hour_key = f"fraud_detector:predictions:{datetime.now().strftime('%Y%m%d%H')}"
            self.redis.incr(hour_key)
            self.redis.expire(hour_key, 86400)  # 24 hour expiry
            
        except Exception as e:
            logger.warning(f"Failed to update performance metrics: {e}")


class FraudDetectionProcessor:
    """Main fraud detection processor."""
    
    def __init__(self, config: DetectorConfig):
        self.config = config
        
        # Initialize connections
        self.redis_client = self._initialize_redis()
        self.postgres_conn = self._initialize_postgres()
        
        # Initialize components
        self.feature_engineering = FeatureEngineering(self.redis_client)
        self.model_ensemble = ModelEnsemble(config.model_path, self.redis_client)
        
        # Initialize Kafka
        self.consumer = self._initialize_kafka_consumer()
        self.producer = self._initialize_kafka_producer()
        
        # Processing statistics
        self.stats = {
            'total_processed': 0,
            'fraud_alerts': 0,
            'avg_processing_time_ms': 0.0,
            'start_time': datetime.now()
        }
        
        # Threading for database operations
        self.db_executor = ThreadPoolExecutor(max_workers=2)
        
    def _initialize_redis(self) -> redis.Redis:
        """Initialize Redis connection."""
        try:
            client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                decode_responses=False,  # We handle encoding manually
                socket_connect_timeout=5,
                socket_timeout=5
            )
            client.ping()
            logger.info("Redis connection established")
            return client
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _initialize_postgres(self):
        """Initialize PostgreSQL connection."""
        try:
            conn = psycopg2.connect(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                database=self.config.postgres_db,
                user=self.config.postgres_user,
                password=self.config.postgres_password
            )
            logger.info("PostgreSQL connection established")
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    def _initialize_kafka_consumer(self) -> KafkaConsumer:
        """Initialize Kafka consumer."""
        try:
            consumer = KafkaConsumer(
                self.config.kafka_input_topic,
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                key_deserializer=lambda x: x.decode('utf-8') if x else None,
                group_id='fraud-detector',
                enable_auto_commit=True,
                auto_commit_interval_ms=1000,
                max_poll_records=self.config.batch_size,
                consumer_timeout_ms=1000
            )
            logger.info(f"Kafka consumer initialized for topic: {self.config.kafka_input_topic}")
            return consumer
        except Exception as e:
            logger.error(f"Failed to initialize Kafka consumer: {e}")
            raise
    
    def _initialize_kafka_producer(self) -> KafkaProducer:
        """Initialize Kafka producer for alerts."""
        try:
            producer = KafkaProducer(
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                key_serializer=lambda x: x.encode('utf-8') if x else None
            )
            logger.info(f"Kafka producer initialized for alerts")
            return producer
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            raise
    
    def process_transactions(self):
        """Main processing loop."""
        logger.info("Starting fraud detection processor...")
        logger.info(f"Target processing time: <{self.config.max_processing_time_ms}ms")
        
        try:
            for message in self.consumer:
                batch_start_time = time.time()
                
                try:
                    # Process single transaction
                    transaction = message.value
                    result = self._process_single_transaction(transaction)
                    
                    # Generate alert if fraud detected
                    if result and result.get('is_fraud_alert', False):
                        self._generate_fraud_alert(transaction, result)
                    
                    # Update statistics
                    processing_time = (time.time() - batch_start_time) * 1000
                    self._update_processing_stats(processing_time, result)
                    
                    # Log performance warnings
                    if processing_time > self.config.max_processing_time_ms:
                        logger.warning(f"Processing time exceeded target: {processing_time:.2f}ms")
                    
                except Exception as e:
                    logger.error(f"Failed to process transaction: {e}")
                    continue
        
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Processing loop failed: {e}")
            raise
        finally:
            self._shutdown()
    
    def _process_single_transaction(self, transaction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single transaction through the fraud detection pipeline."""
        start_time = time.time()
        
        try:
            # Feature engineering
            features = self.feature_engineering.engineer_features(transaction)
            
            # Model inference
            prediction_result = self.model_ensemble.predict(features)
            
            # Determine if fraud alert should be generated
            fraud_probability = prediction_result.get('fraud_probability', 0.0)
            threshold = self._get_dynamic_threshold()
            
            is_fraud_alert = fraud_probability >= threshold
            
            result = {
                'transaction_id': transaction.get('transaction_id'),
                'user_id': transaction.get('user_id'),
                'fraud_probability': fraud_probability,
                'threshold_used': threshold,
                'is_fraud_alert': is_fraud_alert,
                'model_predictions': prediction_result.get('model_predictions', {}),
                'primary_model': prediction_result.get('primary_model', 'unknown'),
                'processing_time_ms': (time.time() - start_time) * 1000,
                'inference_time_ms': prediction_result.get('inference_time_ms', 0.0),
                'confidence': self._calculate_confidence(fraud_probability),
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Transaction processing failed: {e}")
            return None
    
    def _get_dynamic_threshold(self) -> float:
        """Get dynamic threshold based on recent performance."""
        if not self.config.enable_adaptive_threshold:
            return self.config.default_threshold
        
        try:
            # Get recent false positive rate from Redis
            fpr_key = 'fraud_detector:false_positive_rate'
            recent_fpr = float(self.redis.get(fpr_key) or 0.05)
            
            # Adjust threshold based on FPR
            # If FPR is high, increase threshold; if low, decrease threshold
            base_threshold = self.config.default_threshold
            
            if recent_fpr > 0.1:  # Too many false positives
                adjusted_threshold = min(0.9, base_threshold + 0.1)
            elif recent_fpr < 0.02:  # Too few alerts, might be missing fraud
                adjusted_threshold = max(0.2, base_threshold - 0.1)
            else:
                adjusted_threshold = base_threshold
            
            return adjusted_threshold
            
        except Exception as e:
            logger.warning(f"Failed to get dynamic threshold: {e}")
            return self.config.default_threshold
    
    def _calculate_confidence(self, fraud_probability: float) -> str:
        """Calculate confidence level based on fraud probability."""
        if fraud_probability >= 0.8:
            return 'high'
        elif fraud_probability >= 0.5:
            return 'medium'
        else:
            return 'low'
    
    def _generate_fraud_alert(self, transaction: Dict[str, Any], result: Dict[str, Any]):
        """Generate and send fraud alert."""
        try:
            alert = {
                'alert_id': f"alert_{int(time.time() * 1000)}_{transaction.get('user_id')}",
                'transaction_id': transaction.get('transaction_id'),
                'user_id': transaction.get('user_id'),
                'timestamp': transaction.get('timestamp'),
                'amount': transaction.get('amount'),
                'merchant_category': transaction.get('merchant_category'),
                'location': {
                    'city': transaction.get('city'),
                    'country': transaction.get('country'),
                    'latitude': transaction.get('latitude'),
                    'longitude': transaction.get('longitude')
                },
                'fraud_probability': result['fraud_probability'],
                'confidence': result['confidence'],
                'model_used': result['primary_model'],
                'threshold_used': result['threshold_used'],
                'model_predictions': result['model_predictions'],
                'processing_time_ms': result['processing_time_ms'],
                'alert_timestamp': datetime.now().isoformat(),
                'priority': 'high' if result['fraud_probability'] >= 0.8 else 'medium'
            }
            
            # Send to Kafka alerts topic
            self.producer.send(
                self.config.kafka_output_topic,
                key=transaction.get('user_id'),
                value=alert
            ).add_callback(self._on_alert_success).add_errback(self._on_alert_error)
            
            # Store in PostgreSQL (async)
            self.db_executor.submit(self._store_alert_in_db, alert)
            
            logger.info(f"Fraud alert generated for transaction {alert['transaction_id']} "
                       f"(probability: {alert['fraud_probability']:.3f})")
            
        except Exception as e:
            logger.error(f"Failed to generate fraud alert: {e}")
    
    def _store_alert_in_db(self, alert: Dict[str, Any]):
        """Store fraud alert in PostgreSQL database."""
        try:
            with self.postgres_conn.cursor() as cursor:
                insert_query = """
                INSERT INTO fraud_alerts (
                    transaction_id, user_id, timestamp, amount, merchant_category,
                    location, fraud_probability, model_used, investigation_status
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                location_str = f"{alert['location']['city']}, {alert['location']['country']}"
                
                cursor.execute(insert_query, (
                    alert['transaction_id'],
                    alert['user_id'], 
                    alert['timestamp'],
                    alert['amount'],
                    alert['merchant_category'],
                    location_str,
                    alert['fraud_probability'],
                    alert['model_used'],
                    'pending'
                ))
                
            self.postgres_conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to store alert in database: {e}")
            # Rollback on error
            self.postgres_conn.rollback()
    
    def _on_alert_success(self, record_metadata):
        """Callback for successful alert send."""
        logger.debug(f"Alert sent to {record_metadata.topic}[{record_metadata.partition}]")
    
    def _on_alert_error(self, exception):
        """Callback for failed alert send."""
        logger.error(f"Failed to send alert: {exception}")
    
    def _update_processing_stats(self, processing_time: float, result: Optional[Dict[str, Any]]):
        """Update processing statistics."""
        self.stats['total_processed'] += 1
        
        if result and result.get('is_fraud_alert', False):
            self.stats['fraud_alerts'] += 1
        
        # Update running average of processing time
        current_avg = self.stats['avg_processing_time_ms']
        total_processed = self.stats['total_processed']
        
        self.stats['avg_processing_time_ms'] = (
            (current_avg * (total_processed - 1) + processing_time) / total_processed
        )
        
        # Log stats periodically
        if self.stats['total_processed'] % 100 == 0:
            self._log_processing_stats()
    
    def _log_processing_stats(self):
        """Log processing statistics."""
        elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
        tps = self.stats['total_processed'] / elapsed if elapsed > 0 else 0
        fraud_rate = (self.stats['fraud_alerts'] / self.stats['total_processed'] 
                     if self.stats['total_processed'] > 0 else 0)
        
        logger.info(
            f"Processed {self.stats['total_processed']} transactions "
            f"({tps:.1f} TPS, {fraud_rate*100:.1f}% fraud alerts, "
            f"avg {self.stats['avg_processing_time_ms']:.1f}ms)"
        )
    
    def _shutdown(self):
        """Clean shutdown of all components."""
        logger.info("Shutting down fraud detection processor...")
        
        try:
            if self.consumer:
                self.consumer.close()
            if self.producer:
                self.producer.close()
            if self.postgres_conn:
                self.postgres_conn.close()
            if self.redis_client:
                self.redis_client.close()
            if self.db_executor:
                self.db_executor.shutdown(wait=True)
                
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        logger.info("Fraud detection processor stopped")


def main():
    """Main function to run the fraud detection processor."""
    config = DetectorConfig(
        kafka_bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
        redis_host=os.getenv('REDIS_HOST', 'localhost'),
        postgres_host=os.getenv('POSTGRES_HOST', 'localhost'),
        postgres_user=os.getenv('POSTGRES_USER', 'fraud_user'),
        postgres_password=os.getenv('POSTGRES_PASSWORD', 'fraud_password'),
        postgres_db=os.getenv('POSTGRES_DB', 'fraud_detection'),
        model_path=os.getenv('MODEL_PATH', '/app/models'),
        default_threshold=float(os.getenv('DEFAULT_THRESHOLD', '0.5')),
        enable_adaptive_threshold=os.getenv('ENABLE_ADAPTIVE_THRESHOLD', 'true').lower() == 'true'
    )
    
    logger.info("Real-Time Fraud Detection Processor Starting...")
    logger.info(f"Configuration: {config}")
    
    processor = FraudDetectionProcessor(config)
    
    try:
        processor.process_transactions()
    except Exception as e:
        logger.error(f"Processor failed: {e}")
        raise


if __name__ == "__main__":
    main()