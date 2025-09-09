#!/usr/bin/env python3
"""
Real-Time Transaction Generator for Fraud Detection
==================================================

This service generates realistic transaction data in real-time and streams it to Kafka.
Simulates normal user behavior with configurable fraud injection.

Features:
- Configurable transaction rate (TPS)
- Realistic user behaviors and patterns
- Fraud scenario injection
- Kafka streaming integration
- Monitoring and metrics
"""

import json
import time
import random
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
from faker import Faker
from kafka import KafkaProducer
from kafka.errors import KafkaError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TransactionConfig:
    """Configuration for transaction generation."""
    transactions_per_second: int = 10
    fraud_rate: float = 0.05
    num_active_users: int = 1000
    kafka_topic: str = 'fraud-transactions'
    kafka_bootstrap_servers: str = 'localhost:9092'

class UserProfile:
    """Represents a user with consistent behavior patterns."""
    
    def __init__(self, user_id: str, fake: Faker):
        self.user_id = user_id
        self.fake = fake
        
        # User demographics and preferences
        self.name = fake.name()
        self.age = random.randint(18, 80)
        self.location = {
            'city': fake.city(),
            'country': fake.country(),
            'latitude': float(fake.latitude()),
            'longitude': float(fake.longitude())
        }
        
        # Spending patterns
        self.avg_transaction_amount = np.random.lognormal(mean=4.0, sigma=1.2)  # ~$55 average
        self.spending_volatility = random.uniform(0.3, 2.0)
        self.preferred_categories = random.choices([
            'grocery', 'gas_station', 'restaurant', 'retail', 'online',
            'entertainment', 'pharmacy', 'transport', 'utilities', 'other'
        ], k=random.randint(2, 5))
        
        # Behavioral patterns
        self.active_hours = self._generate_active_hours()
        self.transaction_frequency = random.uniform(0.5, 5.0)  # transactions per hour when active
        self.risk_profile = random.choice(['low', 'medium', 'high'])
        
        # Transaction history for pattern consistency
        self.last_transaction_time = None
        self.last_location = self.location.copy()
        self.daily_spending = 0.0
        self.daily_transaction_count = 0
        self.last_reset_date = datetime.now().date()
        
    def _generate_active_hours(self) -> List[int]:
        """Generate typical active hours for this user."""
        # Most users are active during business hours and evenings
        base_hours = list(range(7, 23))  # 7 AM to 11 PM
        
        # Add some variation
        if random.random() < 0.3:  # 30% night owls
            base_hours.extend([23, 0, 1])
        if random.random() < 0.4:  # 40% early birds
            base_hours.extend([6, 5])
            
        return sorted(list(set(base_hours)))
    
    def should_transact_now(self) -> bool:
        """Determine if user should make a transaction now based on patterns."""
        current_time = datetime.now()
        current_hour = current_time.hour
        
        # Check if current hour is in active hours
        if current_hour not in self.active_hours:
            return False
            
        # Reset daily counters if new day
        if current_time.date() != self.last_reset_date:
            self.daily_spending = 0.0
            self.daily_transaction_count = 0
            self.last_reset_date = current_time.date()
        
        # Check if user has exceeded daily limits (natural behavior)
        max_daily_transactions = random.randint(3, 15)
        max_daily_spending = self.avg_transaction_amount * random.uniform(3, 8)
        
        if (self.daily_transaction_count >= max_daily_transactions or 
            self.daily_spending >= max_daily_spending):
            return False
        
        # Probability based on frequency and time since last transaction
        base_probability = self.transaction_frequency / 3600  # per second
        
        if self.last_transaction_time:
            time_since_last = (current_time - self.last_transaction_time).total_seconds()
            # Decrease probability if recent transaction
            if time_since_last < 300:  # 5 minutes
                base_probability *= 0.1
            elif time_since_last < 3600:  # 1 hour
                base_probability *= 0.5
        
        return random.random() < base_probability

    def generate_transaction(self, force_fraud: bool = False) -> Dict[str, Any]:
        """Generate a transaction for this user."""
        current_time = datetime.now()
        
        # Determine if this should be fraud
        is_fraud = force_fraud or (random.random() < 0.02)  # 2% base fraud rate per transaction
        
        if is_fraud:
            transaction = self._generate_fraud_transaction(current_time)
        else:
            transaction = self._generate_normal_transaction(current_time)
        
        # Update user state
        self.last_transaction_time = current_time
        self.daily_spending += transaction['amount']
        self.daily_transaction_count += 1
        
        if not is_fraud:
            # Update location for normal transactions
            self.last_location = {
                'latitude': transaction['latitude'],
                'longitude': transaction['longitude']
            }
        
        return transaction
    
    def _generate_normal_transaction(self, timestamp: datetime) -> Dict[str, Any]:
        """Generate a normal (non-fraudulent) transaction."""
        # Amount based on user's typical spending pattern
        amount = np.random.lognormal(
            mean=np.log(self.avg_transaction_amount),
            sigma=self.spending_volatility * 0.5
        )
        amount = max(1.0, min(amount, 5000.0))  # Cap between $1 and $5000
        
        # Category from user's preferences
        category = random.choice(self.preferred_categories)
        
        # Location close to user's typical location
        lat_offset = random.uniform(-0.1, 0.1)  # ~11km radius
        lng_offset = random.uniform(-0.1, 0.1)
        location = {
            'latitude': self.last_location['latitude'] + lat_offset,
            'longitude': self.last_location['longitude'] + lng_offset,
            'city': self.location['city'],
            'country': self.location['country']
        }
        
        return {
            'transaction_id': f"txn_{int(timestamp.timestamp() * 1000)}_{self.user_id}",
            'user_id': self.user_id,
            'timestamp': timestamp.isoformat(),
            'amount': round(amount, 2),
            'merchant_category': category,
            'transaction_type': random.choice(['card_present', 'online', 'contactless']),
            'is_weekend': timestamp.weekday() >= 5,
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'month': timestamp.month,
            'is_night': timestamp.hour < 6 or timestamp.hour > 22,
            'latitude': location['latitude'],
            'longitude': location['longitude'],
            'city': location['city'],
            'country': location['country'],
            'is_fraud': 0,
            'fraud_type': None
        }
    
    def _generate_fraud_transaction(self, timestamp: datetime) -> Dict[str, Any]:
        """Generate a fraudulent transaction with common fraud patterns."""
        fraud_types = ['card_testing', 'account_takeover', 'stolen_card', 'synthetic_identity']
        fraud_type = random.choice(fraud_types)
        
        if fraud_type == 'card_testing':
            # Small amounts, frequent transactions
            amount = random.uniform(1.0, 50.0)
            category = random.choice(['online', 'retail'])
            
        elif fraud_type == 'account_takeover':
            # Unusual category, different location
            amount = random.uniform(100.0, 2000.0)
            category = random.choice(['electronics', 'jewelry', 'travel'])
            
        elif fraud_type == 'stolen_card':
            # High amounts, unusual locations/times
            amount = random.uniform(200.0, 3000.0)
            category = random.choice(['retail', 'online', 'atm_withdrawal'])
            
        else:  # synthetic_identity
            # New patterns entirely
            amount = random.uniform(50.0, 1500.0)
            category = random.choice(['online', 'retail', 'financial'])
        
        # Unusual location (far from user's normal location)
        if fraud_type in ['stolen_card', 'account_takeover']:
            # Different country/region
            fake_location = {
                'latitude': random.uniform(-90, 90),
                'longitude': random.uniform(-180, 180),
                'city': self.fake.city(),
                'country': self.fake.country()
            }
        else:
            # Same general area but unusual
            lat_offset = random.uniform(-0.5, 0.5)
            lng_offset = random.uniform(-0.5, 0.5)
            fake_location = {
                'latitude': self.location['latitude'] + lat_offset,
                'longitude': self.location['longitude'] + lng_offset,
                'city': self.location['city'],
                'country': self.location['country']
            }
        
        return {
            'transaction_id': f"txn_{int(timestamp.timestamp() * 1000)}_{self.user_id}_fraud",
            'user_id': self.user_id,
            'timestamp': timestamp.isoformat(),
            'amount': round(amount, 2),
            'merchant_category': category,
            'transaction_type': random.choice(['card_not_present', 'online', 'card_present']),
            'is_weekend': timestamp.weekday() >= 5,
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'month': timestamp.month,
            'is_night': timestamp.hour < 6 or timestamp.hour > 22,
            'latitude': fake_location['latitude'],
            'longitude': fake_location['longitude'],
            'city': fake_location['city'],
            'country': fake_location['country'],
            'is_fraud': 1,
            'fraud_type': fraud_type
        }


class RealTimeTransactionGenerator:
    """Main class for generating real-time transaction stream."""
    
    def __init__(self, config: TransactionConfig):
        self.config = config
        self.fake = Faker()
        
        # Initialize Kafka producer
        self.producer = self._initialize_kafka_producer()
        
        # Generate user profiles
        logger.info(f"Generating {config.num_active_users} user profiles...")
        self.users = [
            UserProfile(f"user_{i:06d}", self.fake) 
            for i in range(config.num_active_users)
        ]
        logger.info(f"Generated {len(self.users)} user profiles")
        
        # Statistics
        self.stats = {
            'total_transactions': 0,
            'fraud_transactions': 0,
            'normal_transactions': 0,
            'start_time': datetime.now()
        }
        
    def _initialize_kafka_producer(self) -> KafkaProducer:
        """Initialize Kafka producer with error handling."""
        try:
            producer = KafkaProducer(
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                key_serializer=lambda x: x.encode('utf-8') if x else None,
                retries=3,
                retry_backoff_ms=1000,
                request_timeout_ms=30000,
                max_block_ms=10000
            )
            logger.info(f"Kafka producer initialized: {self.config.kafka_bootstrap_servers}")
            return producer
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            raise
    
    def generate_transactions(self):
        """Main transaction generation loop."""
        logger.info(f"Starting transaction generation: {self.config.transactions_per_second} TPS")
        logger.info(f"Target fraud rate: {self.config.fraud_rate * 100:.1f}%")
        
        transactions_per_interval = max(1, self.config.transactions_per_second)
        interval = 1.0  # 1 second intervals
        
        while True:
            start_time = time.time()
            
            # Generate batch of transactions for this interval
            transactions = []
            for _ in range(transactions_per_interval):
                transaction = self._generate_single_transaction()
                if transaction:
                    transactions.append(transaction)
            
            # Send transactions to Kafka
            self._send_transactions_batch(transactions)
            
            # Update statistics
            self._update_stats(transactions)
            
            # Sleep to maintain desired rate
            elapsed = time.time() - start_time
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Log statistics periodically
            if self.stats['total_transactions'] % 100 == 0:
                self._log_stats()
    
    def _generate_single_transaction(self) -> Optional[Dict[str, Any]]:
        """Generate a single transaction from a random active user."""
        # Select random user
        user = random.choice(self.users)
        
        # Check if user should transact now
        if not user.should_transact_now():
            return None
        
        # Force fraud based on configured rate
        force_fraud = random.random() < self.config.fraud_rate
        
        return user.generate_transaction(force_fraud=force_fraud)
    
    def _send_transactions_batch(self, transactions: List[Dict[str, Any]]):
        """Send batch of transactions to Kafka."""
        for transaction in transactions:
            try:
                # Use user_id as partition key for consistent routing
                self.producer.send(
                    self.config.kafka_topic,
                    key=transaction['user_id'],
                    value=transaction
                ).add_callback(self._on_send_success).add_errback(self._on_send_error)
                
            except Exception as e:
                logger.error(f"Failed to send transaction {transaction['transaction_id']}: {e}")
        
        # Ensure messages are sent
        self.producer.flush()
    
    def _on_send_success(self, record_metadata):
        """Callback for successful Kafka send."""
        logger.debug(f"Transaction sent to {record_metadata.topic}[{record_metadata.partition}]")
    
    def _on_send_error(self, exception):
        """Callback for failed Kafka send."""
        logger.error(f"Failed to send transaction: {exception}")
    
    def _update_stats(self, transactions: List[Dict[str, Any]]):
        """Update generation statistics."""
        for transaction in transactions:
            self.stats['total_transactions'] += 1
            if transaction['is_fraud']:
                self.stats['fraud_transactions'] += 1
            else:
                self.stats['normal_transactions'] += 1
    
    def _log_stats(self):
        """Log current statistics."""
        elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
        rate = self.stats['total_transactions'] / elapsed if elapsed > 0 else 0
        fraud_rate = (self.stats['fraud_transactions'] / self.stats['total_transactions'] 
                     if self.stats['total_transactions'] > 0 else 0)
        
        logger.info(
            f"Generated {self.stats['total_transactions']} transactions "
            f"({rate:.1f} TPS, {fraud_rate*100:.1f}% fraud)"
        )
    
    def close(self):
        """Clean shutdown."""
        logger.info("Shutting down transaction generator...")
        if self.producer:
            self.producer.close()
        logger.info("Transaction generator stopped")


def main():
    """Main function to run the transaction generator."""
    # Load configuration from environment
    config = TransactionConfig(
        transactions_per_second=int(os.getenv('TRANSACTIONS_PER_SECOND', '10')),
        fraud_rate=float(os.getenv('FRAUD_RATE', '0.05')),
        num_active_users=int(os.getenv('NUM_ACTIVE_USERS', '1000')),
        kafka_topic=os.getenv('KAFKA_TOPIC', 'fraud-transactions'),
        kafka_bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    )
    
    logger.info("Real-Time Transaction Generator Starting...")
    logger.info(f"Configuration: {config}")
    
    generator = RealTimeTransactionGenerator(config)
    
    try:
        generator.generate_transactions()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Generator failed: {e}")
        raise
    finally:
        generator.close()


if __name__ == "__main__":
    main()