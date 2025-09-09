"""
Synthetic Credit Card Transaction Data Generator for Fraud Detection

This module generates realistic credit card transaction data with both
legitimate and fraudulent patterns for training ML models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import random
from typing import Tuple, List, Dict
import json

fake = Faker()
random.seed(42)
np.random.seed(42)
Faker.seed(42)

class FraudDataGenerator:
    def __init__(self, num_users: int = 10000, days: int = 90):
        """
        Initialize the fraud data generator.
        
        Args:
            num_users: Number of unique users to generate
            days: Number of days of transaction history
        """
        self.num_users = num_users
        self.days = days
        self.start_date = datetime.now() - timedelta(days=days)
        
        # Define merchant categories
        self.merchant_categories = [
            'grocery', 'gas_station', 'restaurant', 'retail', 'online',
            'pharmacy', 'entertainment', 'travel', 'utilities', 'healthcare',
            'education', 'automotive', 'electronics', 'clothing', 'home_improvement'
        ]
        
        # Define countries and cities
        self.locations = [
            ('US', 'New York', 40.7128, -74.0060),
            ('US', 'Los Angeles', 34.0522, -118.2437),
            ('US', 'Chicago', 41.8781, -87.6298),
            ('US', 'Houston', 29.7604, -95.3698),
            ('US', 'Miami', 25.7617, -80.1918),
            ('CA', 'Toronto', 43.6532, -79.3832),
            ('UK', 'London', 51.5074, -0.1278),
            ('FR', 'Paris', 48.8566, 2.3522),
            ('JP', 'Tokyo', 35.6762, 139.6503),
            ('AU', 'Sydney', -33.8688, 151.2093)
        ]
        
        self.users = self._generate_users()
        
    def _generate_users(self) -> List[Dict]:
        """Generate user profiles with spending patterns."""
        users = []
        
        for user_id in range(self.num_users):
            # Random home location
            home_location = random.choice(self.locations)
            
            # User spending profile
            user = {
                'user_id': f'user_{user_id:06d}',
                'age': random.randint(18, 80),
                'income_level': random.choice(['low', 'medium', 'high']),
                'home_country': home_location[0],
                'home_city': home_location[1],
                'home_lat': home_location[2],
                'home_lng': home_location[3],
                'avg_transaction_amount': np.random.lognormal(3.5, 1.2),  # ~$50 average
                'transaction_frequency': np.random.gamma(2, 3),  # transactions per day
                'preferred_categories': random.sample(self.merchant_categories, k=random.randint(3, 7)),
                'active_hours': (random.randint(6, 10), random.randint(20, 23)),  # (start, end)
                'weekend_behavior': random.choice(['more_active', 'less_active', 'same']),
                'travel_frequency': random.choice(['never', 'rare', 'frequent']),
            }
            users.append(user)
            
        return users
    
    def _generate_legitimate_transaction(self, user: Dict, timestamp: datetime) -> Dict:
        """Generate a legitimate transaction for a user."""
        
        # Amount based on user profile with some randomness
        base_amount = user['avg_transaction_amount']
        amount = max(1.0, np.random.lognormal(np.log(base_amount), 0.5))
        
        # Category based on user preferences
        category = random.choice(user['preferred_categories'])
        
        # Location - mostly home location with some variance
        if random.random() < 0.8:  # 80% near home
            lat = user['home_lat'] + np.random.normal(0, 0.1)
            lng = user['home_lng'] + np.random.normal(0, 0.1)
            country = user['home_country']
            city = user['home_city']
        else:  # 20% travel
            location = random.choice(self.locations)
            country, city, lat, lng = location
        
        # Transaction details
        transaction = {
            'user_id': user['user_id'],
            'transaction_id': f'txn_{fake.uuid4()}',
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'merchant_category': category,
            'country': country,
            'city': city,
            'latitude': round(lat, 4),
            'longitude': round(lng, 4),
            'transaction_type': random.choice(['purchase', 'withdrawal', 'online']),
            'is_weekend': timestamp.weekday() >= 5,
            'hour': timestamp.hour,
            'is_fraud': 0
        }
        
        return transaction
    
    def _generate_fraudulent_transaction(self, user: Dict, timestamp: datetime) -> Dict:
        """Generate a fraudulent transaction with suspicious patterns."""
        
        fraud_patterns = ['high_amount', 'foreign_country', 'unusual_time', 'velocity_attack']
        pattern = random.choice(fraud_patterns)
        
        # Base legitimate transaction
        transaction = self._generate_legitimate_transaction(user, timestamp)
        transaction['is_fraud'] = 1
        
        if pattern == 'high_amount':
            # Unusually high amount for user
            transaction['amount'] = user['avg_transaction_amount'] * random.uniform(5, 20)
            
        elif pattern == 'foreign_country':
            # Transaction from unusual location
            foreign_location = random.choice([loc for loc in self.locations 
                                           if loc[0] != user['home_country']])
            transaction['country'] = foreign_location[0]
            transaction['city'] = foreign_location[1]
            transaction['latitude'] = foreign_location[2]
            transaction['longitude'] = foreign_location[3]
            
        elif pattern == 'unusual_time':
            # Transaction at unusual hours (3-6 AM)
            unusual_hour = random.randint(3, 6)
            transaction['timestamp'] = transaction['timestamp'].replace(hour=unusual_hour)
            transaction['hour'] = unusual_hour
            
        elif pattern == 'velocity_attack':
            # High amount for velocity attacks
            transaction['amount'] = random.uniform(100, 1000)
            
        transaction['amount'] = round(transaction['amount'], 2)
        return transaction
    
    def generate_dataset(self, fraud_rate: float = 0.02) -> pd.DataFrame:
        """
        Generate the complete dataset with legitimate and fraudulent transactions.
        
        Args:
            fraud_rate: Percentage of fraudulent transactions (default 2%)
            
        Returns:
            DataFrame with all transactions
        """
        transactions = []
        
        print("Generating synthetic transaction data...")
        print(f"Users: {self.num_users}")
        print(f"Days: {self.days}")
        print(f"Fraud rate: {fraud_rate*100:.1f}%")
        
        # Generate transactions for each user
        for user in self.users:
            user_transactions = self._generate_user_transactions(user, fraud_rate)
            transactions.extend(user_transactions)
        
        # Convert to DataFrame
        df = pd.DataFrame(transactions)
        
        # Add derived features
        df = self._add_derived_features(df)
        
        print(f"\nGenerated {len(df):,} transactions")
        print(f"Fraudulent transactions: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")
        print(f"Legitimate transactions: {(~df['is_fraud'].astype(bool)).sum():,}")
        
        return df
    
    def _generate_user_transactions(self, user: Dict, fraud_rate: float) -> List[Dict]:
        """Generate transactions for a single user."""
        transactions = []
        
        # Calculate total transactions for this user
        daily_freq = max(0.1, user['transaction_frequency'])
        total_transactions = int(np.random.poisson(daily_freq * self.days))
        
        # Generate random timestamps
        timestamps = []
        for _ in range(total_transactions):
            random_day = random.randint(0, self.days - 1)
            
            # Prefer active hours
            if random.random() < 0.7:  # 70% during active hours
                hour = random.randint(user['active_hours'][0], user['active_hours'][1])
            else:
                hour = random.randint(0, 23)
                
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            
            timestamp = self.start_date + timedelta(
                days=random_day,
                hours=hour,
                minutes=minute,
                seconds=second
            )
            timestamps.append(timestamp)
        
        timestamps.sort()
        
        # Generate transactions
        for timestamp in timestamps:
            if random.random() < fraud_rate:
                transaction = self._generate_fraudulent_transaction(user, timestamp)
            else:
                transaction = self._generate_legitimate_transaction(user, timestamp)
            
            transactions.append(transaction)
        
        return transactions
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for better model performance."""
        
        print("Adding derived features...")
        
        # Sort by user and timestamp
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
        
        # Time-based features
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_night'] = ((df['hour'] >= 23) | (df['hour'] <= 6)).astype(int)
        
        # User-based rolling features (simplified calculation)
        # Convert to datetime if not already
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by user and timestamp for rolling calculations
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
        
        # Initialize rolling features with default values
        df['user_transaction_count_24h'] = 0
        df['user_amount_sum_24h'] = 0.0
        
        # Simple approximation of rolling features (for demo purposes)
        # In a real system, these would be computed in real-time
        user_txn_counts = df.groupby('user_id').cumcount()
        df['user_transaction_count_24h'] = np.minimum(user_txn_counts, 50)  # Cap at reasonable number
        
        user_cumsum = df.groupby('user_id')['amount'].cumsum()
        df['user_amount_sum_24h'] = user_cumsum * 0.7  # Approximation of 24h window
        
        # Distance from previous transaction (simplified)
        df['prev_lat'] = df.groupby('user_id')['latitude'].shift(1)
        df['prev_lng'] = df.groupby('user_id')['longitude'].shift(1)
        
        # Approximate distance (not exact, but good for synthetic data)
        df['distance_from_prev'] = np.sqrt(
            (df['latitude'] - df['prev_lat'])**2 + 
            (df['longitude'] - df['prev_lng'])**2
        ) * 111  # Rough conversion to km
        
        df['distance_from_prev'] = df['distance_from_prev'].fillna(0)
        
        # Time since last transaction
        df['time_since_prev'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 3600  # hours
        df['time_since_prev'] = df['time_since_prev'].fillna(24)  # Default to 24 hours for first transaction
        
        # Clean up temporary columns
        df = df.drop(['prev_lat', 'prev_lng'], axis=1)
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filename: str = None) -> str:
        """Save the dataset to CSV file."""
        if filename is None:
            filename = f"fraud_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = f"/home/hduser/projects/real_time_fraud_detection/data/raw/{filename}"
        df.to_csv(filepath, index=False)
        
        print(f"Dataset saved to: {filepath}")
        return filepath


def main():
    """Generate and save synthetic fraud detection dataset."""
    
    # Initialize generator
    generator = FraudDataGenerator(num_users=5000, days=90)
    
    # Generate dataset
    df = generator.generate_dataset(fraud_rate=0.025)  # 2.5% fraud rate
    
    # Save to file
    filepath = generator.save_dataset(df)
    
    # Display basic statistics
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Unique users: {df['user_id'].nunique():,}")
    print(f"Unique merchants: {df['merchant_category'].nunique()}")
    print(f"Countries: {sorted(df['country'].unique())}")
    
    print(f"\nFraud Distribution:")
    print(df['is_fraud'].value_counts())
    
    print(f"\nAmount Statistics:")
    print(df['amount'].describe())
    
    print(f"\nTransaction Types:")
    print(df['transaction_type'].value_counts())
    
    print(f"\nMerchant Categories:")
    print(df['merchant_category'].value_counts())
    
    return df


if __name__ == "__main__":
    dataset = main()