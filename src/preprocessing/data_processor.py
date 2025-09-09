"""
Data Preprocessing Module for Fraud Detection

This module handles data cleaning, feature engineering, encoding,
and preparation for machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Any
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FraudDataProcessor:
    """
    Comprehensive data processor for fraud detection datasets.
    
    Handles:
    - Data cleaning and validation
    - Feature engineering and selection
    - Categorical encoding
    - Feature scaling
    - Data splitting
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = None
        self.feature_columns = None
        self.target_column = 'is_fraud'
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load dataset from CSV file."""
        print(f"Loading data from: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            raise
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the dataset."""
        print("Validating and cleaning data...")
        
        initial_shape = df.shape
        
        # Check for required columns
        required_columns = ['user_id', 'timestamp', 'amount', 'is_fraud']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert timestamp to datetime
        if df['timestamp'].dtype == 'object':
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Remove duplicate transactions
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['user_id', 'timestamp', 'amount'], keep='first')
        after_dedup = len(df)
        if before_dedup != after_dedup:
            print(f"Removed {before_dedup - after_dedup} duplicate transactions")
        
        # Remove transactions with invalid amounts
        df = df[df['amount'] > 0]
        
        # Remove transactions with missing critical data
        critical_columns = ['user_id', 'timestamp', 'amount', 'merchant_category']
        df = df.dropna(subset=critical_columns)
        
        # Cap extreme amounts (likely data errors)
        amount_99th = df['amount'].quantile(0.99)
        df.loc[df['amount'] > amount_99th * 10, 'amount'] = amount_99th
        
        print(f"Data validation complete. Shape: {initial_shape} -> {df.shape}")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for better model performance."""
        print("Engineering features...")
        
        df = df.copy()
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by user and timestamp for rolling calculations
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
        
        # === TIME-BASED FEATURES ===
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 23) | (df['hour'] <= 6)).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        # === AMOUNT-BASED FEATURES ===
        df['amount_log'] = np.log1p(df['amount'])
        
        # User spending patterns
        user_stats = df.groupby('user_id')['amount'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).reset_index()
        user_stats.columns = ['user_id', 'user_avg_amount', 'user_std_amount', 
                             'user_min_amount', 'user_max_amount', 'user_total_transactions']
        
        df = df.merge(user_stats, on='user_id', how='left')
        
        # Amount deviation from user's normal behavior
        df['amount_deviation_from_user_avg'] = (df['amount'] - df['user_avg_amount']) / (df['user_std_amount'] + 1)
        df['amount_vs_user_max'] = df['amount'] / (df['user_max_amount'] + 1)
        
        # === VELOCITY FEATURES ===
        # Use existing velocity features from synthetic data generation
        # The synthetic data already includes these features: user_transaction_count_24h, user_amount_sum_24h
        
        if 'user_transaction_count_24h' in df.columns:
            df['user_txn_count_24h'] = df['user_transaction_count_24h']
        else:
            df['user_txn_count_24h'] = 0
            
        if 'user_amount_sum_24h' in df.columns:
            df['user_amount_sum_24h_generated'] = df['user_amount_sum_24h']
        else:
            df['user_amount_sum_24h_generated'] = 0
        
        # Create simplified velocity features (1-hour approximations based on time patterns)
        df['user_txn_count_1h'] = (df['user_txn_count_24h'] / 24).fillna(0)  # Approximate hourly rate
        df['user_amount_sum_1h'] = (df['user_amount_sum_24h_generated'] / 24).fillna(0)  # Approximate hourly amount
        
        # === LOCATION-BASED FEATURES ===
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Distance from previous transaction
            df['prev_lat'] = df.groupby('user_id')['latitude'].shift(1)
            df['prev_lng'] = df.groupby('user_id')['longitude'].shift(1)
            
            # Haversine distance approximation
            df['distance_from_prev_km'] = self._calculate_distance(
                df['latitude'], df['longitude'], 
                df['prev_lat'], df['prev_lng']
            )
            df['distance_from_prev_km'] = df['distance_from_prev_km'].fillna(0)
            
            # Time since previous transaction
            df['time_since_prev_hours'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 3600
            df['time_since_prev_hours'] = df['time_since_prev_hours'].fillna(24)
            
            # Velocity (distance/time)
            df['travel_velocity_kmh'] = df['distance_from_prev_km'] / (df['time_since_prev_hours'] + 0.1)
            df['is_impossible_travel'] = (df['travel_velocity_kmh'] > 1000).astype(int)  # > 1000 km/h
            
            # Clean up temporary columns
            df = df.drop(['prev_lat', 'prev_lng'], axis=1)
        
        # === MERCHANT CATEGORY FEATURES ===
        # User's favorite categories
        user_categories = df.groupby(['user_id', 'merchant_category']).size().reset_index(name='category_count')
        user_top_category = user_categories.loc[user_categories.groupby('user_id')['category_count'].idxmax()]
        user_top_category = user_top_category[['user_id', 'merchant_category']].rename(
            columns={'merchant_category': 'user_top_category'}
        )
        
        df = df.merge(user_top_category, on='user_id', how='left')
        df['is_unusual_category'] = (df['merchant_category'] != df['user_top_category']).astype(int)
        
        # === TRANSACTION TYPE FEATURES ===
        # Frequency of transaction types for user
        if 'transaction_type' in df.columns:
            user_txn_types = df.groupby(['user_id', 'transaction_type']).size().reset_index(name='type_count')
            user_txn_type_totals = df.groupby('user_id').size().reset_index(name='total_txns')
            user_txn_types = user_txn_types.merge(user_txn_type_totals, on='user_id')
            user_txn_types['type_frequency'] = user_txn_types['type_count'] / user_txn_types['total_txns']
            
            # Get most common transaction type for each user
            user_common_type = user_txn_types.loc[user_txn_types.groupby('user_id')['type_frequency'].idxmax()]
            user_common_type = user_common_type[['user_id', 'transaction_type']].rename(
                columns={'transaction_type': 'user_common_txn_type'}
            )
            
            df = df.merge(user_common_type, on='user_id', how='left')
            df['is_unusual_txn_type'] = (df['transaction_type'] != df['user_common_txn_type']).astype(int)
        
        print(f"Feature engineering complete. New shape: {df.shape}")
        
        return df
    
    def _calculate_distance(self, lat1: pd.Series, lon1: pd.Series, 
                          lat2: pd.Series, lon2: pd.Series) -> pd.Series:
        """Calculate approximate distance between two points in kilometers."""
        # Simple Euclidean distance approximation (good enough for synthetic data)
        lat_diff = lat1 - lat2
        lon_diff = lon1 - lon2
        distance = np.sqrt(lat_diff**2 + lon_diff**2) * 111  # Approximate km per degree
        return distance
    
    def encode_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables."""
        print("Encoding categorical features...")
        
        df = df.copy()
        
        # Define categorical columns to encode
        categorical_columns = []
        
        # Check which categorical columns exist
        potential_categorical = ['merchant_category', 'country', 'city', 'transaction_type', 
                               'user_top_category', 'user_common_txn_type']
        
        for col in potential_categorical:
            if col in df.columns:
                categorical_columns.append(col)
        
        # Encode categorical variables
        for col in categorical_columns:
            if fit:
                # Fit new encoder
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                # Use existing encoder
                if col in self.label_encoders:
                    # Handle unseen categories
                    le = self.label_encoders[col]
                    categories = df[col].astype(str)
                    encoded = []
                    for cat in categories:
                        if cat in le.classes_:
                            encoded.append(le.transform([cat])[0])
                        else:
                            # Assign most frequent class for unseen categories
                            encoded.append(0)  # or le.transform([le.classes_[0]])[0]
                    df[f'{col}_encoded'] = encoded
        
        print(f"Encoded {len(categorical_columns)} categorical features")
        
        return df
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select relevant features for modeling."""
        print("Selecting features for modeling...")
        
        # Define features to use
        numeric_features = [
            'amount', 'amount_log', 'hour', 'day_of_week', 'month',
            'is_weekend', 'is_night', 'is_business_hours',
            'user_avg_amount', 'user_std_amount', 'user_total_transactions',
            'amount_deviation_from_user_avg', 'amount_vs_user_max',
            'user_txn_count_1h', 'user_txn_count_24h',
            'user_amount_sum_1h', 'user_amount_sum_24h',
            'is_unusual_category', 'is_unusual_txn_type'
        ]
        
        # Add location features if available
        if 'distance_from_prev_km' in df.columns:
            numeric_features.extend([
                'distance_from_prev_km', 'time_since_prev_hours',
                'travel_velocity_kmh', 'is_impossible_travel'
            ])
        
        # Add encoded categorical features
        encoded_features = [col for col in df.columns if col.endswith('_encoded')]
        
        # Combine all features
        all_features = numeric_features + encoded_features
        
        # Select only features that exist in the dataframe
        available_features = [col for col in all_features if col in df.columns]
        
        # Add target variable
        final_columns = available_features + [self.target_column]
        
        # Select features
        df_selected = df[final_columns].copy()
        
        # Handle missing values
        df_selected = df_selected.fillna(0)
        
        # Store feature columns for later use
        self.feature_columns = available_features
        
        print(f"Selected {len(available_features)} features for modeling")
        print("Features:", available_features)
        
        return df_selected
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features."""
        print("Scaling features...")
        
        if fit:
            # Fit new scaler
            self.scaler = RobustScaler()  # Less sensitive to outliers
            X_scaled = self.scaler.fit_transform(X)
        else:
            # Use existing scaler
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            X_scaled = self.scaler.transform(X)
        
        # Convert back to DataFrame
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return X_scaled_df
    
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                    random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Complete data preparation pipeline.
        
        Args:
            df: Raw dataframe
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print("Starting complete data preparation pipeline...")
        
        # 1. Validate data
        df = self.validate_data(df)
        
        # 2. Engineer features
        df = self.engineer_features(df)
        
        # 3. Encode categorical features
        df = self.encode_features(df, fit=True)
        
        # 4. Select features
        df = self.select_features(df)
        
        # 5. Split into features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # 6. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y  # Maintain fraud ratio in both sets
        )
        
        # 7. Scale features
        X_train_scaled = self.scale_features(X_train, fit=True)
        X_test_scaled = self.scale_features(X_test, fit=False)
        
        print(f"Data preparation complete!")
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        print(f"Fraud rate in training: {y_train.mean():.3f}")
        print(f"Fraud rate in test: {y_test.mean():.3f}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_preprocessor(self, filepath: str = None) -> str:
        """Save the preprocessor for later use."""
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"/home/hduser/projects/real_time_fraud_detection/data/models/preprocessor_{timestamp}.joblib"
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        preprocessor_data = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }
        
        joblib.dump(preprocessor_data, filepath)
        print(f"Preprocessor saved to: {filepath}")
        
        return filepath
    
    def load_preprocessor(self, filepath: str):
        """Load a saved preprocessor."""
        preprocessor_data = joblib.load(filepath)
        
        self.label_encoders = preprocessor_data['label_encoders']
        self.scaler = preprocessor_data['scaler']
        self.feature_columns = preprocessor_data['feature_columns']
        self.target_column = preprocessor_data['target_column']
        
        print(f"Preprocessor loaded from: {filepath}")


def main():
    """Example usage of the data processor."""
    
    # Initialize processor
    processor = FraudDataProcessor()
    
    # Load data (replace with actual path)
    data_path = "/home/hduser/projects/real_time_fraud_detection/data/raw/"
    
    # Find the most recent dataset
    import glob
    csv_files = glob.glob(f"{data_path}*.csv")
    if not csv_files:
        print("No CSV files found in data/raw directory")
        print("Please run the synthetic_data_generator.py first")
        return
    
    latest_file = max(csv_files, key=os.path.getctime)
    print(f"Using dataset: {latest_file}")
    
    # Load and process data
    df = processor.load_data(latest_file)
    
    # Prepare data for modeling
    X_train, X_test, y_train, y_test = processor.prepare_data(df)
    
    # Save preprocessor
    processor.save_preprocessor()
    
    # Save processed data
    processed_data_path = "/home/hduser/projects/real_time_fraud_detection/data/processed/"
    os.makedirs(processed_data_path, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    X_train.to_csv(f"{processed_data_path}X_train_{timestamp}.csv", index=False)
    X_test.to_csv(f"{processed_data_path}X_test_{timestamp}.csv", index=False)
    y_train.to_csv(f"{processed_data_path}y_train_{timestamp}.csv", index=False)
    y_test.to_csv(f"{processed_data_path}y_test_{timestamp}.csv", index=False)
    
    print(f"Processed data saved to: {processed_data_path}")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = main()