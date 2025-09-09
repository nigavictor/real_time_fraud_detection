"""
Real-Time Fraud Detection System - Main Pipeline

This script orchestrates the complete fraud detection pipeline:
1. Synthetic data generation
2. Data preprocessing 
3. Model training and evaluation
4. Results visualization and reporting

Run this script to execute the entire pipeline.
"""

import sys
import os
sys.path.append('src')

from src.data_generation.synthetic_data_generator import FraudDataGenerator
from src.preprocessing.data_processor import FraudDataProcessor
from src.modeling.model_trainer import FraudModelTrainer
from src.evaluation.model_evaluator import FraudModelEvaluator

import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionPipeline:
    """
    Complete fraud detection pipeline orchestrator.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Configuration dictionary with pipeline parameters
        """
        self.config = config or self._get_default_config()
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Initialize components
        self.data_generator = None
        self.data_processor = None
        self.model_trainer = None
        self.model_evaluator = None
        
        # Store intermediate results
        self.raw_data_path = None
        self.processed_data = None
        self.trained_models = None
        self.evaluation_results = None
        
    def _get_default_config(self) -> dict:
        """Get default pipeline configuration."""
        return {
            'data_generation': {
                'num_users': 5000,
                'days': 90,
                'fraud_rate': 0.025
            },
            'preprocessing': {
                'test_size': 0.2,
                'random_state': 42
            },
            'modeling': {
                'models_to_train': ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm'],
                'use_sampling': True,
                'create_ensemble': True
            },
            'evaluation': {
                'generate_visualizations': True,
                'business_analysis': True,
                'threshold_optimization': True
            }
        }
    
    def run_data_generation(self) -> str:
        """
        Step 1: Generate synthetic fraud detection dataset.
        
        Returns:
            Path to the generated raw dataset
        """
        print("\n" + "="*60)
        print("STEP 1: SYNTHETIC DATA GENERATION")
        print("="*60)
        
        # Initialize data generator
        config = self.config['data_generation']
        self.data_generator = FraudDataGenerator(
            num_users=config['num_users'],
            days=config['days']
        )
        
        # Generate dataset
        df = self.data_generator.generate_dataset(fraud_rate=config['fraud_rate'])
        
        # Save dataset
        filename = f"fraud_transactions_{self.timestamp}.csv"
        self.raw_data_path = self.data_generator.save_dataset(df, filename)
        
        print(f"✓ Data generation completed: {len(df):,} transactions generated")
        print(f"✓ Dataset saved to: {self.raw_data_path}")
        
        return self.raw_data_path
    
    def run_data_preprocessing(self) -> tuple:
        """
        Step 2: Preprocess data for machine learning.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print("\n" + "="*60)
        print("STEP 2: DATA PREPROCESSING")
        print("="*60)
        
        # Initialize data processor
        self.data_processor = FraudDataProcessor()
        
        # Load raw data
        if self.raw_data_path is None:
            # Look for existing raw data file
            raw_data_dir = "/home/hduser/projects/real_time_fraud_detection/data/raw/"
            if os.path.exists(raw_data_dir):
                raw_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
                if raw_files:
                    # Use the most recent file
                    raw_files.sort(reverse=True)
                    self.raw_data_path = os.path.join(raw_data_dir, raw_files[0])
                    print(f"Found existing raw data: {self.raw_data_path}")
                else:
                    raise ValueError("No raw data found. Run data generation first.")
            else:
                raise ValueError("Raw data directory not found. Run data generation first.")
        
        df = self.data_processor.load_data(self.raw_data_path)
        
        # Prepare data
        config = self.config['preprocessing']
        X_train, X_test, y_train, y_test = self.data_processor.prepare_data(
            df, 
            test_size=config['test_size'],
            random_state=config['random_state']
        )
        
        # Save preprocessor
        preprocessor_path = self.data_processor.save_preprocessor()
        
        # Save processed data
        processed_dir = "/home/hduser/projects/real_time_fraud_detection/data/processed/"
        os.makedirs(processed_dir, exist_ok=True)
        
        X_train.to_csv(f"{processed_dir}X_train_{self.timestamp}.csv", index=False)
        X_test.to_csv(f"{processed_dir}X_test_{self.timestamp}.csv", index=False)
        y_train.to_csv(f"{processed_dir}y_train_{self.timestamp}.csv", index=False)
        y_test.to_csv(f"{processed_dir}y_test_{self.timestamp}.csv", index=False)
        
        self.processed_data = (X_train, X_test, y_train, y_test)
        
        print(f"✓ Data preprocessing completed")
        print(f"✓ Training set: {X_train.shape}")
        print(f"✓ Test set: {X_test.shape}")
        print(f"✓ Preprocessor saved: {preprocessor_path}")
        
        return self.processed_data
    
    def run_model_training(self) -> dict:
        """
        Step 3: Train and evaluate ML models.
        
        Returns:
            Dictionary of trained models
        """
        print("\n" + "="*60)
        print("STEP 3: MODEL TRAINING & EVALUATION")
        print("="*60)
        
        if self.processed_data is None:
            # Look for existing processed data files
            processed_data_dir = "/home/hduser/projects/real_time_fraud_detection/data/processed/"
            if os.path.exists(processed_data_dir):
                processed_files = [f for f in os.listdir(processed_data_dir) if f.endswith('.csv')]
                if processed_files:
                    # Find the most recent set of processed files
                    timestamps = set()
                    for f in processed_files:
                        if 'X_train_' in f and f.endswith('.csv'):
                            # Extract timestamp from X_train_YYYYMMDD_HHMMSS.csv format
                            parts = f.replace('.csv', '').split('_')
                            if len(parts) >= 3:
                                timestamp = '_'.join(parts[-2:])  # Get YYYYMMDD_HHMMSS
                                timestamps.add(timestamp)
                    
                    if timestamps:
                        latest_timestamp = sorted(timestamps, reverse=True)[0]
                        print(f"Found processed data with timestamp: {latest_timestamp}")
                        
                        # Load processed data
                        X_train = pd.read_csv(f"{processed_data_dir}X_train_{latest_timestamp}.csv")
                        X_test = pd.read_csv(f"{processed_data_dir}X_test_{latest_timestamp}.csv")
                        y_train = pd.read_csv(f"{processed_data_dir}y_train_{latest_timestamp}.csv").squeeze()
                        y_test = pd.read_csv(f"{processed_data_dir}y_test_{latest_timestamp}.csv").squeeze()
                        
                        self.processed_data = (X_train, X_test, y_train, y_test)
                    else:
                        raise ValueError("No valid processed data found. Run preprocessing first.")
                else:
                    raise ValueError("No processed data files found. Run preprocessing first.")
            else:
                raise ValueError("Processed data directory not found. Run preprocessing first.")
        
        X_train, X_test, y_train, y_test = self.processed_data
        
        # Initialize model trainer
        self.model_trainer = FraudModelTrainer()
        
        # Train models
        config = self.config['modeling']
        models_to_train = config.get('models_to_train', None)
        
        self.trained_models = self.model_trainer.train_all_models(
            X_train, y_train, models_to_train
        )
        
        # Create ensemble if requested
        if config.get('create_ensemble', True):
            self.model_trainer.create_ensemble_model(X_train, y_train)
        
        # Evaluate models
        summary_df = self.model_trainer.evaluate_all_models(X_test, y_test)
        
        # Save models
        models_dir = self.model_trainer.save_models()
        
        # Generate model report
        report_path = self.model_trainer.generate_model_report()
        
        print(f"✓ Model training completed: {len(self.trained_models)} models trained")
        print(f"✓ Models saved to: {models_dir}")
        print(f"✓ Evaluation report: {report_path}")
        
        return self.trained_models
    
    def run_model_evaluation(self) -> dict:
        """
        Step 4: Comprehensive model evaluation and visualization.
        
        Returns:
            Dictionary of generated evaluation files
        """
        print("\n" + "="*60)
        print("STEP 4: COMPREHENSIVE EVALUATION")
        print("="*60)
        
        if self.processed_data is None or self.trained_models is None:
            raise ValueError("Models not trained. Run training first.")
        
        X_train, X_test, y_train, y_test = self.processed_data
        
        # Initialize evaluator
        self.model_evaluator = FraudModelEvaluator()
        
        # Load trained models
        models_dir = f"/home/hduser/projects/real_time_fraud_detection/data/models/"
        
        # Find latest model directory
        import glob
        model_dirs = glob.glob(f"{models_dir}trained_models_*")
        if model_dirs:
            latest_model_dir = max(model_dirs, key=os.path.getctime)
            self.model_evaluator.load_models(latest_model_dir)
        else:
            # Use models from trainer
            self.model_evaluator.models = self.model_trainer.models
        
        # Generate comprehensive evaluation
        config = self.config['evaluation']
        generated_files = {}
        
        if config.get('generate_visualizations', True):
            generated_files.update(
                self.model_evaluator.generate_evaluation_summary(
                    X_test, y_test, X_test.columns.tolist()
                )
            )
        
        if config.get('business_analysis', True):
            business_df = self.model_evaluator.analyze_business_impact(X_test, y_test)
        
        if config.get('threshold_optimization', True):
            # Optimize thresholds for key models
            for model_name in ['xgboost', 'random_forest', 'ensemble']:
                if model_name in self.model_evaluator.models:
                    threshold_results = self.model_evaluator.optimize_threshold(
                        model_name, X_test, y_test, metric='f1'
                    )
        
        self.evaluation_results = generated_files
        
        print(f"✓ Comprehensive evaluation completed")
        for file_type, file_path in generated_files.items():
            if file_path:
                print(f"✓ {file_type.replace('_', ' ').title()}: {file_path}")
        
        return generated_files
    
    def run_complete_pipeline(self) -> dict:
        """
        Execute the complete fraud detection pipeline.
        
        Returns:
            Dictionary with results from each step
        """
        print("\n" + "🚀 STARTING FRAUD DETECTION PIPELINE 🚀")
        print("="*80)
        
        start_time = datetime.now()
        
        results = {}
        
        try:
            # Step 1: Data Generation
            results['raw_data_path'] = self.run_data_generation()
            
            # Step 2: Data Preprocessing  
            results['processed_data'] = self.run_data_preprocessing()
            
            # Step 3: Model Training
            results['trained_models'] = self.run_model_training()
            
            # Step 4: Model Evaluation
            results['evaluation_files'] = self.run_model_evaluation()
            
            # Pipeline completion
            end_time = datetime.now()
            duration = end_time - start_time
            
            print("\n" + "🎉 PIPELINE COMPLETED SUCCESSFULLY! 🎉")
            print("="*80)
            print(f"Total execution time: {duration}")
            print(f"Timestamp: {self.timestamp}")
            
            # Summary of generated artifacts
            print(f"\n📊 Generated Artifacts:")
            print(f"├── Raw Dataset: {results['raw_data_path']}")
            print(f"├── Processed Data: /data/processed/*_{self.timestamp}.csv")
            print(f"├── Trained Models: /data/models/trained_models_{self.timestamp}/")
            print(f"├── Evaluation Report: /docs/model_evaluation_report_{self.timestamp}.md")
            print(f"└── Visualizations: /docs/*_{self.timestamp}.*")
            
            results['execution_time'] = duration
            results['timestamp'] = self.timestamp
            results['status'] = 'success'
            
        except Exception as e:
            print(f"\n❌ PIPELINE FAILED: {str(e)}")
            results['status'] = 'failed'
            results['error'] = str(e)
            raise
        
        return results


def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(description='Real-Time Fraud Detection Pipeline')
    parser.add_argument('--step', choices=['all', 'data', 'preprocess', 'train', 'evaluate'], 
                       default='all', help='Pipeline step to run')
    parser.add_argument('--num-users', type=int, default=5000, 
                       help='Number of users for synthetic data')
    parser.add_argument('--days', type=int, default=90, 
                       help='Number of days of transaction history')
    parser.add_argument('--fraud-rate', type=float, default=0.025, 
                       help='Fraud rate in synthetic data')
    parser.add_argument('--models', nargs='+', 
                       default=['logistic_regression', 'random_forest', 'xgboost', 'lightgbm'],
                       help='Models to train')
    
    args = parser.parse_args()
    
    # Create configuration from arguments
    config = {
        'data_generation': {
            'num_users': args.num_users,
            'days': args.days,
            'fraud_rate': args.fraud_rate
        },
        'preprocessing': {
            'test_size': 0.2,
            'random_state': 42
        },
        'modeling': {
            'models_to_train': args.models,
            'use_sampling': True,
            'create_ensemble': True
        },
        'evaluation': {
            'generate_visualizations': True,
            'business_analysis': True,
            'threshold_optimization': True
        }
    }
    
    # Initialize pipeline
    pipeline = FraudDetectionPipeline(config)
    
    # Run specified step(s)
    if args.step == 'all':
        results = pipeline.run_complete_pipeline()
    elif args.step == 'data':
        results = pipeline.run_data_generation()
    elif args.step == 'preprocess':
        results = pipeline.run_data_preprocessing()
    elif args.step == 'train':
        results = pipeline.run_model_training()
    elif args.step == 'evaluate':
        results = pipeline.run_model_evaluation()
    
    return results


if __name__ == "__main__":
    results = main()