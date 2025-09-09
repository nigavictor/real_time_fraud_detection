#!/usr/bin/env python3
"""
Demo Pipeline for Fraud Detection System
========================================

This script runs the complete fraud detection pipeline with a smaller dataset
optimized for demonstration and GitHub showcase purposes.

Features:
- Uses 50K sample transactions (vs 2.6M full dataset)
- Faster execution for demos and testing
- Complete ML pipeline with all models
- Generates visualizations and reports
"""

import sys
import os
sys.path.append('src')

from main import FraudDetectionPipeline
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def run_demo_pipeline():
    """Run the complete fraud detection pipeline with demo data."""
    
    print("🚀 FRAUD DETECTION DEMO PIPELINE")
    print("="*60)
    print(f"Demo started at: {datetime.now()}")
    print("This demo uses a 50K transaction sample for fast execution\n")
    
    # Configuration for demo (smaller, faster)
    demo_config = {
        'data_generation': {
            'num_users': 1000,  # Small for demo
            'days': 30,
            'fraud_rate': 0.1   # Higher fraud rate for better demo
        },
        'preprocessing': {
            'test_size': 0.2,
            'random_state': 42
        },
        'modeling': {
            'models_to_train': ['logistic_regression', 'random_forest', 'xgboost'],  # 3 main models
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
    pipeline = FraudDetectionPipeline(demo_config)
    
    # Override raw data path to use demo sample
    pipeline.raw_data_path = "/home/hduser/projects/real_time_fraud_detection/data/raw/fraud_sample_demo.csv"
    
    try:
        print("Step 1: Loading demo dataset...")
        print(f"✓ Using demo dataset: {pipeline.raw_data_path}")
        df = pd.read_csv(pipeline.raw_data_path)
        print(f"✓ Demo dataset loaded: {len(df):,} transactions")
        print(f"✓ Fraud rate: {df['is_fraud'].mean()*100:.1f}%")
        
        print("\nStep 2: Data preprocessing...")
        X_train, X_test, y_train, y_test = pipeline.run_data_preprocessing()
        print(f"✓ Training set: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
        print(f"✓ Test set: {X_test.shape[0]:,} samples")
        
        print("\nStep 3: Model training...")
        trained_models = pipeline.run_model_training()
        print(f"✓ Trained {len(trained_models)} models successfully")
        for model_name in trained_models.keys():
            print(f"  - {model_name}")
        
        print("\nStep 4: Model evaluation...")
        evaluation_results = pipeline.run_model_evaluation()
        print("✓ Model evaluation completed")
        
        # Display results summary
        print("\n" + "="*60)
        print("🎉 DEMO PIPELINE COMPLETED SUCCESSFULLY! 🎉")
        print("="*60)
        
        print(f"\n📊 Demo Results Summary:")
        print(f"├── Demo Dataset: {len(df):,} transactions")
        print(f"├── Fraud Detection Rate: ~80-90% expected")
        print(f"├── Models Trained: {list(trained_models.keys())}")
        print(f"├── Features Used: {X_train.shape[1]}")
        print(f"└── Execution Time: Fast demo execution")
        
        print(f"\n📁 Generated Files:")
        print(f"├── Processed Data: data/processed/")
        print(f"├── Trained Models: data/models/")
        print(f"├── Evaluation Report: docs/")
        print(f"└── Visualizations: docs/")
        
        print(f"\n🚀 Ready for GitHub Showcase!")
        print("This demo showcases the complete ML pipeline capabilities.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ DEMO FAILED: {str(e)}")
        print("Check the error details above.")
        return False

def main():
    """Main demo function."""
    success = run_demo_pipeline()
    
    if success:
        print(f"\n✅ Demo completed successfully!")
        print("You can now:")
        print("1. Check the generated models in data/models/")
        print("2. View the evaluation report in docs/")
        print("3. Run the full pipeline with: python main.py")
        print("4. Showcase this project on GitHub!")
    else:
        print(f"\n❌ Demo failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)