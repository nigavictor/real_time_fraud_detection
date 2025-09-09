"""
Quick Test Script for Fraud Detection Pipeline

This script runs a quick test of the complete pipeline with smaller data
to verify everything is working correctly.
"""

import sys
import os
sys.path.append('src')

from main import FraudDetectionPipeline
import pandas as pd
import numpy as np
from datetime import datetime

def run_quick_test():
    """Run a quick test of the fraud detection pipeline with minimal data."""
    
    print("🧪 Running Quick Pipeline Test")
    print("="*50)
    
    # Create test configuration with smaller dataset
    test_config = {
        'data_generation': {
            'num_users': 100,        # Small dataset for testing
            'days': 30,              # 30 days instead of 90
            'fraud_rate': 0.05       # Higher fraud rate for better testing
        },
        'preprocessing': {
            'test_size': 0.2,
            'random_state': 42
        },
        'modeling': {
            'models_to_train': ['logistic_regression', 'random_forest'],  # Only 2 models for speed
            'use_sampling': True,
            'create_ensemble': True
        },
        'evaluation': {
            'generate_visualizations': True,
            'business_analysis': True,
            'threshold_optimization': False  # Skip for speed
        }
    }
    
    # Initialize pipeline with test config
    pipeline = FraudDetectionPipeline(test_config)
    
    try:
        print("Step 1: Testing data generation...")
        raw_data_path = pipeline.run_data_generation()
        print(f"✓ Generated test dataset: {raw_data_path}")
        
        print("\nStep 2: Testing data preprocessing...")
        X_train, X_test, y_train, y_test = pipeline.run_data_preprocessing()
        print(f"✓ Preprocessing completed")
        print(f"  Training set: {X_train.shape}")
        print(f"  Test set: {X_test.shape}")
        print(f"  Features: {len(X_train.columns)}")
        
        print("\nStep 3: Testing model training...")
        trained_models = pipeline.run_model_training()
        print(f"✓ Trained {len(trained_models)} models")
        
        print("\nStep 4: Testing evaluation...")
        evaluation_files = pipeline.run_model_evaluation()
        print(f"✓ Generated {len(evaluation_files)} evaluation files")
        
        print("\n🎉 QUICK TEST PASSED! 🎉")
        print("="*50)
        print("All pipeline components are working correctly!")
        print(f"Test completed at: {datetime.now()}")
        
        # Display test results summary
        print(f"\nTest Results Summary:")
        print(f"├── Generated transactions: {len(pd.read_csv(raw_data_path)):,}")
        print(f"├── Fraud rate: {pd.read_csv(raw_data_path)['is_fraud'].mean()*100:.1f}%")
        print(f"├── Features created: {len(X_train.columns)}")
        print(f"├── Models trained: {list(trained_models.keys())}")
        print(f"└── Evaluation files: {len([f for f in evaluation_files.values() if f])}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        print("Please check the error above and fix any issues.")
        return False

def validate_installation():
    """Validate that all required packages are installed."""
    
    print("🔍 Validating Installation")
    print("="*50)
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'xgboost', 'lightgbm',
        'matplotlib', 'seaborn', 'plotly', 'faker', 'imbalanced-learn', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        print("Please install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print("\n✅ All required packages are installed!")
        return True

def check_project_structure():
    """Check if project structure is correct."""
    
    print("\n🏗️  Checking Project Structure")
    print("="*50)
    
    required_dirs = [
        'data/raw', 'data/processed', 'data/models',
        'src/data_generation', 'src/preprocessing', 'src/modeling', 'src/evaluation',
        'notebooks', 'docs', 'config'
    ]
    
    missing_dirs = []
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✓ {directory}/")
        else:
            print(f"✗ {directory}/ - MISSING")
            missing_dirs.append(directory)
    
    if missing_dirs:
        print(f"\n❌ Missing directories: {missing_dirs}")
        print("Creating missing directories...")
        for directory in missing_dirs:
            os.makedirs(directory, exist_ok=True)
            print(f"Created: {directory}/")
        print("✅ Project structure fixed!")
    else:
        print("\n✅ Project structure is correct!")
    
    return True

def main():
    """Main test function."""
    
    print("🚀 FRAUD DETECTION PIPELINE TEST SUITE")
    print("="*60)
    print(f"Test started at: {datetime.now()}")
    
    # Step 1: Validate installation
    if not validate_installation():
        return False
    
    # Step 2: Check project structure
    if not check_project_structure():
        return False
    
    # Step 3: Run quick pipeline test
    if not run_quick_test():
        return False
    
    print("\n" + "="*60)
    print("🎊 ALL TESTS PASSED! 🎊")
    print("="*60)
    print("Your fraud detection system is ready to use!")
    print("\nTo run the full pipeline:")
    print("python main.py")
    print("\nTo run with custom settings:")
    print("python main.py --num-users 10000 --days 180 --fraud-rate 0.025")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)