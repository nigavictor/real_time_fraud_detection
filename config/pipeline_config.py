"""
Pipeline Configuration for Fraud Detection System

This module contains all configuration parameters for the fraud detection pipeline.
Modify these settings to customize the pipeline behavior.
"""

import os
from datetime import datetime

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
DOCS_DIR = os.path.join(BASE_DIR, 'docs')

# Data generation configuration
DATA_GENERATION_CONFIG = {
    'num_users': 5000,              # Number of unique users to generate
    'days': 90,                     # Number of days of transaction history
    'fraud_rate': 0.025,            # Percentage of fraudulent transactions (2.5%)
    'random_state': 42,             # For reproducible results
    
    # User profile settings
    'min_age': 18,
    'max_age': 80,
    'income_levels': ['low', 'medium', 'high'],
    
    # Transaction settings
    'merchant_categories': [
        'grocery', 'gas_station', 'restaurant', 'retail', 'online',
        'pharmacy', 'entertainment', 'travel', 'utilities', 'healthcare',
        'education', 'automotive', 'electronics', 'clothing', 'home_improvement'
    ],
    
    # Geographic locations for transactions
    'locations': [
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
}

# Data preprocessing configuration
PREPROCESSING_CONFIG = {
    'test_size': 0.2,               # Train/test split ratio
    'random_state': 42,             # For reproducible splits
    'validation_size': 0.1,         # Validation set size (from training data)
    
    # Feature engineering settings
    'time_windows': {
        'short': '1H',              # 1 hour window for velocity features
        'medium': '24H',            # 24 hour window for daily patterns
        'long': '7D'                # 7 day window for weekly patterns
    },
    
    # Scaling method
    'scaler_type': 'robust',        # 'standard', 'robust', 'minmax'
    
    # Feature selection
    'feature_selection': {
        'method': 'mutual_info',    # 'mutual_info', 'chi2', 'f_classif'
        'k_best': 25,               # Number of top features to select
        'threshold': 0.01           # Minimum score threshold
    },
    
    # Data validation thresholds
    'validation': {
        'max_amount_multiplier': 10,    # Cap amounts at 10x 99th percentile
        'min_transactions_per_user': 3, # Minimum transactions per user
        'max_missing_rate': 0.05        # Maximum allowed missing data rate
    }
}

# Model training configuration
MODELING_CONFIG = {
    'random_state': 42,
    'cv_folds': 5,                  # Cross-validation folds
    'scoring_metric': 'roc_auc',    # Primary metric for model selection
    
    # Models to train
    'models': {
        'logistic_regression': {
            'enabled': True,
            'params': {
                'C': 1.0,
                'class_weight': 'balanced',
                'max_iter': 1000,
                'solver': 'liblinear'
            },
            'hyperparameter_tuning': {
                'enabled': True,
                'param_grid': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2']
                }
            }
        },
        
        'random_forest': {
            'enabled': True,
            'params': {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'class_weight': 'balanced',
                'n_jobs': -1
            },
            'hyperparameter_tuning': {
                'enabled': True,
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [8, 10, 12],
                    'min_samples_split': [5, 10]
                }
            }
        },
        
        'xgboost': {
            'enabled': True,
            'params': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'scale_pos_weight': 10,
                'eval_metric': 'aucpr',
                'use_label_encoder': False
            },
            'hyperparameter_tuning': {
                'enabled': True,
                'param_grid': {
                    'n_estimators': [100, 200],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'scale_pos_weight': [5, 10, 20]
                }
            }
        },
        
        'lightgbm': {
            'enabled': True,
            'params': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'class_weight': 'balanced',
                'verbose': -1
            },
            'hyperparameter_tuning': {
                'enabled': True,
                'param_grid': {
                    'n_estimators': [100, 200],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'num_leaves': [31, 50, 70]
                }
            }
        },
        
        'isolation_forest': {
            'enabled': True,
            'params': {
                'contamination': 0.025,
                'n_estimators': 100,
                'max_samples': 'auto',
                'random_state': 42
            }
        }
    },
    
    # Ensemble configuration
    'ensemble': {
        'enabled': True,
        'voting': 'soft',           # 'hard' or 'soft'
        'models_to_include': ['random_forest', 'xgboost', 'lightgbm'],
        'weights': None             # Equal weights if None
    },
    
    # Class imbalance handling
    'sampling': {
        'enabled': True,
        'strategy': 'smote_undersampling',  # 'smote', 'undersampling', 'smote_undersampling'
        'smote_params': {
            'k_neighbors': 5,
            'sampling_strategy': 'auto'
        },
        'undersampling_params': {
            'sampling_strategy': 0.3    # Ratio after undersampling
        }
    }
}

# Model evaluation configuration
EVALUATION_CONFIG = {
    # Metrics to calculate
    'metrics': [
        'roc_auc', 'average_precision', 'f1_score',
        'precision', 'recall', 'accuracy', 'balanced_accuracy'
    ],
    
    # Visualization settings
    'plots': {
        'roc_curve': {'enabled': True, 'figsize': (10, 8)},
        'pr_curve': {'enabled': True, 'figsize': (10, 8)},
        'confusion_matrix': {'enabled': True, 'figsize': (8, 6)},
        'feature_importance': {'enabled': True, 'top_n': 20, 'figsize': (12, 8)},
        'threshold_analysis': {'enabled': True, 'figsize': (10, 6)}
    },
    
    # Business impact analysis
    'business_impact': {
        'enabled': True,
        'avg_fraud_amount': 500.0,      # Average amount of fraudulent transaction
        'investigation_cost': 50.0,      # Cost to investigate each flagged transaction
        'prevention_value': 0.8,         # Percentage of fraud amount recovered when caught
        'reputation_cost_multiplier': 2.0  # Reputation damage multiplier for missed fraud
    },
    
    # Threshold optimization
    'threshold_optimization': {
        'enabled': True,
        'metrics_to_optimize': ['f1_score', 'precision', 'recall'],
        'threshold_range': (0.01, 0.99),
        'n_thresholds': 99
    },
    
    # Model interpretation
    'interpretation': {
        'enabled': True,
        'shap_analysis': True,          # SHAP values for model interpretation
        'feature_interaction': True,   # Feature interaction analysis
        'partial_dependence': True     # Partial dependence plots
    }
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',                # DEBUG, INFO, WARNING, ERROR
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_handler': {
        'enabled': True,
        'filename': 'fraud_detection.log',
        'max_bytes': 10485760,      # 10MB
        'backup_count': 5
    },
    'console_handler': {
        'enabled': True
    }
}

# Performance monitoring configuration
MONITORING_CONFIG = {
    'enabled': True,
    'metrics_to_track': [
        'training_time', 'inference_time', 'memory_usage',
        'model_accuracy', 'data_drift', 'prediction_drift'
    ],
    'alert_thresholds': {
        'accuracy_drop': 0.05,      # Alert if accuracy drops by 5%
        'inference_time': 0.1,      # Alert if inference takes > 100ms
        'memory_usage': 0.8         # Alert if memory usage > 80%
    }
}

# Export configuration for easy access
CONFIG = {
    'data_generation': DATA_GENERATION_CONFIG,
    'preprocessing': PREPROCESSING_CONFIG,
    'modeling': MODELING_CONFIG,
    'evaluation': EVALUATION_CONFIG,
    'logging': LOGGING_CONFIG,
    'monitoring': MONITORING_CONFIG,
    'paths': {
        'base_dir': BASE_DIR,
        'data_dir': DATA_DIR,
        'models_dir': MODELS_DIR,
        'docs_dir': DOCS_DIR
    }
}

def get_config(section: str = None):
    """
    Get configuration for a specific section or entire config.
    
    Args:
        section: Configuration section name (optional)
        
    Returns:
        Configuration dictionary
    """
    if section is None:
        return CONFIG
    
    if section not in CONFIG:
        raise ValueError(f"Configuration section '{section}' not found. "
                        f"Available sections: {list(CONFIG.keys())}")
    
    return CONFIG[section]

def update_config(section: str, updates: dict):
    """
    Update configuration for a specific section.
    
    Args:
        section: Configuration section name
        updates: Dictionary of updates to apply
    """
    if section not in CONFIG:
        raise ValueError(f"Configuration section '{section}' not found")
    
    CONFIG[section].update(updates)

def save_config(filepath: str):
    """Save current configuration to file."""
    import json
    
    # Convert any non-serializable objects to strings
    serializable_config = {}
    for key, value in CONFIG.items():
        try:
            json.dumps(value)  # Test if serializable
            serializable_config[key] = value
        except TypeError:
            serializable_config[key] = str(value)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_config, f, indent=2)
    
    print(f"Configuration saved to: {filepath}")

def load_config(filepath: str):
    """Load configuration from file."""
    import json
    
    with open(filepath, 'r') as f:
        loaded_config = json.load(f)
    
    # Update global config
    CONFIG.update(loaded_config)
    print(f"Configuration loaded from: {filepath}")

if __name__ == "__main__":
    # Example usage
    print("Fraud Detection Pipeline Configuration")
    print("="*50)
    
    print(f"Data Generation: {DATA_GENERATION_CONFIG['num_users']} users, "
          f"{DATA_GENERATION_CONFIG['days']} days")
    
    print(f"Models enabled: {[name for name, config in MODELING_CONFIG['models'].items() if config['enabled']]}")
    
    print(f"Evaluation metrics: {EVALUATION_CONFIG['metrics']}")
    
    # Save example config
    config_path = os.path.join(BASE_DIR, 'config', 'example_config.json')
    save_config(config_path)