"""
Model Training Module for Fraud Detection

This module implements multiple machine learning models for fraud detection,
including traditional ML models and ensemble methods.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, precision_recall_curve, roc_curve,
                           average_precision_score, f1_score)
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, List
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FraudModelTrainer:
    """
    Comprehensive fraud detection model trainer supporting multiple algorithms
    and evaluation strategies.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.model_configs = self._get_model_configs()
        self.evaluation_results = {}
        
    def _get_model_configs(self) -> Dict[str, Dict]:
        """Define model configurations."""
        return {
            'logistic_regression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'params': {
                    'C': 1.0,
                    'class_weight': 'balanced'
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': 200,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'class_weight': 'balanced',
                    'n_jobs': -1
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'scale_pos_weight': 10,  # Handle imbalanced data
                    'eval_metric': 'aucpr',
                    'use_label_encoder': False
                }
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'class_weight': 'balanced',
                    'verbose': -1
                }
            },
            'isolation_forest': {
                'model': IsolationForest(random_state=self.random_state),
                'params': {
                    'contamination': 0.025,  # Expected fraud rate
                    'n_estimators': 100,
                    'max_samples': 'auto'
                }
            }
        }
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                   use_sampling: bool = True) -> Any:
        """
        Train a specific model.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            use_sampling: Whether to apply SMOTE for balancing
            
        Returns:
            Trained model
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not supported. Available: {list(self.model_configs.keys())}")
        
        print(f"Training {model_name}...")
        
        config = self.model_configs[model_name]
        model = config['model']
        model.set_params(**config['params'])
        
        # Handle class imbalance with sampling (except for isolation forest)
        if use_sampling and model_name != 'isolation_forest':
            print("Applying SMOTE for class balancing...")
            
            # Use SMOTE + RandomUnderSampler pipeline
            smote = SMOTE(random_state=self.random_state, k_neighbors=min(5, sum(y_train) - 1))
            under_sampler = RandomUnderSampler(random_state=self.random_state, 
                                             sampling_strategy={0: int(sum(y_train) * 3)})
            
            # Create pipeline
            sampling_pipeline = ImbPipeline([
                ('over', smote),
                ('under', under_sampler)
            ])
            
            X_resampled, y_resampled = sampling_pipeline.fit_resample(X_train, y_train)
            
            print(f"Original distribution: {np.bincount(y_train)}")
            print(f"Resampled distribution: {np.bincount(y_resampled)}")
            
            # Train on resampled data
            model.fit(X_resampled, y_resampled)
        else:
            # Train on original data
            if model_name == 'isolation_forest':
                # Isolation Forest is unsupervised, only fit on normal transactions
                X_normal = X_train[y_train == 0]
                model.fit(X_normal)
            else:
                model.fit(X_train, y_train)
        
        self.models[model_name] = model
        print(f"{model_name} training completed!")
        
        return model
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        models_to_train: List[str] = None) -> Dict[str, Any]:
        """Train all or specified models."""
        
        if models_to_train is None:
            models_to_train = list(self.model_configs.keys())
        
        print(f"Training {len(models_to_train)} models...")
        print(f"Training data shape: {X_train.shape}")
        print(f"Fraud rate: {y_train.mean():.3f}")
        
        trained_models = {}
        
        for model_name in models_to_train:
            try:
                model = self.train_model(model_name, X_train, y_train)
                trained_models[model_name] = model
                print(f"✓ {model_name} completed")
            except Exception as e:
                print(f"✗ {model_name} failed: {str(e)}")
                continue
        
        print(f"\nTraining completed! {len(trained_models)}/{len(models_to_train)} models successful")
        
        return trained_models
    
    def evaluate_model(self, model_name: str, model: Any, 
                      X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            model_name: Name of the model
            model: Trained model
            X_test: Test features  
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"Evaluating {model_name}...")
        
        results = {'model_name': model_name}
        
        try:
            # Make predictions
            if model_name == 'isolation_forest':
                # Isolation forest returns -1 for outliers, 1 for inliers
                y_pred_raw = model.predict(X_test)
                y_pred = (y_pred_raw == -1).astype(int)  # Convert to 0/1
                y_pred_proba = model.decision_function(X_test)
                # Convert decision function to probabilities
                y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
                y_pred_proba = np.column_stack([1 - y_pred_proba, y_pred_proba])
            else:
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
            
            # Basic metrics
            results['accuracy'] = (y_pred == y_test).mean()
            results['f1_score'] = f1_score(y_test, y_pred)
            
            # ROC AUC
            if y_pred_proba.shape[1] == 2:
                results['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
                results['avg_precision'] = average_precision_score(y_test, y_pred_proba[:, 1])
            else:
                results['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                results['avg_precision'] = average_precision_score(y_test, y_pred_proba)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            results['confusion_matrix'] = cm.tolist()
            
            # Precision, Recall, F1 for each class
            report = classification_report(y_test, y_pred, output_dict=True)
            results['classification_report'] = report
            
            # Additional metrics
            tn, fp, fn, tp = cm.ravel()
            results['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            results['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            results['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            results['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # Business metrics
            results['fraud_detection_rate'] = tp / sum(y_test) if sum(y_test) > 0 else 0
            results['false_alarm_rate'] = fp / (len(y_test) - sum(y_test)) if (len(y_test) - sum(y_test)) > 0 else 0
            
            print(f"✓ {model_name} evaluation completed")
            print(f"  ROC AUC: {results['roc_auc']:.3f}")
            print(f"  F1 Score: {results['f1_score']:.3f}")
            print(f"  Precision: {results['precision']:.3f}")
            print(f"  Recall: {results['recall']:.3f}")
            
        except Exception as e:
            print(f"✗ {model_name} evaluation failed: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def evaluate_all_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """Evaluate all trained models."""
        
        print(f"Evaluating {len(self.models)} models...")
        print(f"Test data shape: {X_test.shape}")
        print(f"Test fraud rate: {y_test.mean():.3f}")
        
        evaluation_results = []
        
        for model_name, model in self.models.items():
            results = self.evaluate_model(model_name, model, X_test, y_test)
            evaluation_results.append(results)
        
        # Create summary DataFrame
        summary_metrics = []
        for results in evaluation_results:
            if 'error' not in results:
                summary_metrics.append({
                    'model': results['model_name'],
                    'roc_auc': results['roc_auc'],
                    'avg_precision': results['avg_precision'],
                    'f1_score': results['f1_score'],
                    'precision': results['precision'],
                    'recall': results['recall'],
                    'fraud_detection_rate': results['fraud_detection_rate'],
                    'false_alarm_rate': results['false_alarm_rate']
                })
        
        summary_df = pd.DataFrame(summary_metrics)
        
        # Store detailed results
        self.evaluation_results = evaluation_results
        
        print("\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)
        print(summary_df.round(3).to_string(index=False))
        
        return summary_df
    
    def cross_validate_model(self, model_name: str, X: pd.DataFrame, y: pd.Series,
                           cv_folds: int = 5) -> Dict[str, Any]:
        """Perform cross-validation for a specific model."""
        
        print(f"Cross-validating {model_name} with {cv_folds} folds...")
        
        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not supported")
        
        config = self.model_configs[model_name]
        model = config['model']
        model.set_params(**config['params'])
        
        # Skip isolation forest for CV (unsupervised)
        if model_name == 'isolation_forest':
            print("Skipping cross-validation for isolation forest (unsupervised)")
            return {'model_name': model_name, 'cv_scores': None}
        
        # Stratified K-Fold for imbalanced data
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Cross-validation scores
        scoring_metrics = ['roc_auc', 'f1', 'precision', 'recall']
        cv_results = {}
        
        for metric in scoring_metrics:
            scores = cross_val_score(model, X, y, cv=skf, scoring=metric, n_jobs=-1)
            cv_results[f'{metric}_scores'] = scores
            cv_results[f'{metric}_mean'] = scores.mean()
            cv_results[f'{metric}_std'] = scores.std()
        
        print(f"✓ Cross-validation completed for {model_name}")
        for metric in scoring_metrics:
            mean_score = cv_results[f'{metric}_mean']
            std_score = cv_results[f'{metric}_std']
            print(f"  {metric}: {mean_score:.3f} (±{std_score:.3f})")
        
        return cv_results
    
    def create_ensemble_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                            models_to_ensemble: List[str] = None) -> Any:
        """Create an ensemble model using voting."""
        
        from sklearn.ensemble import VotingClassifier
        
        if models_to_ensemble is None:
            # Use all models except isolation forest
            models_to_ensemble = [name for name in self.models.keys() 
                                if name != 'isolation_forest']
        
        print(f"Creating ensemble from {len(models_to_ensemble)} models...")
        
        # Create list of (name, model) tuples for VotingClassifier
        estimators = []
        for model_name in models_to_ensemble:
            if model_name in self.models:
                estimators.append((model_name, self.models[model_name]))
        
        if len(estimators) < 2:
            print("Not enough models for ensemble. Need at least 2.")
            return None
        
        # Create voting ensemble
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use probability voting
            n_jobs=-1
        )
        
        print("Training ensemble model...")
        ensemble.fit(X_train, y_train)
        
        self.models['ensemble'] = ensemble
        print("✓ Ensemble model created and trained")
        
        return ensemble
    
    def save_models(self, save_dir: str = None) -> str:
        """Save all trained models."""
        
        if save_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_dir = f"/home/hduser/projects/real_time_fraud_detection/data/models/trained_models_{timestamp}"
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Saving {len(self.models)} models to {save_dir}...")
        
        saved_models = {}
        
        for model_name, model in self.models.items():
            model_path = os.path.join(save_dir, f"{model_name}_model.joblib")
            joblib.dump(model, model_path)
            saved_models[model_name] = model_path
            print(f"✓ {model_name} saved")
        
        # Save model metadata
        metadata = {
            'models': list(self.models.keys()),
            'timestamp': datetime.now().isoformat(),
            'evaluation_results': self.evaluation_results
        }
        
        metadata_path = os.path.join(save_dir, "model_metadata.json")
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ All models saved to: {save_dir}")
        
        return save_dir
    
    def generate_model_report(self, save_path: str = None) -> str:
        """Generate a comprehensive model evaluation report."""
        
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = f"/home/hduser/projects/real_time_fraud_detection/docs/model_evaluation_report_{timestamp}.md"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print(f"Generating model evaluation report...")
        
        report = []
        report.append("# Fraud Detection Model Evaluation Report")
        report.append(f"\n**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n**Number of models evaluated:** {len(self.evaluation_results)}")
        
        report.append("\n## Model Performance Summary")
        
        # Create summary table
        summary_data = []
        for results in self.evaluation_results:
            if 'error' not in results:
                summary_data.append([
                    results['model_name'],
                    f"{results['roc_auc']:.3f}",
                    f"{results['f1_score']:.3f}",
                    f"{results['precision']:.3f}",
                    f"{results['recall']:.3f}",
                    f"{results['fraud_detection_rate']:.3f}"
                ])
        
        if summary_data:
            report.append("\n| Model | ROC AUC | F1 Score | Precision | Recall | Fraud Detection Rate |")
            report.append("|-------|---------|----------|-----------|--------|---------------------|")
            for row in summary_data:
                report.append(f"| {' | '.join(row)} |")
        
        # Detailed results for each model
        report.append("\n## Detailed Model Results")
        
        for results in self.evaluation_results:
            if 'error' not in results:
                model_name = results['model_name']
                report.append(f"\n### {model_name.replace('_', ' ').title()}")
                
                report.append(f"\n**Performance Metrics:**")
                report.append(f"- ROC AUC: {results['roc_auc']:.3f}")
                report.append(f"- Average Precision: {results['avg_precision']:.3f}")
                report.append(f"- F1 Score: {results['f1_score']:.3f}")
                report.append(f"- Precision: {results['precision']:.3f}")
                report.append(f"- Recall: {results['recall']:.3f}")
                report.append(f"- Fraud Detection Rate: {results['fraud_detection_rate']:.3f}")
                report.append(f"- False Alarm Rate: {results['false_alarm_rate']:.3f}")
                
                # Confusion matrix
                cm = results['confusion_matrix']
                report.append(f"\n**Confusion Matrix:**")
                report.append("```")
                report.append(f"                Predicted")
                report.append(f"Actual    Legit  Fraud")
                report.append(f"Legit     {cm[0][0]:5d}  {cm[0][1]:5d}")
                report.append(f"Fraud     {cm[1][0]:5d}  {cm[1][1]:5d}")
                report.append("```")
        
        # Best model recommendation
        if summary_data:
            best_roc_model = max(self.evaluation_results, key=lambda x: x.get('roc_auc', 0) if 'error' not in x else 0)
            best_f1_model = max(self.evaluation_results, key=lambda x: x.get('f1_score', 0) if 'error' not in x else 0)
            
            report.append("\n## Recommendations")
            report.append(f"\n**Best Overall Model (ROC AUC):** {best_roc_model['model_name']} (ROC AUC: {best_roc_model['roc_auc']:.3f})")
            report.append(f"**Best Balanced Model (F1):** {best_f1_model['model_name']} (F1: {best_f1_model['f1_score']:.3f})")
        
        # Write report to file
        with open(save_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"✓ Report saved to: {save_path}")
        
        return save_path


def main():
    """Main function to demonstrate model training pipeline."""
    
    # Initialize trainer
    trainer = FraudModelTrainer()
    
    # Load processed data
    processed_data_path = "/home/hduser/projects/real_time_fraud_detection/data/processed/"
    
    import glob
    
    # Find the most recent processed data files
    train_files = glob.glob(f"{processed_data_path}X_train_*.csv")
    if not train_files:
        print("No processed training data found!")
        print("Please run data_processor.py first")
        return
    
    latest_timestamp = max([f.split('_')[-1].replace('.csv', '') for f in train_files])
    
    # Load data
    X_train = pd.read_csv(f"{processed_data_path}X_train_{latest_timestamp}.csv")
    X_test = pd.read_csv(f"{processed_data_path}X_test_{latest_timestamp}.csv")
    y_train = pd.read_csv(f"{processed_data_path}y_train_{latest_timestamp}.csv").iloc[:, 0]
    y_test = pd.read_csv(f"{processed_data_path}y_test_{latest_timestamp}.csv").iloc[:, 0]
    
    print(f"Loaded data with timestamp: {latest_timestamp}")
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Train all models
    trained_models = trainer.train_all_models(X_train, y_train)
    
    # Create ensemble
    trainer.create_ensemble_model(X_train, y_train)
    
    # Evaluate all models
    summary_df = trainer.evaluate_all_models(X_test, y_test)
    
    # Save models
    models_dir = trainer.save_models()
    
    # Generate report
    report_path = trainer.generate_model_report()
    
    print("\n" + "="*60)
    print("MODEL TRAINING PIPELINE COMPLETED!")
    print("="*60)
    print(f"✓ Models saved to: {models_dir}")
    print(f"✓ Report saved to: {report_path}")
    print("\nBest performing models:")
    if not summary_df.empty:
        best_auc = summary_df.loc[summary_df['roc_auc'].idxmax()]
        best_f1 = summary_df.loc[summary_df['f1_score'].idxmax()]
        print(f"  Best ROC AUC: {best_auc['model']} ({best_auc['roc_auc']:.3f})")
        print(f"  Best F1 Score: {best_f1['model']} ({best_f1['f1_score']:.3f})")
    
    return trainer, summary_df


if __name__ == "__main__":
    trainer, summary_df = main()