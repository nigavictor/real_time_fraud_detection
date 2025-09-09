"""
Model Evaluation Module for Fraud Detection

This module provides comprehensive evaluation and visualization tools
for fraud detection models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score, classification_report
)
import joblib
import os
from typing import Dict, List, Any, Tuple
from datetime import datetime
import json

plt.style.use('default')
sns.set_palette("husl")

class FraudModelEvaluator:
    """
    Comprehensive evaluation suite for fraud detection models.
    
    Provides:
    - Model performance visualization
    - Feature importance analysis  
    - Business impact assessment
    - Threshold optimization
    """
    
    def __init__(self):
        self.models = {}
        self.evaluation_results = {}
        self.feature_names = None
        
    def load_models(self, models_dir: str):
        """Load trained models from directory."""
        print(f"Loading models from: {models_dir}")
        
        model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.joblib')]
        
        for model_file in model_files:
            model_name = model_file.replace('_model.joblib', '')
            model_path = os.path.join(models_dir, model_file)
            
            try:
                model = joblib.load(model_path)
                self.models[model_name] = model
                print(f"✓ Loaded {model_name}")
            except Exception as e:
                print(f"✗ Failed to load {model_name}: {str(e)}")
        
        print(f"Successfully loaded {len(self.models)} models")
        
    def plot_roc_curves(self, X_test: pd.DataFrame, y_test: pd.Series, 
                       save_path: str = None) -> str:
        """Plot ROC curves for all models."""
        
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.models)))
        
        for i, (model_name, model) in enumerate(self.models.items()):
            try:
                # Get predictions
                if model_name == 'isolation_forest':
                    y_scores = model.decision_function(X_test)
                    y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
                else:
                    y_proba = model.predict_proba(X_test)
                    y_scores = y_proba[:, 1]
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_scores)
                auc_score = roc_auc_score(y_test, y_scores)
                
                # Plot
                plt.plot(fpr, tpr, color=colors[i], linewidth=2, 
                        label=f'{model_name.replace("_", " ").title()} (AUC = {auc_score:.3f})')
                
            except Exception as e:
                print(f"Error plotting ROC for {model_name}: {str(e)}")
                continue
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Fraud Detection Models', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = f"/home/hduser/projects/real_time_fraud_detection/docs/roc_curves_{timestamp}.png"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ROC curves saved to: {save_path}")
        return save_path
    
    def plot_precision_recall_curves(self, X_test: pd.DataFrame, y_test: pd.Series,
                                   save_path: str = None) -> str:
        """Plot Precision-Recall curves for all models."""
        
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.models)))
        
        # Baseline (random classifier)
        baseline_precision = sum(y_test) / len(y_test)
        plt.axhline(y=baseline_precision, color='gray', linestyle='--', 
                   label=f'Baseline (AP = {baseline_precision:.3f})')
        
        for i, (model_name, model) in enumerate(self.models.items()):
            try:
                # Get predictions
                if model_name == 'isolation_forest':
                    y_scores = model.decision_function(X_test)
                    y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
                else:
                    y_proba = model.predict_proba(X_test)
                    y_scores = y_proba[:, 1]
                
                # Calculate PR curve
                precision, recall, _ = precision_recall_curve(y_test, y_scores)
                avg_precision = average_precision_score(y_test, y_scores)
                
                # Plot
                plt.plot(recall, precision, color=colors[i], linewidth=2,
                        label=f'{model_name.replace("_", " ").title()} (AP = {avg_precision:.3f})')
                
            except Exception as e:
                print(f"Error plotting PR curve for {model_name}: {str(e)}")
                continue
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves - Fraud Detection Models', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = f"/home/hduser/projects/real_time_fraud_detection/docs/pr_curves_{timestamp}.png"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Precision-Recall curves saved to: {save_path}")
        return save_path
    
    def plot_confusion_matrices(self, X_test: pd.DataFrame, y_test: pd.Series,
                               save_path: str = None) -> str:
        """Plot confusion matrices for all models."""
        
        n_models = len(self.models)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, (model_name, model) in enumerate(self.models.items()):
            try:
                # Get predictions
                if model_name == 'isolation_forest':
                    y_pred_raw = model.predict(X_test)
                    y_pred = (y_pred_raw == -1).astype(int)
                else:
                    y_pred = model.predict(X_test)
                
                # Calculate confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                
                # Plot heatmap
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Legit', 'Fraud'], 
                           yticklabels=['Legit', 'Fraud'],
                           ax=axes[i])
                
                axes[i].set_title(f'{model_name.replace("_", " ").title()}', fontweight='bold')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')
                
            except Exception as e:
                print(f"Error plotting confusion matrix for {model_name}: {str(e)}")
                axes[i].text(0.5, 0.5, f'Error: {model_name}', 
                           transform=axes[i].transAxes, ha='center', va='center')
                continue
        
        # Hide empty subplots
        for i in range(len(self.models), len(axes)):
            axes[i].set_visible(False)
        
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = f"/home/hduser/projects/real_time_fraud_detection/docs/confusion_matrices_{timestamp}.png"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrices saved to: {save_path}")
        return save_path
    
    def plot_feature_importance(self, feature_names: List[str] = None,
                               top_n: int = 20, save_path: str = None) -> str:
        """Plot feature importance for tree-based models."""
        
        # Filter models that have feature importance
        models_with_importance = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                models_with_importance[model_name] = model
            elif model_name == 'ensemble' and hasattr(model, 'estimators_'):
                # For ensemble models, try to get feature importance from base models
                try:
                    importances = []
                    for estimator in model.estimators_:
                        if hasattr(estimator[1], 'feature_importances_'):
                            importances.append(estimator[1].feature_importances_)
                    
                    if importances:
                        avg_importance = np.mean(importances, axis=0)
                        # Create a dummy object to store importance
                        class DummyModel:
                            def __init__(self, importance):
                                self.feature_importances_ = importance
                        models_with_importance[model_name] = DummyModel(avg_importance)
                except:
                    continue
        
        if not models_with_importance:
            print("No models with feature importance found")
            return None
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(list(models_with_importance.values())[0].feature_importances_))]
        
        n_models = len(models_with_importance)
        n_cols = min(2, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12*n_cols, 8*n_rows))
        
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1 and n_cols > 1:
            axes = axes
        else:
            axes = axes.flatten() if n_models > 1 else [axes]
        
        for i, (model_name, model) in enumerate(models_with_importance.items()):
            try:
                # Get feature importance
                importance = model.feature_importances_
                
                # Get top features
                top_indices = np.argsort(importance)[-top_n:][::-1]
                top_importance = importance[top_indices]
                top_features = [feature_names[idx] for idx in top_indices]
                
                # Plot horizontal bar chart
                y_pos = np.arange(len(top_features))
                axes[i].barh(y_pos, top_importance, color=plt.cm.viridis(np.linspace(0, 1, len(top_features))))
                axes[i].set_yticks(y_pos)
                axes[i].set_yticklabels(top_features)
                axes[i].set_xlabel('Importance')
                axes[i].set_title(f'{model_name.replace("_", " ").title()} - Top {len(top_features)} Features', 
                                 fontweight='bold')
                axes[i].grid(axis='x', alpha=0.3)
                
                # Invert y-axis to show most important at top
                axes[i].invert_yaxis()
                
            except Exception as e:
                print(f"Error plotting feature importance for {model_name}: {str(e)}")
                continue
        
        # Hide empty subplots
        for i in range(len(models_with_importance), len(axes)):
            if i < len(axes):
                axes[i].set_visible(False)
        
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = f"/home/hduser/projects/real_time_fraud_detection/docs/feature_importance_{timestamp}.png"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance plots saved to: {save_path}")
        return save_path
    
    def analyze_business_impact(self, X_test: pd.DataFrame, y_test: pd.Series,
                               avg_fraud_amount: float = 500.0,
                               investigation_cost: float = 50.0) -> pd.DataFrame:
        """Analyze the business impact of different models."""
        
        print("Analyzing business impact...")
        
        business_results = []
        
        for model_name, model in self.models.items():
            try:
                # Get predictions
                if model_name == 'isolation_forest':
                    y_pred_raw = model.predict(X_test)
                    y_pred = (y_pred_raw == -1).astype(int)
                else:
                    y_pred = model.predict(X_test)
                
                # Calculate confusion matrix components
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                
                # Business impact calculations
                total_fraud_cases = tp + fn
                total_legit_cases = tn + fp
                
                # Financial impact
                prevented_fraud_loss = tp * avg_fraud_amount
                missed_fraud_loss = fn * avg_fraud_amount
                investigation_costs = (tp + fp) * investigation_cost
                
                net_savings = prevented_fraud_loss - investigation_costs
                potential_loss = total_fraud_cases * avg_fraud_amount
                savings_rate = net_savings / potential_loss if potential_loss > 0 else 0
                
                # Operational metrics
                fraud_detection_rate = tp / total_fraud_cases if total_fraud_cases > 0 else 0
                false_positive_rate = fp / total_legit_cases if total_legit_cases > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                
                business_results.append({
                    'model': model_name,
                    'fraud_detection_rate': fraud_detection_rate,
                    'false_positive_rate': false_positive_rate,
                    'precision': precision,
                    'prevented_fraud_loss': prevented_fraud_loss,
                    'missed_fraud_loss': missed_fraud_loss,
                    'investigation_costs': investigation_costs,
                    'net_savings': net_savings,
                    'savings_rate': savings_rate,
                    'total_transactions_flagged': tp + fp
                })
                
            except Exception as e:
                print(f"Error analyzing business impact for {model_name}: {str(e)}")
                continue
        
        business_df = pd.DataFrame(business_results)
        
        print("\n" + "="*80)
        print("BUSINESS IMPACT ANALYSIS")
        print("="*80)
        print(f"Assumptions:")
        print(f"  - Average fraud amount: ${avg_fraud_amount:,.2f}")
        print(f"  - Investigation cost per alert: ${investigation_cost:,.2f}")
        print(f"  - Total test transactions: {len(y_test):,}")
        print(f"  - Total fraud cases: {sum(y_test):,}")
        
        if not business_df.empty:
            # Display results
            display_df = business_df.copy()
            financial_cols = ['prevented_fraud_loss', 'missed_fraud_loss', 'investigation_costs', 'net_savings']
            for col in financial_cols:
                display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}")
            
            display_df['fraud_detection_rate'] = display_df['fraud_detection_rate'].apply(lambda x: f"{x:.1%}")
            display_df['false_positive_rate'] = display_df['false_positive_rate'].apply(lambda x: f"{x:.1%}")
            display_df['precision'] = display_df['precision'].apply(lambda x: f"{x:.1%}")
            display_df['savings_rate'] = display_df['savings_rate'].apply(lambda x: f"{x:.1%}")
            
            print("\nModel Comparison:")
            print(display_df.to_string(index=False))
            
            # Recommendations
            best_savings = business_df.loc[business_df['net_savings'].idxmax()]
            best_detection = business_df.loc[business_df['fraud_detection_rate'].idxmax()]
            
            print(f"\nRecommendations:")
            print(f"  Best Financial Performance: {best_savings['model']} (${best_savings['net_savings']:,.2f} net savings)")
            print(f"  Best Fraud Detection: {best_detection['model']} ({best_detection['fraud_detection_rate']:.1%} detection rate)")
        
        return business_df
    
    def optimize_threshold(self, model_name: str, X_test: pd.DataFrame, y_test: pd.Series,
                          metric: str = 'f1') -> Dict[str, Any]:
        """Optimize the decision threshold for a given model and metric."""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        print(f"Optimizing threshold for {model_name} using {metric} metric...")
        
        model = self.models[model_name]
        
        try:
            # Get probability predictions
            if model_name == 'isolation_forest':
                y_scores = model.decision_function(X_test)
                y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
            else:
                y_proba = model.predict_proba(X_test)
                y_scores = y_proba[:, 1]
            
            # Try different thresholds
            thresholds = np.arange(0.01, 1.0, 0.01)
            scores = []
            
            for threshold in thresholds:
                y_pred_thresh = (y_scores >= threshold).astype(int)
                
                if metric == 'f1':
                    from sklearn.metrics import f1_score
                    score = f1_score(y_test, y_pred_thresh)
                elif metric == 'precision':
                    from sklearn.metrics import precision_score
                    score = precision_score(y_test, y_pred_thresh, zero_division=0)
                elif metric == 'recall':
                    from sklearn.metrics import recall_score
                    score = recall_score(y_test, y_pred_thresh)
                elif metric == 'balanced_accuracy':
                    from sklearn.metrics import balanced_accuracy_score
                    score = balanced_accuracy_score(y_test, y_pred_thresh)
                else:
                    raise ValueError(f"Metric {metric} not supported")
                
                scores.append(score)
            
            # Find optimal threshold
            optimal_idx = np.argmax(scores)
            optimal_threshold = thresholds[optimal_idx]
            optimal_score = scores[optimal_idx]
            
            # Get performance at optimal threshold
            y_pred_optimal = (y_scores >= optimal_threshold).astype(int)
            
            from sklearn.metrics import classification_report, confusion_matrix
            cm = confusion_matrix(y_test, y_pred_optimal)
            report = classification_report(y_test, y_pred_optimal, output_dict=True)
            
            results = {
                'model_name': model_name,
                'optimal_threshold': optimal_threshold,
                'optimal_score': optimal_score,
                'metric_optimized': metric,
                'confusion_matrix': cm.tolist(),
                'classification_report': report,
                'thresholds': thresholds.tolist(),
                'scores': scores
            }
            
            print(f"✓ Optimal threshold for {model_name}: {optimal_threshold:.3f}")
            print(f"  {metric.capitalize()} score: {optimal_score:.3f}")
            print(f"  Precision: {report['1']['precision']:.3f}")
            print(f"  Recall: {report['1']['recall']:.3f}")
            
            return results
            
        except Exception as e:
            print(f"Error optimizing threshold for {model_name}: {str(e)}")
            return None
    
    def create_interactive_dashboard(self, X_test: pd.DataFrame, y_test: pd.Series,
                                   save_path: str = None) -> str:
        """Create an interactive dashboard with model comparisons."""
        
        print("Creating interactive dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ROC Curves', 'Precision-Recall Curves', 
                           'Model Performance', 'Feature Importance'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        colors = px.colors.qualitative.Set3
        
        # ROC Curves
        for i, (model_name, model) in enumerate(self.models.items()):
            try:
                if model_name == 'isolation_forest':
                    y_scores = model.decision_function(X_test)
                    y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
                else:
                    y_proba = model.predict_proba(X_test)
                    y_scores = y_proba[:, 1]
                
                fpr, tpr, _ = roc_curve(y_test, y_scores)
                auc_score = roc_auc_score(y_test, y_scores)
                
                fig.add_trace(
                    go.Scatter(x=fpr, y=tpr, mode='lines',
                             name=f'{model_name} (AUC={auc_score:.3f})',
                             line=dict(color=colors[i % len(colors)], width=2)),
                    row=1, col=1
                )
                
                # PR Curves
                precision, recall, _ = precision_recall_curve(y_test, y_scores)
                avg_precision = average_precision_score(y_test, y_scores)
                
                fig.add_trace(
                    go.Scatter(x=recall, y=precision, mode='lines',
                             name=f'{model_name} (AP={avg_precision:.3f})',
                             line=dict(color=colors[i % len(colors)], width=2),
                             showlegend=False),
                    row=1, col=2
                )
                
            except Exception as e:
                print(f"Error adding {model_name} to dashboard: {str(e)}")
                continue
        
        # Add diagonal line to ROC plot
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                     line=dict(color='gray', dash='dash'),
                     name='Random Classifier', showlegend=False),
            row=1, col=1
        )
        
        # Model Performance Comparison
        performance_data = []
        for model_name, model in self.models.items():
            try:
                if model_name == 'isolation_forest':
                    y_pred_raw = model.predict(X_test)
                    y_pred = (y_pred_raw == -1).astype(int)
                    y_scores = model.decision_function(X_test)
                    y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
                else:
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)
                    y_scores = y_proba[:, 1]
                
                from sklearn.metrics import f1_score, precision_score, recall_score
                
                performance_data.append({
                    'model': model_name,
                    'roc_auc': roc_auc_score(y_test, y_scores),
                    'f1_score': f1_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred)
                })
                
            except Exception as e:
                print(f"Error calculating performance for {model_name}: {str(e)}")
                continue
        
        if performance_data:
            perf_df = pd.DataFrame(performance_data)
            
            # Add performance bars
            fig.add_trace(
                go.Bar(x=perf_df['model'], y=perf_df['roc_auc'],
                      name='ROC AUC', marker_color='lightblue'),
                row=2, col=1
            )
            fig.add_trace(
                go.Bar(x=perf_df['model'], y=perf_df['f1_score'],
                      name='F1 Score', marker_color='lightgreen'),
                row=2, col=1
            )
            fig.add_trace(
                go.Bar(x=perf_df['model'], y=perf_df['precision'],
                      name='Precision', marker_color='lightcoral'),
                row=2, col=1
            )
            fig.add_trace(
                go.Bar(x=perf_df['model'], y=perf_df['recall'],
                      name='Recall', marker_color='lightyellow'),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Fraud Detection Models - Interactive Dashboard",
            title_x=0.5,
            showlegend=True
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
        fig.update_xaxes(title_text="Recall", row=1, col=2)
        fig.update_yaxes(title_text="Precision", row=1, col=2)
        fig.update_xaxes(title_text="Model", row=2, col=1)
        fig.update_yaxes(title_text="Score", row=2, col=1)
        
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = f"/home/hduser/projects/real_time_fraud_detection/docs/interactive_dashboard_{timestamp}.html"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        
        print(f"Interactive dashboard saved to: {save_path}")
        return save_path
    
    def generate_evaluation_summary(self, X_test: pd.DataFrame, y_test: pd.Series,
                                   feature_names: List[str] = None) -> Dict[str, str]:
        """Generate comprehensive evaluation summary with all visualizations."""
        
        print("Generating comprehensive evaluation summary...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        generated_files = {}
        
        try:
            # Generate all plots
            generated_files['roc_curves'] = self.plot_roc_curves(X_test, y_test)
            generated_files['pr_curves'] = self.plot_precision_recall_curves(X_test, y_test)
            generated_files['confusion_matrices'] = self.plot_confusion_matrices(X_test, y_test)
            generated_files['feature_importance'] = self.plot_feature_importance(feature_names)
            generated_files['interactive_dashboard'] = self.create_interactive_dashboard(X_test, y_test)
            
            # Business impact analysis
            business_df = self.analyze_business_impact(X_test, y_test)
            
            print("\n" + "="*60)
            print("EVALUATION SUMMARY GENERATED!")
            print("="*60)
            
            for file_type, file_path in generated_files.items():
                if file_path:
                    print(f"✓ {file_type.replace('_', ' ').title()}: {file_path}")
            
        except Exception as e:
            print(f"Error generating evaluation summary: {str(e)}")
        
        return generated_files


def main():
    """Example usage of the model evaluator."""
    
    # Initialize evaluator
    evaluator = FraudModelEvaluator()
    
    # Load models (replace with actual path)
    models_dir = "/home/hduser/projects/real_time_fraud_detection/data/models/"
    
    # Find the most recent model directory
    import glob
    model_dirs = glob.glob(f"{models_dir}trained_models_*")
    if not model_dirs:
        print("No trained models found!")
        print("Please run model_trainer.py first")
        return
    
    latest_model_dir = max(model_dirs, key=os.path.getctime)
    print(f"Using models from: {latest_model_dir}")
    
    # Load models
    evaluator.load_models(latest_model_dir)
    
    # Load test data
    processed_data_path = "/home/hduser/projects/real_time_fraud_detection/data/processed/"
    
    # Find the most recent test data
    test_files = glob.glob(f"{processed_data_path}X_test_*.csv")
    if not test_files:
        print("No test data found!")
        return
    
    latest_timestamp = max([f.split('_')[-1].replace('.csv', '') for f in test_files])
    
    X_test = pd.read_csv(f"{processed_data_path}X_test_{latest_timestamp}.csv")
    y_test = pd.read_csv(f"{processed_data_path}y_test_{latest_timestamp}.csv").iloc[:, 0]
    
    print(f"Loaded test data: {X_test.shape}")
    
    # Generate comprehensive evaluation
    generated_files = evaluator.generate_evaluation_summary(X_test, y_test, X_test.columns.tolist())
    
    return evaluator, generated_files


if __name__ == "__main__":
    evaluator, generated_files = main()