
import os
import json
import argparse
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

import xgboost as xgb

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

# Import data loader
from data_loader_preprocess import (
    get_renovate_data_filtered,
    prepare_data_for_tabpfn,
    calculate_pos_weight
)


MODELS_REQUIRE_NORMALIZATION = {
    "logistic_regression": True,
    "svm": True,
    "decision_tree": False,
    "xgboost": False,
    "gradient_boosting": False
}

MODEL_CONFIGS = {
    "logistic_regression": {
        "C": tune.loguniform(1e-4, 100),
        "penalty": tune.choice(["l1", "l2", "elasticnet"]),
        "class_weight": tune.choice([None, "balanced"]),
        "solver": "saga",  # saga supports all penalties
        "max_iter": 2000,
        "random_state": 42
    },
    
    "svm": {
        "C": tune.loguniform(1e-3, 100),
        "kernel": tune.choice(["linear", "rbf", "poly"]),
        "class_weight": tune.choice([None, "balanced"]),
        "max_iter": 2000,
        "random_state": 42,
        "probability": True 
    },
    
    "decision_tree": {
        "max_depth": tune.randint(3, 20),
        "min_samples_split": tune.randint(2, 20),
        "min_samples_leaf": tune.randint(1, 10),
        "max_features": tune.choice(["sqrt", "log2", None]),
        "criterion": tune.choice(["gini", "entropy"]),
        "class_weight": tune.choice([None, "balanced"]),
        "splitter": tune.choice(["best", "random"]),
        "random_state": 42
    },
    
    "xgboost": {
        "n_estimators": tune.randint(10, 100),
        "max_depth": tune.randint(3, 12),
        "learning_rate": tune.loguniform(1e-3, 0.3),
        "subsample": tune.uniform(0.6, 1.0),
        "colsample_bytree": tune.uniform(0.6, 1.0),
        "colsample_bylevel": tune.uniform(0.6, 1.0),
        "min_child_weight": tune.randint(1, 10),
        "gamma": tune.loguniform(1e-8, 1.0),
        "scale_pos_weight": tune.uniform(1, 10),
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "logloss"
    },
    
    "gradient_boosting": {
        "n_estimators": tune.randint(10, 100),
        "max_depth": tune.randint(3, 10),
        "learning_rate": tune.loguniform(1e-3, 0.3),
        "subsample": tune.uniform(0.6, 1.0),
        "min_samples_split": tune.randint(2, 20),
        "min_samples_leaf": tune.randint(1, 10),
        "max_features": tune.choice(["sqrt", "log2", None]),
        "loss": tune.choice(["log_loss", "exponential"]),
        "random_state": 42
    }
}


def create_model(model_name: str, config: Dict[str, Any]):
    """Create a model instance with given hyperparameters."""
    
    params = {k: v for k, v in config.items() if not isinstance(v, tune.sample.Domain)}
    
    if model_name == "logistic_regression":
        if params.get("penalty") == "elasticnet":
            params["l1_ratio"] = 0.5  
        return LogisticRegression(**params)
    
    elif model_name == "svm":
        return SVC(**params)
    
    elif model_name == "decision_tree":
        return DecisionTreeClassifier(**params)
    
    elif model_name == "xgboost":
        return xgb.XGBClassifier(**params)
    
    elif model_name == "gradient_boosting":
        return GradientBoostingClassifier(**params)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_cv_fold(
    model_name: str,
    config: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, float]:
  
    
    requires_normalization = MODELS_REQUIRE_NORMALIZATION.get(model_name, False)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_metrics = {
        'balanced_accuracy': [],
        'accuracy': [],
        'auroc': [],
        'auprc': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        if requires_normalization:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_val_scaled = scaler.transform(X_val_fold)
        else:
            X_train_scaled = X_train_fold
            X_val_scaled = X_val_fold
        
        model = create_model(model_name, config)
        
        try:
            model.fit(X_train_scaled, y_train_fold)
            
            y_pred = model.predict(X_val_scaled)
            
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_val_scaled)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_prob = model.decision_function(X_val_scaled)
            else:
                y_prob = y_pred
            
            fold_metrics['balanced_accuracy'].append(
                balanced_accuracy_score(y_val_fold, y_pred)
            )
            fold_metrics['accuracy'].append(
                accuracy_score(y_val_fold, y_pred)
            )
            fold_metrics['precision'].append(
                precision_score(y_val_fold, y_pred, zero_division=0)
            )
            fold_metrics['recall'].append(
                recall_score(y_val_fold, y_pred, zero_division=0)
            )
            fold_metrics['f1'].append(
                f1_score(y_val_fold, y_pred, zero_division=0)
            )
            
            if len(np.unique(y_val_fold)) > 1:
                fold_metrics['auroc'].append(roc_auc_score(y_val_fold, y_prob))
                fold_metrics['auprc'].append(average_precision_score(y_val_fold, y_prob))
            else:
                fold_metrics['auroc'].append(np.nan)
                fold_metrics['auprc'].append(np.nan)
        
        except Exception as e:
            print(f"Error in fold {fold_idx}: {e}")
            return {
                'mean_balanced_accuracy': 0.0,
                'std_balanced_accuracy': 0.0,
                'mean_accuracy': 0.0,
                'mean_auroc': 0.0,
                'mean_auprc': 0.0
            }
    
    results = {}
    for metric_name, values in fold_metrics.items():
        valid_values = [v for v in values if not np.isnan(v)]
        if valid_values:
            results[f'mean_{metric_name}'] = np.mean(valid_values)
            results[f'std_{metric_name}'] = np.std(valid_values)
        else:
            results[f'mean_{metric_name}'] = 0.0
            results[f'std_{metric_name}'] = 0.0
    
    return results


def trainable_function(config: Dict[str, Any], model_name: str = None, data: Dict = None):
    
    X = data['X']
    y = data['y']
    
    # Perform 5-fold CV
    cv_results = train_cv_fold(
        model_name=model_name,
        config=config,
        X=X,
        y=y,
        n_splits=5,
        random_state=42
    )
    
    tune.report(
        balanced_accuracy=cv_results['mean_balanced_accuracy'],
        balanced_accuracy_std=cv_results['std_balanced_accuracy'],
        accuracy=cv_results['mean_accuracy'],
        auroc=cv_results['mean_auroc'],
        auprc=cv_results['mean_auprc'],
        precision=cv_results['mean_precision'],
        recall=cv_results['mean_recall'],
        f1=cv_results['mean_f1']
    )



def optimize_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    num_samples: int = 50,
    max_concurrent: int = 4,
    grace_period: int = 3,
    output_dir: str = "./ray_tune_results"
) -> Dict[str, Any]:
   
    print(f"\n{'='*80}")
    print(f"OPTIMIZING: {model_name.upper()}")
    print(f"{'='*80}")
    
    # Get model configuration
    config = MODEL_CONFIGS[model_name]
    
    # Create ASHA scheduler
    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        max_t=5,  # 5-fold CV
        grace_period=grace_period,
        reduction_factor=3,
        brackets=1
    )
    
    # Prepare data for trainable function
    data_dict = {'X': X, 'y': y}
    
    # Run optimization
    analysis = tune.run(
        tune.with_parameters(
            trainable_function,
            model_name=model_name,
            data=data_dict
        ),
        config=config,
        scheduler=scheduler,
        num_samples=num_samples,
        max_concurrent_trials=max_concurrent,
        metric="balanced_accuracy",
        mode="max",
        verbose=1,
        resources_per_trial={"cpu": 2, "gpu": 0}, 
        name=f"{model_name}_optimization"
    )
    
    best_trial = analysis.get_best_trial(metric="balanced_accuracy", mode="max")
    
    print(f"\n{'='*80}")
    print(f"BEST RESULTS FOR {model_name.upper()}")
    print(f"{'='*80}")
    print(f"Best balanced accuracy: {best_trial.last_result['balanced_accuracy']:.4f} "
          f"(±{best_trial.last_result['balanced_accuracy_std']:.4f})")
    print(f"Best hyperparameters:")
    for param, value in best_trial.config.items():
        print(f"  {param}: {value}")
    
    results = {
        'model_name': model_name,
        'best_config': best_trial.config,
        'best_metrics': {
            'balanced_accuracy': best_trial.last_result['balanced_accuracy'],
            'balanced_accuracy_std': best_trial.last_result['balanced_accuracy_std'],
            'accuracy': best_trial.last_result['accuracy'],
            'auroc': best_trial.last_result['auroc'],
            'auprc': best_trial.last_result['auprc'],
            'precision': best_trial.last_result['precision'],
            'recall': best_trial.last_result['recall'],
            'f1': best_trial.last_result['f1']
        },
        'all_trials': len(analysis.trials),
        'dataframe': analysis.dataframe()
    }
    
    return results


def retrain_best_model(
    model_name: str,
    best_config: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray = None,
    y_test: np.ndarray = None
) -> Dict[str, Any]:

    
    print(f"\n{'='*80}")
    print(f"RETRAINING {model_name.upper()} WITH BEST HYPERPARAMETERS")
    print(f"{'='*80}")
    
    requires_normalization = MODELS_REQUIRE_NORMALIZATION.get(model_name, False)
    
    if requires_normalization:
        print(f"Applying Z-score normalization (regression-based model)")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        print(f"No normalization applied (tree-based model)")
        scaler = None
        X_train_scaled = X_train
    
    model = create_model(model_name, best_config)
    model.fit(X_train_scaled, y_train)
    
    y_train_pred = model.predict(X_train_scaled)
    if hasattr(model, 'predict_proba'):
        y_train_prob = model.predict_proba(X_train_scaled)[:, 1]
    else:
        y_train_prob = y_train_pred
    
    train_metrics = {
        'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
        'accuracy': accuracy_score(y_train, y_train_pred),
        'auroc': roc_auc_score(y_train, y_train_prob),
        'auprc': average_precision_score(y_train, y_train_prob),
        'precision': precision_score(y_train, y_train_pred, zero_division=0),
        'recall': recall_score(y_train, y_train_pred, zero_division=0),
        'f1': f1_score(y_train, y_train_pred, zero_division=0)
    }
    
    print("\nTraining Metrics:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    test_metrics = None
    if X_test is not None and y_test is not None:
        if requires_normalization:
            X_test_scaled = scaler.transform(X_test)
        else:
            X_test_scaled = X_test
            
        y_test_pred = model.predict(X_test_scaled)
        
        if hasattr(model, 'predict_proba'):
            y_test_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_test_prob = y_test_pred
        
        test_metrics = {
            'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
            'accuracy': accuracy_score(y_test, y_test_pred),
            'auroc': roc_auc_score(y_test, y_test_prob) if len(np.unique(y_test)) > 1 else np.nan,
            'auprc': average_precision_score(y_test, y_test_prob) if len(np.unique(y_test)) > 1 else np.nan,
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'f1': f1_score(y_test, y_test_pred, zero_division=0)
        }
        
        print("\nTest Metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    return {
        'model': model,
        'scaler': scaler, 
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'best_config': best_config,
        'requires_normalization': requires_normalization
    }


def main():
    parser = argparse.ArgumentParser(
        description="Ray Tune Hyperparameter Optimization with Stratified 5-Fold CV"
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['logistic_regression', 'svm', 'decision_tree', 'xgboost', 'gradient_boosting'],
        help='Models to optimize (space-separated)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=50,
        help='Number of hyperparameter samples per model'
    )
    parser.add_argument(
        '--max_concurrent',
        type=int,
        default=4,
        help='Maximum concurrent trials'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./ray_tune_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--save_models',
        action='store_true',
        help='Save retrained models'
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not ray.is_initialized():
        ray.init(num_cpus=args.max_concurrent * 2, ignore_reinit_error=True)
    
    print("="*80)
    print("RAY TUNE HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    print(f"Models: {args.models}")
    print(f"Samples per model: {args.num_samples}")
    print(f"Max concurrent trials: {args.max_concurrent}")
    print(f"Output directory: {args.output_dir}")
    
    print("\nNormalization Strategy:")
    print("  Regression-based models (SVM, Logistic Regression): Z-score normalization")
    print("  Tree-based models (Decision Tree, XGBoost, Gradient Boosting): No normalization")
    
    print("\nLoading RENOVATE training data...")
    df_train = get_renovate_data_filtered()
    X_train, y_train = prepare_data_for_tabpfn(df_train)
    
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Class distribution: {pd.Series(y_train).value_counts().to_dict()}")

    X_train_np = X_train.values
    y_train_np = y_train.values
    
    all_results = {}
    
    for model_name in args.models:
        if model_name not in MODEL_CONFIGS:
            print(f"\nWarning: Unknown model '{model_name}', skipping...")
            continue
        
        try:
            results = optimize_model(
                model_name=model_name,
                X=X_train_np,
                y=y_train_np,
                num_samples=args.num_samples,
                max_concurrent=args.max_concurrent,
                output_dir=args.output_dir
            )
            
            all_results[model_name] = results
            
            results_file = os.path.join(
                args.output_dir,
                f"{model_name}_best_config.json"
            )
            with open(results_file, 'w') as f:
                json.dump({
                    'best_config': results['best_config'],
                    'best_metrics': results['best_metrics']
                }, f, indent=2)
            
            print(f"\nResults saved to: {results_file}")
            
            if args.save_models:
                retrained = retrain_best_model(
                    model_name=model_name,
                    best_config=results['best_config'],
                    X_train=X_train_np,
                    y_train=y_train_np
                )
  
                import joblib
                model_file = os.path.join(
                    args.output_dir,
                    f"{model_name}_best_model.joblib"
                )
                joblib.dump(retrained, model_file)
                print(f"Model saved to: {model_file}")
        
        except Exception as e:
            print(f"\nError optimizing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("OPTIMIZATION SUMMARY")
    print("="*80)
    
    summary_data = []
    for model_name, results in all_results.items():
        summary_data.append({
            'Model': model_name,
            'Balanced Accuracy': f"{results['best_metrics']['balanced_accuracy']:.4f}",
            'Accuracy': f"{results['best_metrics']['accuracy']:.4f}",
            'AUROC': f"{results['best_metrics']['auroc']:.4f}",
            'AUPRC': f"{results['best_metrics']['auprc']:.4f}",
            'F1': f"{results['best_metrics']['f1']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    summary_file = os.path.join(args.output_dir, "optimization_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary saved to: {summary_file}")
    
    ray.shutdown()
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()