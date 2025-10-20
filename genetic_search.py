
import os
import json
import argparse
from random import randint
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

import xgboost as xgb


from tabpfn import TabPFNClassifier


from data_loader_preprocess import (
    get_renovate_data_filtered,
    prepare_data_for_tabpfn,
)


# ==================== Model Configurations ====================

MODELS_REQUIRE_NORMALIZATION = {
    "logistic_regression": True,
    "svm": True,
    "decision_tree": False,
    "xgboost": False,
    "gradient_boosting": False,
    "gaussian_nb": True,
    "tabpfn": False
}


def create_base_model(model_name: str, random_state: int = 42):

    
    if model_name == "logistic_regression":
        return LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=random_state,
            solver='liblinear'
        )
    
    elif model_name == "svm":
        return SVC(
            class_weight='balanced',
            kernel='rbf',
            probability=True,
            random_state=random_state
        )
    
    elif model_name == "decision_tree":
        return DecisionTreeClassifier(
            class_weight='balanced',
            random_state=random_state
        )
    
    elif model_name == "xgboost":
        return xgb.XGBClassifier(
            scale_pos_weight=4.0, 
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    
    elif model_name == "gradient_boosting":
        return GradientBoostingClassifier(
            random_state=random_state
        )
    
    elif model_name == "gaussian_nb":
        return GaussianNB()
    
    elif model_name == "tabpfn":
        return TabPFNClassifier(
                device='cuda',
                balance_probabilities=True
            )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def initialization_of_population(size: int, n_feat: int) -> List[np.ndarray]:
    
    population = []
    for i in range(size):
        chromosome = np.ones(n_feat, dtype=bool)
        chromosome[:int(0.5 * n_feat)] = False
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population


def evaluate_feature_subset(
    chromosome: np.ndarray,
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_splits: int = 5,
    n_repeats: int = 2,
    random_state: int = 42
) -> Tuple[float, float, float, float, Dict[str, float]]:
    
    indices = np.where(chromosome)[0]
    
    if len(indices) < 2:
        return 0.0, 0.0, 0.0, 0.0, {}
    
    requires_normalization = MODELS_REQUIRE_NORMALIZATION.get(model_name, False)
   
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state
    )
    
    fold_metrics = {
        'balanced_accuracy': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'specificity': [],
        'f1': [],
        'auroc': [],
        'auprc': [],
        'train_accuracy': []
    }
    
    try:
        for train_idx, val_idx in rskf.split(X, y):
            X_train_fold, X_val_fold = X[train_idx][:, indices], X[val_idx][:, indices]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            if requires_normalization:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_fold)
                X_val_scaled = scaler.transform(X_val_fold)
            else:
                X_train_scaled = X_train_fold
                X_val_scaled = X_val_fold
            
            model = create_base_model(model_name, random_state=random_state)
            model.fit(X_train_scaled, y_train_fold)
            y_pred = model.predict(X_val_scaled)
            y_train_pred = model.predict(X_train_scaled)
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
            fold_metrics['train_accuracy'].append(
                accuracy_score(y_train_fold, y_train_pred)
            )
            
            tn, fp, fn, tp = confusion_matrix(y_val_fold, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            fold_metrics['specificity'].append(specificity)
            if len(np.unique(y_val_fold)) > 1:
                fold_metrics['auroc'].append(roc_auc_score(y_val_fold, y_prob))
                fold_metrics['auprc'].append(average_precision_score(y_val_fold, y_prob))
            else:
                fold_metrics['auroc'].append(np.nan)
                fold_metrics['auprc'].append(np.nan)
        mean_metrics = {}
        for metric_name, values in fold_metrics.items():
            valid_values = [v for v in values if not np.isnan(v)]
            if valid_values:
                mean_metrics[metric_name] = np.mean(valid_values)
                mean_metrics[f'{metric_name}_std'] = np.std(valid_values)
            else:
                mean_metrics[metric_name] = 0.0
                mean_metrics[f'{metric_name}_std'] = 0.0
        
        return (
            mean_metrics['balanced_accuracy'],
            mean_metrics['accuracy'],
            mean_metrics['recall'],
            mean_metrics['train_accuracy'],
            mean_metrics
        )
    
    except Exception as e:
        print(f"Error evaluating chromosome: {e}")
        return 0.0, 0.0, 0.0, 0.0, {}


def fitness_score(
    population: List[np.ndarray],
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_splits: int = 5,
    n_repeats: int = 2
) -> Tuple[List[float], List[np.ndarray], pd.DataFrame]:
    scores = []
    records = []
    
    for chromosome in population:
        balanced_acc, acc, recall, train_acc, all_metrics = evaluate_feature_subset(
            chromosome=chromosome,
            model_name=model_name,
            X=X,
            y=y,
            feature_names=feature_names,
            n_splits=n_splits,
            n_repeats=n_repeats
        )
        
        scores.append(balanced_acc)
        
        selected_indices = np.where(chromosome)[0]
        selected_features = [feature_names[i] for i in selected_indices]

        record = {
            'Model': model_name,
            'Features': str(selected_features),
            'N_Features': len(selected_features),
            'Balanced_Accuracy': balanced_acc,
            'Accuracy': acc,
            'Recall': recall,
            'Specificity': all_metrics.get('specificity', 0.0),
            'Precision': all_metrics.get('precision', 0.0),
            'F1': all_metrics.get('f1', 0.0),
            'AUROC': all_metrics.get('auroc', 0.0),
            'AUPRC': all_metrics.get('auprc', 0.0),
            'Train_Accuracy': train_acc
        }
        records.append(record)
    
    scores = np.array(scores)
    population = np.array(population)
    inds = np.argsort(scores)[::-1]  
    df = pd.DataFrame(records)
    df = df.iloc[inds].reset_index(drop=True)
    
    return list(scores[inds]), list(population[inds]), df


def selection(pop_after_fit: List[np.ndarray], n_parents: int) -> List[np.ndarray]:
    return pop_after_fit[:n_parents]


def crossover(pop_after_sel: List[np.ndarray]) -> List[np.ndarray]:
    pop_nextgen = list(pop_after_sel)
    
    for i in range(0, len(pop_after_sel) - 1, 2):
        child_1, child_2 = pop_after_sel[i], pop_after_sel[i + 1]
        crossover_point = len(child_1) // 2
        new_child = np.concatenate((child_1[:crossover_point], child_2[crossover_point:]))
        pop_nextgen.append(new_child)
    
    return pop_nextgen


def mutation(
    pop_after_cross: List[np.ndarray],
    mutation_rate: float,
    n_feat: int
) -> List[np.ndarray]:
    pop_next_gen = []
    
    for chromo in pop_after_cross:
        mutated_chromo = chromo.copy()
        mutation_range = max(1, int(mutation_rate * n_feat))
        rand_positions = [randint(0, n_feat - 1) for _ in range(mutation_range)]
        
        for pos in rand_positions:
            mutated_chromo[pos] = not mutated_chromo[pos]
        
        if sum(mutated_chromo) >= 2:
            pop_next_gen.append(mutated_chromo)
    
    return pop_next_gen


def run_genetic_algorithm(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    population_size: int = 200,
    n_parents: int = 120,
    mutation_rate: float = 0.25,
    n_generations: int = 20,
    n_splits: int = 5,
    n_repeats: int = 20,
    save_path: str = "./ga_results",
    random_state: int = 42
) -> Tuple[List[np.ndarray], List[float], pd.DataFrame]:
    
    print(f"\n{'='*80}")
    print(f"GENETIC ALGORITHM: {model_name.upper()}")
    print(f"{'='*80}")
    print(f"Population size: {population_size}")
    print(f"Number of parents: {n_parents}")
    print(f"Mutation rate: {mutation_rate}")
    print(f"Generations: {n_generations}")
    print(f"CV: {n_repeats} x {n_splits}-fold")
    
    model_save_path = os.path.join(save_path, model_name)
    os.makedirs(model_save_path, exist_ok=True)
    
    n_feat = X.shape[1]
    population_nextgen = initialization_of_population(population_size, n_feat)
    
    best_chromosomes = []
    best_scores = []
    all_generation_results = []
    
    for gen in range(n_generations):
        print(f"\nGeneration {gen + 1}/{n_generations}")
        
        scores, pop_after_fit, df = fitness_score(
            population=population_nextgen,
            model_name=model_name,
            X=X,
            y=y,
            feature_names=feature_names,
            n_splits=n_splits,
            n_repeats=n_repeats
        )
        
        print(f"  Best balanced accuracy: {scores[0]:.4f}")
        print(f"  Best features ({df.iloc[0]['N_Features']}): {df.iloc[0]['Features'][:100]}...")
    
        pop_after_sel = selection(pop_after_fit, n_parents)
    
        pop_after_cross = crossover(pop_after_sel)
        
        population_nextgen = mutation(pop_after_cross, mutation_rate, n_feat)
        
        best_chromosomes.append(pop_after_fit[0])
        best_scores.append(scores[0])
        
        df['Generation'] = gen + 1
        all_generation_results.append(df)
        
        gen_file = os.path.join(model_save_path, f'generation_{gen + 1:02d}.csv')
        df.to_csv(gen_file, index=False)
        print(f"  Saved: {gen_file}")

    all_results_df = pd.concat(all_generation_results, ignore_index=True)
    
    final_file = os.path.join(model_save_path, 'all_generations.csv')
    all_results_df.to_csv(final_file, index=False)
    
    best_idx = np.argmax(best_scores)
    best_chromosome = best_chromosomes[best_idx]
    selected_features = [feature_names[i] for i in np.where(best_chromosome)[0]]
    
    best_results = {
        'model': model_name,
        'best_generation': int(best_idx + 1),
        'best_balanced_accuracy': float(best_scores[best_idx]),
        'n_features': len(selected_features),
        'selected_features': selected_features,
        'chromosome': best_chromosome.tolist()
    }
    
    with open(os.path.join(model_save_path, 'best_results.json'), 'w') as f:
        json.dump(best_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS FOR {model_name.upper()}")
    print(f"{'='*80}")
    print(f"Best balanced accuracy: {best_scores[best_idx]:.4f}")
    print(f"Best generation: {best_idx + 1}")
    print(f"Number of features: {len(selected_features)}")
    print(f"Selected features: {selected_features}")
    
    return best_chromosomes, best_scores, all_results_df


def main():
    parser = argparse.ArgumentParser(
        description="Genetic Algorithm Feature Selection with Multiple Models"
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['logistic_regression', 'svm', 'decision_tree', 'xgboost', 
                 'gradient_boosting', 'gaussian_nb'],
        help='Models to use for feature selection'
    )
    parser.add_argument(
        '--population_size',
        type=int,
        default=100,
        help='Population size'
    )
    parser.add_argument(
        '--n_parents',
        type=int,
        default=60,
        help='Number of parents to select'
    )
    parser.add_argument(
        '--mutation_rate',
        type=float,
        default=0.25,
        help='Mutation rate (0-1)'
    )
    parser.add_argument(
        '--n_generations',
        type=int,
        default=20,
        help='Number of generations'
    )
    parser.add_argument(
        '--n_splits',
        type=int,
        default=5,
        help='Number of CV folds'
    )
    parser.add_argument(
        '--n_repeats',
        type=int,
        default=2,
        help='Number of CV repetitions'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='./ga_feature_selection_results',
        help='Path to save results'
    )
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=1,
        help='Number of parallel jobs (-1 for all cores)'
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    
    print("="*80)
    print("GENETIC ALGORITHM FEATURE SELECTION")
    print("="*80)
    print(f"Models: {args.models}")
    print(f"Population size: {args.population_size}")
    print(f"Generations: {args.n_generations}")
    print(f"CV: {args.n_repeats} x {args.n_splits}-fold")
    
    print("\nLoading RENOVATE training data...")
    df_train = get_renovate_data_filtered()
    X_train, y_train = prepare_data_for_tabpfn(df_train)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Features: {list(X_train.columns)}")
    print(f"Class distribution: {pd.Series(y_train).value_counts().to_dict()}")
    
    X_train_np = X_train.values
    y_train_np = y_train.values
    feature_names = list(X_train.columns)
    
    all_results = {}
    
    if args.n_jobs == 1:
        # Sequential execution
        for model_name in args.models:
            if model_name == 'tabpfn' and not TABPFN_AVAILABLE:
                print(f"\nSkipping {model_name}: TabPFN not available")
                continue
            
            try:
                best_chromo, best_score, results_df = run_genetic_algorithm(
                    model_name=model_name,
                    X=X_train_np,
                    y=y_train_np,
                    feature_names=feature_names,
                    population_size=args.population_size,
                    n_parents=args.n_parents,
                    mutation_rate=args.mutation_rate,
                    n_generations=args.n_generations,
                    n_splits=args.n_splits,
                    n_repeats=args.n_repeats,
                    save_path=args.save_path
                )
                
                all_results[model_name] = {
                    'best_chromosomes': best_chromo,
                    'best_scores': best_score,
                    'results_df': results_df
                }
            
            except Exception as e:
                print(f"\nError with {model_name}: {e}")
                import traceback
                traceback.print_exc()
    else:
        print(f"\nRunning in parallel with {args.n_jobs} jobs...")
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(run_genetic_algorithm)(
                model_name=model_name,
                X=X_train_np,
                y=y_train_np,
                feature_names=feature_names,
                population_size=args.population_size,
                n_parents=args.n_parents,
                mutation_rate=args.mutation_rate,
                n_generations=args.n_generations,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
                save_path=args.save_path
            )
            for model_name in args.models
            if model_name != 'tabpfn' or TABPFN_AVAILABLE
        )
        
        for model_name, (best_chromo, best_score, results_df) in zip(args.models, results):
            all_results[model_name] = {
                'best_chromosomes': best_chromo,
                'best_scores': best_score,
                'results_df': results_df
            }
    
    # Summary
    print("\n" + "="*80)
    print("FEATURE SELECTION SUMMARY")
    print("="*80)
    
    summary_data = []
    for model_name, results in all_results.items():
        best_idx = np.argmax(results['best_scores'])
        best_chromosome = results['best_chromosomes'][best_idx]
        selected_features = [feature_names[i] for i in np.where(best_chromosome)[0]]
        
        summary_data.append({
            'Model': model_name,
            'Best_Balanced_Accuracy': f"{results['best_scores'][best_idx]:.4f}",
            'N_Features': len(selected_features),
            'Features': str(selected_features)[:100] + '...'
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    summary_file = os.path.join(args.save_path, "feature_selection_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary saved to: {summary_file}")
    
    print("\n" + "="*80)
    print("FEATURE SELECTION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()