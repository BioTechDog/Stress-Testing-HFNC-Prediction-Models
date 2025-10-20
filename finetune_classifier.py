
from functools import partial
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabpfn import TabPFNClassifier
from tabpfn.finetune_utils import clone_model_for_evaluation
from tabpfn.utils import meta_dataset_collator
from tabpfn.model_loading import save_fitted_tabpfn_model, load_fitted_tabpfn_model

# Import data loading functions
from data_loader_preprocess import (
    get_renovate_data_filtered, 
    get_hfno_all, 
    prepare_data_for_tabpfn
)


def metrics_calculation(y, y_pred):
    accuracy = accuracy_score(y, y_pred)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    # Calculate sensitivity and specificity
    sensitivity = tp / (tp + fn)  # True Positive Rate
    specificity = tn / (tn + fp)  # True Negative Rate

    # Calculate correctly predicted positives rate and correctly predicted negatives rate
    correctly_predicted_positives_rate = tp / (tp + fp)
    correctly_predicted_negatives_rate = tn / (tn + fn)

    # Print the results
    print(f"Accuracy: {accuracy:.3f}")
    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Sensitivity (True Positive Rate): {sensitivity:.3f}")
    print(f"Specificity (True Negative Rate): {specificity:.3f}")
    print(f"Correctly Predicted Positives Rate: {correctly_predicted_positives_rate:.3f}")
    print(f"Correctly Predicted Negatives Rate: {correctly_predicted_negatives_rate:.3f}")
    
    return {
        'accuracy': accuracy,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'correctly_predicted_positives_rate': correctly_predicted_positives_rate,
        'correctly_predicted_negatives_rate': correctly_predicted_negatives_rate
    }


def prepare_data(config: dict):
    print("--- 1. Data Preparation ---")
    train_dataset = get_renovate_data_filtered()
    print(f"Loaded RENOVATE training dataset: {len(train_dataset)} samples")
    
    test_dataset = get_hfno_all()
    print(f"Loaded HFNO test dataset: {len(test_dataset)} samples")
    
    X_train, y_train = prepare_data_for_tabpfn(train_dataset)
    print(f"Training data after preprocessing: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    
    X_test, y_test = prepare_data_for_tabpfn(test_dataset)
    print(f"Test data after preprocessing: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    print(f"Training class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution: {np.bincount(y_test)}")
    
    print("---------------------------\n")
    return X_train, X_test, y_train, y_test


def setup_model_and_optimizer(config: dict):
    classifier_config = {
        "ignore_pretraining_limits": True,
        "device": config["device"],
        "n_estimators": 2,
        "random_state": config["random_seed"],
        "inference_precision": torch.float32,
    }
    classifier = TabPFNClassifier(
        **classifier_config, fit_mode="batched", differentiable_input=False, balance_probabilities=True
    )
    classifier._initialize_model_variables()
    
    # Optimizer uses finetuning-specific learning rate
    optimizer = Adam(
        classifier.model_.parameters(), lr=config["finetuning"]["learning_rate"]
    )

    print(f"Using device: {config['device']}")
    print(f"Optimizer: Adam, Finetuning LR: {config['finetuning']['learning_rate']}")
    print("----------------------------------\n")
    return classifier, optimizer, classifier_config


def evaluate_model(
    classifier: TabPFNClassifier,
    eval_config: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    eval_classifier = clone_model_for_evaluation(
        classifier, eval_config, TabPFNClassifier
    )
    
    # Fit the evaluation classifier on training data
    eval_classifier.fit(X_train, y_train)
    
    # Make predictions
    try:
        y_pred = eval_classifier.predict(X_test)
        
        # Calculate metrics
        metrics = metrics_calculation(y_test, y_pred)
        
        print("Evaluation completed successfully.")
        
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        metrics = {}
    
    print("-----------------------\n")
    return metrics


def finetune_model(
    classifier: TabPFNClassifier,
    optimizer: Optimizer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict,
):
    
    # Prepare training datasets for fine-tuning
    splitter = partial(train_test_split, test_size=config["valid_set_ratio"])
    training_datasets = classifier.get_preprocessed_datasets(
        X_train, y_train, splitter, config["finetuning"]["batch_size"]
    )
    
    finetuning_dataloader = DataLoader(
        training_datasets,
        batch_size=config["finetuning"]["meta_batch_size"],
        collate_fn=meta_dataset_collator,
    )
    
    class_counts = np.bincount(y_train.astype(int))
    total_samples = len(y_train)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights).to(config["device"])
    
    print(f"Class distribution: {class_counts}")
    print(f"Class weights: {class_weights}")
    
    # Loss function with class weights for balanced training
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Fine-tuning loop
    for epoch in range(config["finetuning"]["epochs"]):
        progress_bar = tqdm(finetuning_dataloader, desc=f"Finetuning Epoch {epoch + 1}")
        
        for (
            X_train_batch,
            X_test_batch,
            y_train_batch,
            y_test_batch,
            cat_ixs,
            confs,
        ) in progress_bar:
            # Skip batch if splits don't have all classes
            if len(np.unique(y_train_batch)) != len(np.unique(y_test_batch)):
                continue

            optimizer.zero_grad()
            classifier.fit_from_preprocessed(
                X_train_batch, y_train_batch, cat_ixs, confs
            )
            predictions = classifier.forward(X_test_batch, return_logits=True)
            loss = loss_function(predictions, y_test_batch.to(config["device"]))
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    
    print("Fine-tuning completed.")
    print("------------------------\n")


def save_model(classifier: TabPFNClassifier, model_path: str = "fintuned_tab_hfnc.tabpfn_fit"):
    try:
        save_fitted_tabpfn_model(classifier, model_path)
        print(f"Model saved successfully to: {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
    print("-------------------\n")


def load_and_evaluate_model(
    model_path: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classifier_config: dict,
    eval_config: dict,
):
    print("--- 5. Loading and Evaluating Saved Model ---")
    try:
        loaded_classifier = load_fitted_tabpfn_model(model_path, TabPFNClassifier)
        print(f"Model loaded successfully from: {model_path}")
        eval_classifier = clone_model_for_evaluation(
            loaded_classifier, eval_config, TabPFNClassifier
        )
        
        eval_classifier.fit(X_train, y_train)
        
        print("--- Model Evaluation with Loaded Model ---")
        y_pred = eval_classifier.predict(X_test)
        metrics = metrics_calculation(y_test, y_pred)
        print("Evaluation completed successfully.")
        print("-----------------------\n")
        
        return metrics
        
    except Exception as e:
        print(f"Error loading or evaluating model: {e}")
        import traceback
        traceback.print_exc()
        return {}
    print("-------------------------------------------\n")


def main():
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "random_seed": 42,
        "valid_set_ratio": 0.3,
        "n_inference_context_samples": 500,
    }
    
    config["finetuning"] = {
        "epochs": 1000,
        "learning_rate": 1e-6,
        "meta_batch_size": 1,
        "batch_size": 400,
    }
    
    print("=== TabPFN Fine-tuning for HFNC Failure Prediction ===\n")
    
    X_train, X_test, y_train, y_test = prepare_data(config)
    
    classifier, optimizer, classifier_config = setup_model_and_optimizer(config)

    initial_eval_config = {
        **classifier_config,
        "inference_config": {
            "SUBSAMPLE_SAMPLES": config["n_inference_context_samples"]  # Use full context (200 samples)
        },
    }
    
    print("Initial Model Evaluation (Before Fine-tuning):")
    initial_metrics = evaluate_model(
        classifier, initial_eval_config, X_train, y_train, X_test, y_test
    )
    
    finetune_model(classifier, optimizer, X_train, y_train, config)

    finetuned_eval_config = {
        **classifier_config,
        "inference_config": {
            "SUBSAMPLE_SAMPLES": min(config["n_inference_context_samples"], 1000)  # Use up to 1000 samples (model is domain-adapted)
        },
    }
    
    print("Fine-tuned Model Evaluation:")
    finetuned_metrics = evaluate_model(
        classifier, finetuned_eval_config, X_train, y_train, X_test, y_test
    )
    model_path = "fintuned_tab_hfnc.tabpfn_fit"
    save_model(classifier, model_path)

    loaded_metrics = load_and_evaluate_model(
        model_path, X_train, y_train, X_test, y_test, classifier_config, finetuned_eval_config
    )
    
    print(f"Fine-tuned model saved as: {model_path}")
    print("The model can be loaded later using load_fitted_tabpfn_model()")


if __name__ == "__main__":
    main()
