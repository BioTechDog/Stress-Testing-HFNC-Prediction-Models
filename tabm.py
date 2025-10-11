import argparse
import os
import math
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt


DATA_IMPORT_ERROR = (
    "Could not import get_renovate_data_filtered() / get_hfno_all().\n"
    "Please update the import block in finetune_classifier_tabm.py to point to your project.\n"
)

from data_loader import (
    get_renovate_data_filtered, 
    get_hfno_all, 
    prepare_data_for_tabpfn
)



Features = ['HR_T0 (bpm)', 
'SpO2_T0 (%)', 'HR_T1 (bpm)', 'RR_T1 (bpm)', 'SpO2_T1 (%)', 'PaO2_T1 (mmHg)', 'FiO2_diff', 'SpO2_diff', 'RR_diff', 'HR_diff']

Features_aterial = ['PaO2_T0 (mmHg)', 'P/F_T0 (mmHg)', 
'HR_T1 (bpm)', 'RR_T1 (bpm)', 'FiO2_T1 (%)', 'PaO2_T1 (mmHg)', 'SpO2_T1 (%)', 'pCO2_T1 (mmHg)', 'FiO2_diff', 'RR_diff', 'HR_diff', 'P/F_diff', 'PaO2_diff', 'pCO2_diff']

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in DataFrame: {missing}")


class TabM(nn.Module):

    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model

        self.W = nn.Parameter(torch.randn(num_features, d_model) * 0.02)
        self.b = nn.Parameter(torch.zeros(num_features, d_model))

      
        self.col_emb = nn.Parameter(torch.randn(num_features, d_model) * 0.02)

        self.cls = nn.Parameter(torch.randn(1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1), )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, F) continuous features
        returns logits: (B, 1)
        """
        B, F = x.shape
        assert F == self.num_features, f"Expected {self.num_features} features, got {F}"

      
        x_exp = x.unsqueeze(-1)
        tokens = x_exp * self.W + self.b + self.col_emb 

       
        cls = self.cls.expand(B, -1).unsqueeze(1) 
        tokens = torch.cat([cls, tokens], dim=1)

        # Transformer
        h = self.encoder(tokens)  
        cls_h = self.norm(h[:, 0, :])  

        logits = self.head(cls_h)  
        return logits.squeeze(1)


# ------------------------------------
# Training
# ------------------------------------
@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 64
    lr: float = 1e-5
    weight_decay: float = 1e-4
    epochs: int = 100
    patience: int = 12
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    ff_mult: int = 4
    dropout: float = 0.15
    num_workers: int = 0


def make_datasets(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target: str,
    features: List[str],
    val_size: float = 0.30,
    seed: int = 42,
) -> Tuple[TensorDataset, TensorDataset, TensorDataset, StandardScaler, dict]:
    ensure_columns(df_train, features + [target])
    ensure_columns(df_test, features)


    y_full = df_train[target].astype(int).values
    X_full = df_train[features].copy()
    X_test_df = df_test[features].copy()

    medians = X_full.median()
    X_full = X_full.fillna(medians)
    X_test_df = X_test_df.fillna(medians)

    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_full.values)
    X_test_scaled = scaler.transform(X_test_df.values)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_full_scaled,
        y_full,
        test_size=val_size,
        random_state=seed,
        stratify=y_full if len(np.unique(y_full)) > 1 else None,
    )

    X_test = X_test_scaled  

    # Convert to tensors
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    test_ds = TensorDataset(X_test_t, torch.zeros(len(X_test_t)))  # dummy labels

    meta = {
        "medians": medians.to_dict(),
        "feature_order": features,
    }
    return train_ds, val_ds, test_ds, scaler, meta


def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
    return total_loss / max(n, 1)


def plot_losses(train_losses: List[float], val_losses: List[float], out_path: str):
    plt.figure(figsize=(7.5, 5.0))
    epochs = np.arange(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    # (User preference noted: do not use tight_layout)
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train TABM on RENOVATE (train) / HFNO (test).")
    parser.add_argument("--target", type=str, default="HFNO_failure",
                        help="Binary target column name in the training dataframe (default: 'HFNO Failure').")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--ff_mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="./outputs_tabm")
    args = parser.parse_args()

    set_seed(args.seed)


    if (get_renovate_data_filtered is None) or (get_hfno_all is None):
        raise ImportError(DATA_IMPORT_ERROR)

    df_train = get_renovate_data_filtered()
    df_test = get_hfno_all()

    
    if not isinstance(df_train, pd.DataFrame) or not isinstance(df_test, pd.DataFrame):
        raise TypeError("Data functions must return pandas.DataFrame objects.")

    ensure_columns(df_train, FEATURES + [args.target])
    ensure_columns(df_test, FEATURES)


    train_ds, val_ds, test_ds, scaler, meta = make_datasets(
        df_train=df_train, df_test=df_test, target=args.target, features=FEATURES,
        val_size=0.30, seed=args.seed
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TabM(
        num_features=len(FEATURES),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
    ).to(device)

    # BCEWithLogitsLoss with pos_weight for class imbalance 
    y_train_tensor = train_ds.tensors[1].cpu().numpy()
    pos = (y_train_tensor == 1).sum()
    neg = (y_train_tensor == 0).sum()
    if pos > 0:
        pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)
    else:
        pos_weight = torch.tensor([1.0], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

  
    best_val = math.inf
    best_state = None
    train_losses = []
    val_losses = []
    patience_ctr = 0

    os.makedirs(args.outdir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, device, criterion, optimizer)
        val_loss = evaluate(model, val_loader, device, criterion)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:03d} | train_loss={tr_loss:.5f} | val_loss={val_loss:.5f}")

        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"Early stopping at epoch {epoch}. Best val loss: {best_val:.6f}")
                break

 
    model_path = os.path.join(args.outdir, "tabm_best.pt")
    if best_state is not None:
        torch.save(best_state, model_path)
    else:
        torch.save(model.state_dict(), model_path)

    loss_fig_path = os.path.join(args.outdir, "loss_curve.png")
    plot_losses(train_losses, val_losses, loss_fig_path)

    config = {
        "target": args.target,
        "features": FEATURES,
        "train_len": len(train_ds),
        "val_len": len(val_ds),
        "test_len": len(test_ds),
        "device": str(device),
        "class_balance_train": {"pos": int(pos), "neg": int(neg)},
        "train_losses": train_losses,
        "val_losses": val_losses,
        "model_hparams": {
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "ff_mult": args.ff_mult,
            "dropout": args.dropout,
        },
    }
    with open(os.path.join(args.outdir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)


    try:
        if args.target in df_test.columns:
            # Prepare test with same preprocessing
            X_test = df_test[FEATURES].fillna(pd.Series(meta["medians"])).values
            X_test = scaler.transform(X_test)
            y_test = df_test[args.target].astype(int).values

            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            all_logits = []
            with torch.no_grad():
                for xb, _ in test_loader:
                    xb = xb.to(device)
                    logits = model(xb)
                    all_logits.append(logits.detach().cpu().numpy())
            logits = np.concatenate(all_logits, axis=0).squeeze()
            probs = 1 / (1 + np.exp(-logits))
            preds = (probs >= 0.5).astype(int)

        
            m = min(len(probs), len(y_test))
            probs = probs[:m]
            preds = preds[:m]
            y_test = y_test[:m]

            metrics = {
                "AUROC": float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else float("nan"),
                "AUPRC": float(average_precision_score(y_test, probs)) if len(np.unique(y_test)) > 1 else float("nan"),
                "Accuracy": float(accuracy_score(y_test, preds)),
                "Precision": float(precision_score(y_test, preds, zero_division=0)),
                "Recall": float(recall_score(y_test, preds, zero_division=0)),
                "F1": float(f1_score(y_test, preds, zero_division=0)),
            }

            with open(os.path.join(args.outdir, "test_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

     
            out_pred = df_test.copy()
            out_pred["y_prob"] = probs
            out_pred["y_pred"] = preds
            out_pred.to_csv(os.path.join(args.outdir, "test_predictions.csv"), index=False)

            print("Test metrics:", metrics)
        else:
            print(f"[Info] Target column '{args.target}' not found in HFNO test set. Skipping test metrics.")
    except Exception as e:
        print(f"[Warning] Test evaluation skipped due to error: {e}")

    print("\nArtifacts saved to:", os.path.abspath(args.outdir))
    print(" - Best model:", model_path)
    print(" - Loss curve:", loss_fig_path)
    print(" - Config:", os.path.join(args.outdir, "run_config.json"))
    print(" - (Optional) test_metrics.json and test_predictions.csv if labels available.\n")


if __name__ == "__main__":
    main()