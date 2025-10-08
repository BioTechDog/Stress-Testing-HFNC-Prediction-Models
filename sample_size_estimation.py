# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import roc_auc_score, make_scorer, accuracy_score, confusion_matrix
# from tabpfn import TabPFNClassifier

# from sklearn.impute import KNNImputer

# from data_loader_preprocess import (
#     get_renovate_data_filtered, 
#     get_renovate_data_all,
#     get_hfno_all
# )

# # Load data
# RENOVATE_data_all = get_renovate_data_all()
# # features_used = ['Age (y)', 'HR_T0 (bpm)', 'RR_T0 (bpm)', 'FiO2_T0 (%)', 'PaO2_T0 (mmHg)', 'P/F_T0 (mmHg)',
# #        'SpO2_T0 (%)', 'pCO2_T0 (mmHg)', 'FiO2_T1 (%)', 'HR_T1 (bpm)', 'P/F_T1 (mmHg)', 'PaO2_T1 (mmHg)',
# #        'RR_T1 (bpm)', 'SpO2_T1 (%)', 'pCO2_T1 (mmHg)']

# features_used = ['P/F_T1 (mmHg)', 'PaO2_T1 (mmHg)', 'RR_T1 (bpm)', 'pCO2_T1 (mmHg)', 'mROX_T1', 'ROX_T1']

# HFNO_combined = RENOVATE_data_all.dropna(subset=features_used).copy()

# x = HFNO_combined[features_used]
# y = HFNO_combined["HFNO_failure"]

# # Fit reference model (Logistic Regression example)
# # base_model = TabPFNClassifier(device='cuda', balance_probabilities=True)
# base_model = XGBClassifier(class_weight='balanced')
# #lr_model = LogisticRegression()
# base_model.fit(x, y)
# base_probs = base_model.predict_proba(x)[:, 1]

# # Generate new binary outcomes
# np.random.seed(1)
# runis = np.random.uniform(0, 1, size=len(base_probs))
# lry = (runis < base_probs).astype(int)

# # Create new dataset with generated outcome
# BASE = HFNO_combined[features_used].copy()
# BASE['lry'] = lry


# # Split into development and validation sets
# devBASE, valBASE = train_test_split(BASE, test_size=0.3, random_state=1)

# # Initialize results storage
# results = []

# # Sample sizes to test
# sample_sizes = [20, 50, 100, 150, 200, 250, 300, 400, 500, len(devBASE)]

# for j in sample_sizes:
#     for i in range(50):
#         sampledata = devBASE.sample(n=min(j, len(devBASE)), random_state=i)

#         X_train = sampledata.drop(columns=['lry'])
#         y_train = sampledata['lry']
#         X_test = valBASE.drop(columns=['lry'])
#         y_test = valBASE['lry']

#         #Logistic Regression
#         lr_model = LogisticRegression()
#         lr_model.fit(X_train, y_train)
#         lr_probs_train = lr_model.predict_proba(X_train)[:, 1]
#         lr_probs_test = lr_model.predict_proba(X_test)[:, 1]
#         lr_auc_train = roc_auc_score(y_train, lr_probs_train)
#         lr_auc_test = roc_auc_score(y_test, lr_probs_test)

#         # Decision Tree
#         cart_model = DecisionTreeClassifier()
#         cart_model.fit(X_train, y_train)
#         cart_probs_train = cart_model.predict_proba(X_train)[:, 1]
#         cart_probs_test = cart_model.predict_proba(X_test)[:, 1]
#         cart_auc_train = roc_auc_score(y_train, cart_probs_train)
#         cart_auc_test = roc_auc_score(y_test, cart_probs_test)

#         # SVM
#         svm = SVC(probability=True)
#         svm.fit(X_train, y_train)
#         nn_probs_train = svm.predict_proba(X_train)[:, 1]
#         nn_probs_test = svm.predict_proba(X_test)[:, 1]
#         nn_auc_train = roc_auc_score(y_train, nn_probs_train)
#         nn_auc_test = roc_auc_score(y_test, nn_probs_test)

#         #Random Forest
#         rf_model = RandomForestClassifier()
#         rf_model.fit(X_train, y_train)
#         rf_probs_train = rf_model.predict_proba(X_train)[:, 1]
#         rf_probs_test = rf_model.predict_proba(X_test)[:, 1]
#         rf_auc_train = roc_auc_score(y_train, rf_probs_train)
#         rf_auc_test = roc_auc_score(y_test, rf_probs_test)

#         # Gaussian Naive Bayes
#         nb_model = GaussianNB()
#         nb_model.fit(X_train, y_train)
#         nb_probs_train = nb_model.predict_proba(X_train)[:, 1]
#         nb_probs_test = nb_model.predict_proba(X_test)[:, 1]
#         nb_auc_train = roc_auc_score(y_train, nb_probs_train)
#         nb_auc_test = roc_auc_score(y_test, nb_probs_test)

#         # XGBoost
#         xg_model = XGBClassifier()
#         xg_model.fit(X_train, y_train)
#         xg_probs_train = xg_model.predict_proba(X_train)[:, 1]
#         xg_probs_test = xg_model.predict_proba(X_test)[:, 1]
#         xg_auc_train = roc_auc_score(y_train, xg_probs_train)
#         xg_auc_test = roc_auc_score(y_test, xg_probs_test)

#         # TabPFN
#         clf_tab = TabPFNClassifier(device='cuda', balance_probabilities=True)
#         clf_tab.fit(X_train, y_train)
#         tabf_probs_train = clf_tab.predict_proba(X_train)[:, 1]
#         tabf_probs_test = clf_tab.predict_proba(X_test)[:, 1]
#         tabf_auc_train = roc_auc_score(y_train, tabf_probs_train)
#         tabf_auc_test = roc_auc_score(y_test, tabf_probs_test)


#         # Store results
#         results.append({
#             'Sample number per size': i,
#             'Sample size': j,
#             'LogisticROCtraining': lr_auc_train,
#             'LogisticROCtest': lr_auc_test,
#             'DecisionTreeROCtraining': cart_auc_train,
#             'DecisionTreeROCtest': cart_auc_test,
#             'RandomForestROCtraining': rf_auc_train,
#             'RandomForestROCtest': rf_auc_test,
#             'GaussianNBROCtraining': nb_auc_train,
#             'GaussianNBROCtest': nb_auc_test,
#             'XGBoostROCtraining': xg_auc_train,
#             'XGBoostROCtest': xg_auc_test,
#             'SVMROCtraining': nn_auc_train,
#             'SVMROCtest': nn_auc_test,
#             'TabPFNROCtraining': tabf_auc_train,
#             'TabPFNROCtest': tabf_auc_test
#         })

# # Convert results to DataFrame and save to CSV
# results_df = pd.DataFrame(results)
# # results_df.to_csv("C:/Users/10059/Downloads/Critical Care submitted HR_RR_TabPFN/HFNC_tab_code/results/HNSCC_training_and_test_vs_RandomClassifier.csv", index=False)
# # results_df.to_csv("C:/Users/10059/Downloads/Critical Care submitted HR_RR_TabPFN/HFNC_tab_code/results/HNSCC_training_and_test_vs_Logistic.csv", index=False)
# # results_df.to_csv("C:/Users/10059/Downloads/Critical Care submitted HR_RR_TabPFN/HFNC_tab_code/results/HNSCC_training_and_test_vs_TabPFN.csv", index=False)
# results_df.to_csv("C:/Users/10059/Downloads/Critical Care submitted HR_RR_TabPFN/HFNC_tab_code/results/HNSCC_training_and_test_vs_SVM.csv", index=False)
# # Print a sample of results
# print(results_df.head())

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier

from sklearn.impute import KNNImputer

from data_loader_preprocess import (
    get_renovate_data_filtered, 
    get_renovate_data_all,
    get_hfno_all
)

# ===== NEW: PyTorch TABM imports =====
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# ===== NEW: Simple, compact TABM for tabular data =====
class TabM(nn.Module):
    """Lightweight Transformer-style model for tabular (continuous) features.
    Projects each feature to a token, prepends a CLS token, runs a TransformerEncoder,
    then a small MLP head for a single logit (binary classification).
    """
    def __init__(self, num_features, d_model=64, n_heads=4, n_layers=2, ff_mult=4, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model

        # Per-feature tokenizer parameters
        self.W = nn.Parameter(torch.randn(num_features, d_model) * 0.02)
        self.b = nn.Parameter(torch.zeros(num_features, d_model))
        self.col_emb = nn.Parameter(torch.randn(num_features, d_model) * 0.02)

        # CLS token
        self.cls = nn.Parameter(torch.randn(1, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        # x: (B, F)
        B, F = x.shape
        assert F == self.num_features
        x_exp = x.unsqueeze(-1)                                # (B,F,1)
        tokens = x_exp * self.W + self.b + self.col_emb        # (B,F,D)
        cls = self.cls.expand(B, -1).unsqueeze(1)              # (B,1,D)
        tokens = torch.cat([cls, tokens], dim=1)               # (B,F+1,D)
        h = self.encoder(tokens)                               # (B,F+1,D)
        cls_h = self.norm(h[:, 0, :])                          # (B,D)
        logits = self.head(cls_h).squeeze(1)                   # (B,)
        return logits

def train_tabm_return_probs(X_train, y_train, X_test, seed=1,
                            max_epochs=60, batch_size=64, lr=1e-3, weight_decay=1e-4,
                            d_model=64, n_heads=4, n_layers=2, ff_mult=4, dropout=0.1,
                            val_frac=0.2, patience=8, device=None):
    """Train TABM on (X_train,y_train) with a small internal validation split.
    Returns train_probs, test_probs (sigmoid outputs).
    """
    rng = np.random.RandomState(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make internal val split for early stopping
    if len(np.unique(y_train)) > 1 and len(y_train) >= 10 and val_frac > 0:
        idx = np.arange(len(y_train))
        rng.shuffle(idx)
        cut = int(len(idx) * (1 - val_frac))
        tr_idx, va_idx = idx[:cut], idx[cut:]
    else:
        tr_idx = np.arange(len(y_train))
        va_idx = np.array([], dtype=int)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train.iloc[tr_idx] if hasattr(X_train, 'iloc') else X_train[tr_idx])
    y_tr = (y_train.iloc[tr_idx] if hasattr(y_train, 'iloc') else y_train[tr_idx]).astype(float)

    if va_idx.size > 0:
        X_va = scaler.transform(X_train.iloc[va_idx] if hasattr(X_train, 'iloc') else X_train[va_idx])
        y_va = (y_train.iloc[va_idx] if hasattr(y_train, 'iloc') else y_train[va_idx]).astype(float)
    else:
        X_va = None
        y_va = None

    X_te = scaler.transform(X_test)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr.values if hasattr(y_tr, 'values') else y_tr, dtype=torch.float32)

    if X_va is not None:
        X_va_t = torch.tensor(X_va, dtype=torch.float32)
        y_va_t = torch.tensor(y_va.values if hasattr(y_va, 'values') else y_va, dtype=torch.float32)

    X_te_t = torch.tensor(X_te, dtype=torch.float32)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = TabM(num_features=X_tr.shape[1], d_model=d_model, n_heads=n_heads,
                 n_layers=n_layers, ff_mult=ff_mult, dropout=dropout).to(device)

    # Handle imbalance with pos_weight
    pos = float((y_tr_t == 1).sum().item())
    neg = float((y_tr_t == 0).sum().item())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = np.inf
    best_state = None
    patience_ctr = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # Early stopping on internal val if present, else on training loss
        with torch.no_grad():
            model.eval()
            if X_va is not None:
                logits_va = model(X_va_t.to(device))
                val_loss = criterion(logits_va, y_va_t.to(device)).item()
            else:
                logits_tr = model(X_tr_t.to(device))
                val_loss = criterion(logits_tr, y_tr_t.to(device)).item()

            if val_loss + 1e-6 < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Return probabilities for train and test
    with torch.no_grad():
        model.eval()
        logits_tr = model(X_tr_t.to(device)).cpu().numpy()
        logits_te = model(X_te_t.to(device)).cpu().numpy()
        probs_tr = 1.0 / (1.0 + np.exp(-logits_tr))
        probs_te = 1.0 / (1.0 + np.exp(-logits_te))

    # Map back to original train order
    if va_idx.size > 0:
        # rebuild full-length train probs aligned with X_train
        probs_full = np.empty(len(y_train), dtype=float)
        probs_full[tr_idx] = probs_tr
        # for the held-out internal val, compute probs too
        X_va_full_t = torch.tensor(scaler.transform(X_train.iloc[va_idx] if hasattr(X_train, 'iloc') else X_train[va_idx]), dtype=torch.float32)
        with torch.no_grad():
            logits_va_full = model(X_va_full_t.to(device)).cpu().numpy()
        probs_full[va_idx] = 1.0 / (1.0 + np.exp(-logits_va_full))
        probs_tr = probs_full

    return probs_tr, probs_te

# =========================================
# Original pipeline
# =========================================

# Load data
RENOVATE_data_all = get_renovate_data_all()

# features_used = ['Age (y)', 'HR_T0 (bpm)', 'RR_T0 (bpm)', 'FiO2_T0 (%)', 'PaO2_T0 (mmHg)', 'P/F_T0 (mmHg)',
#        'SpO2_T0 (%)', 'pCO2_T0 (mmHg)', 'FiO2_T1 (%)', 'HR_T1 (bpm)', 'P/F_T1 (mmHg)', 'PaO2_T1 (mmHg)',
#        'RR_T1 (bpm)', 'SpO2_T1 (%)', 'pCO2_T1 (mmHg)']

features_used = ['P/F_T1 (mmHg)', 'PaO2_T1 (mmHg)', 'RR_T1 (bpm)', 'pCO2_T1 (mmHg)', 'mROX_T1', 'ROX_T1']

HFNO_combined = RENOVATE_data_all.dropna(subset=features_used).copy()

x = HFNO_combined[features_used]
y = HFNO_combined["HFNO_failure"]

# Fit reference model to generate synthetic outcomes
# base_model = TabPFNClassifier(device='cuda', balance_probabilities=True)
base_model = TabPFNClassifier(device='cuda', balance_probabilities=True)
base_model.fit(x, y)
base_probs = base_model.predict_proba(x)[:, 1]
# base_train_probs, _ = train_tabm_return_probs(
#     X_train=x,
#     y_train=y,
#     X_test=x,  # not used for base, but required by helper
#     seed=1,
#     max_epochs=80,
#     batch_size=64,
#     lr=1e-3,
#     weight_decay=1e-4,
#     d_model=64,
#     n_heads=4,
#     n_layers=2,
#     ff_mult=4,
#     dropout=0.15,
#     val_frac=0.2,
#     patience=10
# )
# base_probs = base_train_probs

# Generate new binary outcomes
np.random.seed(1)
runis = np.random.uniform(0, 1, size=len(base_probs))
lry = (runis < base_probs).astype(int)

# Create new dataset with generated outcome
BASE = HFNO_combined[features_used].copy()
BASE['lry'] = lry

# Split into development and validation sets
devBASE, valBASE = train_test_split(BASE, test_size=0.3, random_state=1)

# Initialize results storage
results = []

# Sample sizes to test
sample_sizes = [20, 50, 100, 150, 200, 250, 300, 400, 500, len(devBASE)]

for j in sample_sizes:
    for i in range(50):
        sampledata = devBASE.sample(n=min(j, len(devBASE)), random_state=i)

        X_train = sampledata.drop(columns=['lry'])
        y_train = sampledata['lry']
        X_test = valBASE.drop(columns=['lry'])
        y_test = valBASE['lry']

        # Logistic Regression
        lr_model = LogisticRegression()
        lr_model.fit(X_train, y_train)
        lr_probs_train = lr_model.predict_proba(X_train)[:, 1]
        lr_probs_test = lr_model.predict_proba(X_test)[:, 1]
        lr_auc_train = roc_auc_score(y_train, lr_probs_train)
        lr_auc_test = roc_auc_score(y_test, lr_probs_test)

        # Decision Tree
        cart_model = DecisionTreeClassifier()
        cart_model.fit(X_train, y_train)
        cart_probs_train = cart_model.predict_proba(X_train)[:, 1]
        cart_probs_test = cart_model.predict_proba(X_test)[:, 1]
        cart_auc_train = roc_auc_score(y_train, cart_probs_train)
        cart_auc_test = roc_auc_score(y_test, cart_probs_test)

        # SVM
        svm = SVC(probability=True)
        svm.fit(X_train, y_train)
        nn_probs_train = svm.predict_proba(X_train)[:, 1]
        nn_probs_test = svm.predict_proba(X_test)[:, 1]
        nn_auc_train = roc_auc_score(y_train, nn_probs_train)
        nn_auc_test = roc_auc_score(y_test, nn_probs_test)

        # Random Forest
        rf_model = RandomForestClassifier()
        rf_model.fit(X_train, y_train)
        rf_probs_train = rf_model.predict_proba(X_train)[:, 1]
        rf_probs_test = rf_model.predict_proba(X_test)[:, 1]
        rf_auc_train = roc_auc_score(y_train, rf_probs_train)
        rf_auc_test = roc_auc_score(y_test, rf_probs_test)

        # Gaussian Naive Bayes
        nb_model = GaussianNB()
        nb_model.fit(X_train, y_train)
        nb_probs_train = nb_model.predict_proba(X_train)[:, 1]
        nb_probs_test = nb_model.predict_proba(X_test)[:, 1]
        nb_auc_train = roc_auc_score(y_train, nb_probs_train)
        nb_auc_test = roc_auc_score(y_test, nb_probs_test)

        # XGBoost
        xg_model = XGBClassifier()
        xg_model.fit(X_train, y_train)
        xg_probs_train = xg_model.predict_proba(X_train)[:, 1]
        xg_probs_test = xg_model.predict_proba(X_test)[:, 1]
        xg_auc_train = roc_auc_score(y_train, xg_probs_train)
        xg_auc_test = roc_auc_score(y_test, xg_probs_test)

        # TabPFN
        clf_tab = TabPFNClassifier(device='cuda', balance_probabilities=True)
        clf_tab.fit(X_train, y_train)
        tabf_probs_train = clf_tab.predict_proba(X_train)[:, 1]
        tabf_probs_test = clf_tab.predict_proba(X_test)[:, 1]
        tabf_auc_train = roc_auc_score(y_train, tabf_probs_train)
        tabf_auc_test = roc_auc_score(y_test, tabf_probs_test)

        # ===== NEW: TABM (PyTorch) =====
        # Small, regularized model + early stopping on internal split.
        tabm_train_probs, tabm_test_probs = train_tabm_return_probs(
            X_train=X_train, y_train=y_train, X_test=X_test,
            seed=i + j,
            max_epochs=60, batch_size=64, lr=1e-3, weight_decay=1e-4,
            d_model=64, n_heads=4, n_layers=2, ff_mult=4, dropout=0.15,
            val_frac=0.2, patience=8
        )
        tabm_auc_train = roc_auc_score(y_train, tabm_train_probs)
        tabm_auc_test = roc_auc_score(y_test, tabm_test_probs)

        # Store results
        results.append({
            'Sample number per size': i,
            'Sample size': j,
            'LogisticROCtraining': lr_auc_train,
            'LogisticROCtest': lr_auc_test,
            'DecisionTreeROCtraining': cart_auc_train,
            'DecisionTreeROCtest': cart_auc_test,
            'RandomForestROCtraining': rf_auc_train,
            'RandomForestROCtest': rf_auc_test,
            'GaussianNBROCtraining': nb_auc_train,
            'GaussianNBROCtest': nb_auc_test,
            'XGBoostROCtraining': xg_auc_train,
            'XGBoostROCtest': xg_auc_test,
            'SVMROCtraining': nn_auc_train,
            'SVMROCtest': nn_auc_test,
            'TabPFNROCtraining': tabf_auc_train,
            'TabPFNROCtest': tabf_auc_test,
            'TABMROCtraining': tabm_auc_train,
            'TABMROCtest': tabm_auc_test
        })

# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
# Path below can be changed to your own output directory
results_df.to_csv("results/HFNC_sample_size_results_with_TabPFN.csv", index=False)