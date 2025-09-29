# src/decision_trees_rf.py
"""
Decision Trees & Random Forests - runnable script (compat with old/new sklearn).
Saves outputs to ../outputs/
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ------------------ Output folder ------------------
OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTDIR, exist_ok=True)

# ------------------ Load dataset --------------------
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'heart.csv')
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    print("Loaded data/heart.csv")
    possible_targets = ['target', 'Target', 'outcome', 'Outcome', 'label', 'Label',
                        'HeartDisease', 'heart_disease', 'diagnosis', 'Diagnosis', 'y']
    target_col = next((c for c in possible_targets if c in df.columns), None)
    if target_col is None:
        # fallback: ask user to edit script or pick last column
        print("No obvious target column found in data/heart.csv. Using last column as target.")
        target_col = df.columns[-1]
else:
    # fallback: sklearn built-in dataset (for quick testing)
    from sklearn.datasets import load_breast_cancer
    ds = load_breast_cancer(as_frame=True)
    df = ds.frame
    target_col = 'target'
    print("Using sklearn's breast_cancer dataset for demo.")

# ------------------ Separate X and y ----------------
y = df[target_col]
X = df.drop(columns=[target_col])

# ------------------ detect numeric & categorical ----------------
numeric_cols = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

# ------------------ OneHotEncoder compatibility helper ----------------
def make_ohe(**kwargs):
    """
    Return an OneHotEncoder instance compatible with scikit-learn versions that
    use `sparse_output` (>=1.4) or `sparse` (<1.4).
    We try `sparse_output` first, fall back to `sparse`.
    """
    try:
        # preferred (newer sklearn)
        return OneHotEncoder(**{**kwargs, **{"sparse_output": False}})
    except TypeError:
        # older sklearn fallback
        return OneHotEncoder(**{**kwargs, **{"sparse": False}})

# ------------------ Preprocessing pipelines ----------------
num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])

if cat_cols:
    cat_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', make_ohe(handle_unknown='ignore'))  # will be dense matrix thanks to sparse_output/sparse False
    ])
else:
    cat_pipeline = None

transformers = []
if numeric_cols:
    transformers.append(('num', num_pipeline, numeric_cols))
if cat_cols:
    transformers.append(('cat', cat_pipeline, cat_cols))

preprocessor = ColumnTransformer(transformers, remainder='drop', sparse_threshold=0)

# ------------------ Train/test split ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Fit preprocessor and transform
preprocessor.fit(X_train)
X_train_p = preprocessor.transform(X_train)
X_test_p = preprocessor.transform(X_test)

# ------------------ Build feature names after transform ----------------
feature_names = []
if numeric_cols:
    feature_names.extend(numeric_cols)

if cat_cols:
    # get the fitted OneHotEncoder and its feature names
    try:
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        # get_feature_names_out works in modern sklearn; if not available, try alternative
        try:
            cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
        except Exception:
            # older versions may use get_feature_names
            cat_feature_names = list(ohe.get_feature_names(cat_cols))
    except Exception:
        # Fallback: create naive names (conservative)
        cat_feature_names = []
        for c in cat_cols:
            cat_feature_names.append(c)
    feature_names.extend(cat_feature_names)

# If transformer returned numpy array without 1-1 mapping, ensure length matches
try:
    if hasattr(X_train_p, "shape"):
        transformed_len = X_train_p.shape[1]
        if len(feature_names) != transformed_len:
            # fallback to generic feature names
            feature_names = [f"f{i}" for i in range(transformed_len)]
except Exception:
    pass

# ------------------ Decision Tree ----------------
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_p, y_train)
train_acc = dt.score(X_train_p, y_train)
test_acc = dt.score(X_test_p, y_test)
print(f"Decision Tree accuracy — train: {train_acc:.4f}, test: {test_acc:.4f}")

# Cross-validation (pipeline to include preprocessing)
dt_pipeline = Pipeline([('pre', preprocessor), ('clf', DecisionTreeClassifier(random_state=42))])
cv_scores = cross_val_score(dt_pipeline, X, y, cv=5, scoring='accuracy')
print("DT 5-fold CV accuracy:", cv_scores.mean(), cv_scores)

# Save a shallow tree plot (max_depth=3) for quick visualization
plt.figure(figsize=(12, 8))
plot_tree(dt, feature_names=feature_names, class_names=[str(c) for c in np.unique(y)], filled=True, max_depth=3)
plt.title("Decision Tree (shallow view max_depth=3)")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'decision_tree_shallow.png'))
plt.close()

# Export dot (full tree) - may require Graphviz system binary to render to PNG later
try:
    export_graphviz(dt, out_file=os.path.join(OUTDIR, 'tree.dot'),
                    feature_names=feature_names,
                    class_names=[str(c) for c in np.unique(y)],
                    filled=True, rounded=True)
    print("Exported tree.dot to outputs/")
except Exception as ex:
    print("Could not export dot file:", ex)

# ------------------ Random Forest ----------------
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_p, y_train)
rf_test_acc = rf.score(X_test_p, y_test)
print(f"Random Forest test accuracy: {rf_test_acc:.4f}")

rf_pipeline = Pipeline([('pre', preprocessor), ('clf', RandomForestClassifier(n_estimators=100, random_state=42))])
cv_rf = cross_val_score(rf_pipeline, X, y, cv=5, scoring='accuracy')
print("RF 5-fold CV accuracy:", cv_rf.mean(), cv_rf)

# Feature importances (match to feature names)
try:
    fi = rf.feature_importances_
    fi_series = pd.Series(fi, index=feature_names).sort_values(ascending=False)[:30]
    plt.figure(figsize=(8, 6))
    fi_series.plot(kind='barh')
    plt.gca().invert_yaxis()
    plt.title('Top Feature Importances (Random Forest)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'rf_feature_importances.png'))
    plt.close()
except Exception as ex:
    print("Could not plot feature importances:", ex)

# ------------------ Confusion matrix and report for RF (correct place) ----------
y_pred_rf = rf.predict(X_test_p)
print("Random Forest classification report:\n", classification_report(y_test, y_pred_rf))
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title('RF Confusion Matrix')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'rf_confusion_matrix.png'))
plt.close()

# Optionally save trained models (uncomment if desired)
# joblib.dump(dt, os.path.join(OUTDIR, 'decision_tree.joblib'))
# joblib.dump(rf, os.path.join(OUTDIR, 'random_forest.joblib'))
