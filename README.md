# Decision-tree-and-Random-forest
Decision Trees & Random Forests Classification

This repository contains a Python script that demonstrates Decision Tree and Random Forest classifiers on tabular datasets. 
It includes preprocessing, model training, evaluation, visualization, and cross-validation. The script is compatible with modern and older versions of scikit-learn.
Features
Automatically loads a dataset:
Primary: data/heart.csv if available.
Fallback: scikit-learn’s built-in breast_cancer dataset for quick testing.
Automatic detection of numeric and categorical features.
Preprocessing pipelines:
Numeric features: median imputation + standard scaling.
Categorical features: most frequent imputation + one-hot encoding (supports old/new scikit-learn versions).
Train/test split with stratification.
Decision Tree Classifier:
Train & test accuracy.
5-fold cross-validation.
Shallow tree visualization (max depth=3) saved to outputs/decision_tree_shallow.png.
Export full tree as Graphviz .dot file (outputs/tree.dot).
Random Forest Classifier:
Train & test accuracy.
5-fold cross-validation.
Feature importance visualization (top 30 features) saved to outputs/rf_feature_importances.png.
Confusion matrix saved to outputs/rf_confusion_matrix.png.
Classification report (precision, recall, F1-score) for test set.
Fully reproducible with random seed (random_state=42).

Output folder (outputs/) automatically created for all saved plots and exported files.

Dependencies

Python ≥3.9

numpy

pandas

scikit-learn ≥0.24

matplotlib

seaborn

joblib (optional, for saving models)

Install dependencies using:

pip install numpy pandas scikit-learn matplotlib seaborn joblib

Usage

Run the script directly:

python src/decision_trees_rf.py


The script will automatically detect numeric/categorical columns, preprocess them, train Decision Tree and Random Forest classifiers, and save the outputs to the outputs/ folder.

If data/heart.csv is not found, it uses scikit-learn’s breast_cancer dataset for demonstration.

Output Files

outputs/decision_tree_shallow.png → visualization of a shallow decision tree.

outputs/tree.dot → full decision tree exported in Graphviz .dot format.

outputs/rf_feature_importances.png → top feature importances for Random Forest.

outputs/rf_confusion_matrix.png → heatmap of Random Forest confusion matrix.

(Optional) decision_tree.joblib / random_forest.joblib → saved trained models if joblib.dump is enabled.

Example Output
Decision Tree accuracy — train: 1.0000, test: 0.9123
DT 5-fold CV accuracy: 0.9173 [0.912, 0.903, 0.929, 0.956, 0.884]
Random Forest test accuracy: 0.9561
RF 5-fold CV accuracy: 0.9561 [0.921, 0.938, 0.982, 0.964, 0.973]
Random Forest classification report:
               precision    recall  f1-score   support

           0       0.95      0.93      0.94        42
           1       0.96      0.97      0.97        72

    accuracy                           0.96       114
   macro avg       0.96      0.95      0.95       114
weighted avg       0.96      0.96      0.96       114


The feature importance plot can help identify the most predictive features in your dataset.

Cross-validation provides an unbiased estimate of model performance.
