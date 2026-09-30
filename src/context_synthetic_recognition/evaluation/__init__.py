"""Evaluation: protocols, baselines, metrics, AUC/ROC and margins (Definitions 2 and 3).

The layer between the core and the services: it fits the model on folds, collects decisions and
scores, and computes the workbook's evaluation sheets (*Margin Analysis*, *Accuracy*, *Confusion
Matrix*, *Precision, Recall, F1 Score*, *ROC Curve & AUC*, *Leave-One-Out*). It performs no I/O.
"""
