"""
Visualisierung von Evaluations-Ergebnissen.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List
import json
import numpy as np

sns.set_style("whitegrid")


def plot_model_comparison(
    results: Dict[str, Dict], save_path: Path, metric_keys: List[str] = None
):
    """
    Erstellt Vergleichs-Plots für verschiedene Modelle.

    Args:
        results: Dict mit Modellnamen als Keys und Metrics als Values
        save_path: Wo die Plots gespeichert werden
        metric_keys: Welche Metriken geplottet werden sollen
    """
    if metric_keys is None:
        metric_keys = ["exact_match_accuracy", "precision", "recall", "f1"]

    # Prepare data
    models = list(results.keys())

    # Plot metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(metric_keys[:4]):
        values = [results[model].get(metric, 0) for model in models]

        axes[idx].bar(
            models, values, color=["#1f77b4", "#ff7f0e", "#2ca02c"][: len(models)]
        )
        axes[idx].set_ylabel(metric.replace("_", " ").title())
        axes[idx].set_ylim([0, 1])
        axes[idx].set_title(f'{metric.replace("_", " ").title()} Comparison')

        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / "model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, save_path: Path, labels: List[str] = None):
    """Plottet Confusion Matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_results_report(results: Dict[str, Dict], save_path: Path):
    """Erstellt HTML-Report mit allen Ergebnissen."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Evaluation Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            h1 { color: #333; }
            .metric { font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Medical Diagnosis Model Evaluation Report</h1>
    """

    # Add results table
    html += "<h2>Model Comparison</h2><table><tr><th>Model</th>"

    # Get all metric keys
    all_metrics = set()
    for model_results in results.values():
        all_metrics.update(model_results.keys())

    for metric in sorted(all_metrics):
        html += f"<th>{metric}</th>"
    html += "</tr>"

    for model_name, metrics in results.items():
        html += f"<tr><td class='metric'>{model_name}</td>"
        for metric in sorted(all_metrics):
            value = metrics.get(metric, "N/A")
            if isinstance(value, (int, float)):
                html += f"<td>{value:.4f}</td>"
            else:
                html += f"<td>{value}</td>"
        html += "</tr>"

    html += "</table></body></html>"

    # Save report
    report_path = save_path / "evaluation_report.html"
    with open(report_path, "w") as f:
        f.write(html)

    return report_path
