#!/usr/bin/env python
"""
Analysis script for prediction results
Provides detailed analysis of model predictions including confusion matrices and error analysis
"""

import argparse
import json
import os
from collections import Counter, defaultdict
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
import pandas as pd


def analyze_single_task(results, mappings=None, output_file=None):
    """Analyze single-task model predictions

    Args:
        results: Prediction results dictionary
        mappings: Optional label mappings for decoding. If None, assumes labels are already decoded strings.
        output_file: Path to save the analysis report
    """
    predictions = results['predictions']

    # Set up output file
    if output_file is None:
        output_file = "analysis_report.txt"

    # Also create a plots directory
    plot_dir = os.path.join("./PLOTS", os.path.splitext(output_file)[0])
    os.makedirs(plot_dir, exist_ok=True)

    # Open output file for writing
    with open(output_file, 'w') as f:
        def print_and_write(text=""):
            """Helper to print to console and write to file"""
            print(text)
            f.write(text + "\n")

        print_and_write("\n" + "="*80)
        print_and_write("SINGLE-TASK MODEL ANALYSIS")
        print_and_write("="*80)

        print_and_write(f"\nOverall Statistics:")
        print_and_write(f"  Total samples: {results['num_samples']}")
        print_and_write(f"  Accuracy: {results['accuracy']:.4f}")

        # Decode labels if mappings provided
        if mappings and mappings.get('type') == 'single_task':
            id2label = {int(k): v for k, v in mappings['id2label'].items()}

            def decode_label(label_id):
                # Handle both int and string inputs (backwards compatibility)
                if isinstance(label_id, str):
                    return label_id
                return id2label.get(label_id, f"Unknown_{label_id}")
        else:
            def decode_label(label):
                return label

        # Separate correct and incorrect predictions
        correct = [p for p in predictions if p['true_label'] == p['predicted_label']]
        incorrect = [p for p in predictions if p['true_label'] != p['predicted_label']]

        print_and_write(f"  Correct: {len(correct)}")
        print_and_write(f"  Incorrect: {len(incorrect)}")

        # Calculate per-class accuracy and F1 scores
        print_and_write("\n" + "-"*80)
        print_and_write("Calculating Per-Class Metrics (Accuracy, Precision, Recall, F1)...")
        print_and_write("-"*80)

        label_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'tp': 0, 'fp': 0, 'fn': 0})

        # Calculate TP, FP, FN for each class
        for p in predictions:
            true_label = p['true_label']
            pred_label = p['predicted_label']

            # Update totals
            label_stats[true_label]['total'] += 1

            # True Positive
            if true_label == pred_label:
                label_stats[true_label]['correct'] += 1
                label_stats[true_label]['tp'] += 1
            else:
                # False Negative for true class
                label_stats[true_label]['fn'] += 1
                # False Positive for predicted class
                label_stats[pred_label]['fp'] += 1

        # Calculate metrics for each class
        class_metrics = []
        for label, stats in label_stats.items():
            total = stats['total']
            tp = stats['tp']
            fp = stats['fp']
            fn = stats['fn']

            # Accuracy
            accuracy = stats['correct'] / total if total > 0 else 0

            # Precision
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0

            # Recall
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            # F1 Score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            class_metrics.append({
                'label': label,
                'label_str': decode_label(label),
                'total': total,
                'correct': stats['correct'],
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })

        print(f"Calculated metrics for {len(class_metrics)} classes")
        df = pd.DataFrame.from_dict(class_metrics)
        df.to_csv(os.path.join(plot_dir, "per_class_metrics.csv"), index=False)
        print(f'Saved per-class metrics to: {os.path.join(plot_dir, "per_class_metrics.csv")}')

        # Sort by label
        class_metrics.sort(key=lambda x: x['label'])

        # Extract accuracy and F1 arrays for statistics
        accuracies = np.array([m['accuracy'] for m in class_metrics])
        f1_scores = np.array([m['f1'] for m in class_metrics])
        sizes = np.array([m['total'] for m in class_metrics])

        # Calculate distribution statistics
        print_and_write("\n" + "="*80)
        print_and_write("PER-CLASS ACCURACY DISTRIBUTION STATISTICS")
        print_and_write("="*80)

        print_and_write(f"\nAccuracy Statistics:")
        print_and_write(f"  Mean:       {np.mean(accuracies):.4f}")
        print_and_write(f"  Median:     {np.median(accuracies):.4f}")
        print_and_write(f"  Std Dev:    {np.std(accuracies):.4f}")
        print_and_write(f"  Variance:   {np.var(accuracies):.4f}")
        print_and_write(f"  Min:        {np.min(accuracies):.4f}")
        print_and_write(f"  Max:        {np.max(accuracies):.4f}")
        print_and_write(f"  Kurtosis:   {scipy_stats.kurtosis(accuracies):.4f} (peakiness)")
        print_and_write(f"  Skewness:   {scipy_stats.skew(accuracies):.4f}")

        print_and_write(f"\nPercentiles:")
        for percentile in [10, 25, 50, 75, 90, 95, 99]:
            val = np.percentile(accuracies, percentile)
            print_and_write(f"  {percentile}th percentile: {val:.4f}")

        # Low accuracy classes
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        print_and_write(f"\nLow Accuracy Classes:")
        for threshold in thresholds:
            count = np.sum(accuracies < threshold)
            percentage = (count / len(accuracies)) * 100
            print_and_write(f"  Accuracy < {threshold:.1f}: {count}/{len(accuracies)} ({percentage:.2f}%)")

        # High accuracy classes
        print_and_write(f"\nHigh Accuracy Classes:")
        for threshold in [0.9, 0.95, 0.99, 1.0]:
            count = np.sum(accuracies >= threshold)
            percentage = (count / len(accuracies)) * 100
            print_and_write(f"  Accuracy >= {threshold:.2f}: {count}/{len(accuracies)} ({percentage:.2f}%)")

        # F1 Score Statistics
        print_and_write("\n" + "="*80)
        print_and_write("PER-CLASS F1 SCORE DISTRIBUTION STATISTICS")
        print_and_write("="*80)

        print_and_write(f"\nF1 Score Statistics:")
        print_and_write(f"  Mean:       {np.mean(f1_scores):.4f}")
        print_and_write(f"  Median:     {np.median(f1_scores):.4f}")
        print_and_write(f"  Std Dev:    {np.std(f1_scores):.4f}")
        print_and_write(f"  Variance:   {np.var(f1_scores):.4f}")
        print_and_write(f"  Min:        {np.min(f1_scores):.4f}")
        print_and_write(f"  Max:        {np.max(f1_scores):.4f}")
        print_and_write(f"  Kurtosis:   {scipy_stats.kurtosis(f1_scores):.4f} (peakiness)")
        print_and_write(f"  Skewness:   {scipy_stats.skew(f1_scores):.4f}")

        print_and_write(f"\nPercentiles:")
        for percentile in [10, 25, 50, 75, 90, 95, 99]:
            val = np.percentile(f1_scores, percentile)
            print_and_write(f"  {percentile}th percentile: {val:.4f}")

        # Low F1 classes
        print_and_write(f"\nLow F1 Score Classes:")
        for threshold in thresholds:
            count = np.sum(f1_scores < threshold)
            percentage = (count / len(f1_scores)) * 100
            print_and_write(f"  F1 < {threshold:.1f}: {count}/{len(f1_scores)} ({percentage:.2f}%)")

        # High F1 classes
        print_and_write(f"\nHigh F1 Score Classes:")
        for threshold in [0.9, 0.95, 0.99, 1.0]:
            count = np.sum(f1_scores >= threshold)
            percentage = (count / len(f1_scores)) * 100
            print_and_write(f"  F1 >= {threshold:.2f}: {count}/{len(f1_scores)} ({percentage:.2f}%)")

        # Per-Class Details (show all classes)
        # print_and_write("\n" + "="*80)
        # print_and_write("PER-CLASS DETAILED METRICS")
        # print_and_write("="*80)

        # print_and_write(f"\n{'Label':<40} {'Total':<8} {'Correct':<8} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8}")
        # print_and_write("-"*100)

        # for m in class_metrics:
        #     # Truncate long labels
        #     display_label = m['label_str'][:37] + "..." if len(m['label_str']) > 40 else m['label_str']
        #     print_and_write(
        #         f"{display_label:<40} {m['total']:<8} {m['correct']:<8} "
        #         f"{m['accuracy']:<8.4f} {m['precision']:<8.4f} {m['recall']:<8.4f} {m['f1']:<8.4f}"
        #     )

        # Most common errors
        print_and_write("\n" + "="*80)
        print_and_write("MOST COMMON ERRORS (Top 20)")
        print_and_write("="*80)

        error_pairs = Counter()
        for p in incorrect:
            error_pairs[(p['true_label'], p['predicted_label'])] += 1

        print_and_write(f"\n{'True Label':<35} {'Predicted':<35} {'Count':<10}")
        print_and_write("-"*80)
        for (true_label, pred_label), count in error_pairs.most_common(20):
            true_decoded = decode_label(true_label)
            pred_decoded = decode_label(pred_label)
            # Truncate long labels
            true_display = true_decoded[:32] + "..." if len(true_decoded) > 35 else true_decoded
            pred_display = pred_decoded[:32] + "..." if len(pred_decoded) > 35 else pred_decoded
            print_and_write(f"{true_display:<35} {pred_display:<35} {count:<10}")

        # Show some example errors
        print_and_write("\n" + "="*80)
        print_and_write("EXAMPLE ERRORS (First 10)")
        print_and_write("="*80)
        for i, p in enumerate(incorrect[:10]):
            print_and_write(f"\n{i+1}. {p['image_path']}")
            true_decoded = decode_label(p['true_label'])
            pred_decoded = decode_label(p['predicted_label'])
            print_and_write(f"   True: {true_decoded}")
            print_and_write(f"   Predicted: {pred_decoded}")

    # Create plots
    print(f"\nGenerating distribution plots...")

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)

    # Plot 1: Accuracy Distribution
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram
    axes[0].hist(accuracies, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(accuracies), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(accuracies):.3f}')
    axes[0].axvline(np.median(accuracies), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(accuracies):.3f}')
    axes[0].set_xlabel('Per-Class Accuracy', fontsize=12)
    axes[0].set_ylabel('Number of Classes', fontsize=12)
    axes[0].set_title('Distribution of Per-Class Accuracy', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # KDE plot
    axes[1].hist(accuracies, bins=50, density=True, alpha=0.5, edgecolor='black', label='Histogram')
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(accuracies)
    x_range = np.linspace(0, 1, 200)
    axes[1].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    axes[1].axvline(np.mean(accuracies), color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[1].set_xlabel('Per-Class Accuracy', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title('Density Plot of Per-Class Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    accuracy_plot_path = os.path.join(plot_dir, 'accuracy_distribution.png')
    plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved accuracy distribution plot to: {accuracy_plot_path}")
    plt.close()

    # Plot 2: F1 Score Distribution
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram
    axes[0].hist(f1_scores, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[0].axvline(np.mean(f1_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(f1_scores):.3f}')
    axes[0].axvline(np.median(f1_scores), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(f1_scores):.3f}')
    axes[0].set_xlabel('Per-Class F1 Score', fontsize=12)
    axes[0].set_ylabel('Number of Classes', fontsize=12)
    axes[0].set_title('Distribution of Per-Class F1 Score', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # KDE plot
    axes[1].hist(f1_scores, bins=50, density=True, alpha=0.5, edgecolor='black', color='orange', label='Histogram')
    kde_f1 = gaussian_kde(f1_scores)
    axes[1].plot(x_range, kde_f1(x_range), 'r-', linewidth=2, label='KDE')
    axes[1].axvline(np.mean(f1_scores), color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[1].set_xlabel('Per-Class F1 Score', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title('Density Plot of Per-Class F1 Score', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    f1_plot_path = os.path.join(plot_dir, 'f1_distribution.png')
    plt.savefig(f1_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved F1 distribution plot to: {f1_plot_path}")
    plt.close()

    # Plot 3: Combined comparison
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    ax.hist(accuracies, bins=40, alpha=0.5, label='Accuracy', edgecolor='black')
    ax.hist(f1_scores, bins=40, alpha=0.5, label='F1 Score', edgecolor='black', color='orange')
    ax.set_xlabel('Score', fontsize=12)
    ax.set_ylabel('Number of Classes', fontsize=12)
    ax.set_title('Comparison of Per-Class Accuracy vs F1 Score', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    comparison_plot_path = os.path.join(plot_dir, 'accuracy_vs_f1_comparison.png')
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to: {comparison_plot_path}")
    plt.close()

    # Plot 4: Box plots
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    box_data = [accuracies, f1_scores]
    bp = ax.boxplot(box_data, labels=['Accuracy', 'F1 Score'], patch_artist=True)

    # Color the boxes
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Box Plot Comparison: Accuracy vs F1 Score', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    boxplot_path = os.path.join(plot_dir, 'boxplot_comparison.png')
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    print(f"Saved box plot to: {boxplot_path}")
    plt.close()

    # Plat 5: Accuracy-Label Size Correlation
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.scatter(sizes, accuracies, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Samples in Class', fontsize=12)
    ax.set_ylabel('Per-Class Accuracy', fontsize=12)
    ax.set_title('Correlation between Class Size and Accuracy', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    correlation_plot_path = os.path.join(plot_dir, 'accuracy_vs_class_size.png')
    plt.tight_layout()
    plt.savefig(correlation_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved accuracy vs class size plot to: {correlation_plot_path}")

    print(f"\nAnalysis complete!")
    print(f"Report saved to: {output_file}")
    print(f"Plots saved to: {plot_dir}/")


def analyze_multi_task(results, mappings=None, output_file=None):
    """Analyze multi-task model predictions

    Args:
        results: Prediction results dictionary
        mappings: Optional label mappings for decoding. If None, assumes labels are already decoded strings.
        output_file: Path to save the analysis report (currently not implemented for multi-task)
    """
    predictions = results['predictions']

    print("\n" + "="*60)
    print("MULTI-TASK MODEL ANALYSIS")
    print("="*60)

    print(f"\nOverall Statistics:")
    print(f"  Total samples: {results['num_samples']}")
    print(f"  Species Accuracy: {results['species_accuracy']:.4f}")
    print(f"  Genus Accuracy: {results['genus_accuracy']:.4f}")
    print(f"  Family Accuracy: {results['family_accuracy']:.4f}")

    # Set up decoders if mappings provided
    if mappings and mappings.get('type') == 'multi_task':
        id2family = {int(k): v for k, v in mappings['id2family'].items()}
        id2genus = {int(k): v for k, v in mappings['id2genus'].items()}
        id2species = {int(k): v for k, v in mappings['id2species'].items()}

        def decode_family(fam_id):
            if isinstance(fam_id, str):
                return fam_id  # Already decoded
            return id2family.get(fam_id, f"Unknown_{fam_id}")

        def decode_genus(gen_id):
            if isinstance(gen_id, str):
                return gen_id  # Already decoded
            return id2genus.get(gen_id, f"Unknown_{gen_id}")

        def decode_species(sp_id):
            if isinstance(sp_id, str):
                return sp_id  # Already decoded
            return id2species.get(sp_id, f"Unknown_{sp_id}")
    else:
        # No decoding needed, pass through
        def decode_family(fam):
            return fam
        def decode_genus(gen):
            return gen
        def decode_species(sp):
            return sp

    # Count correct predictions at each level
    species_correct = sum(1 for p in predictions if p['species_true'] == p['species_pred'])
    genus_correct = sum(1 for p in predictions if p['genus_true'] == p['genus_pred'])
    family_correct = sum(1 for p in predictions if p['family_true'] == p['family_pred'])

    print(f"\nCorrect Predictions:")
    print(f"  Species: {species_correct}/{len(predictions)}")
    print(f"  Genus: {genus_correct}/{len(predictions)}")
    print(f"  Family: {family_correct}/{len(predictions)}")

    # Hierarchical consistency analysis
    print("\n" + "-"*60)
    print("Hierarchical Consistency:")
    print("-"*60)

    all_correct = sum(1 for p in predictions
                     if p['species_true'] == p['species_pred']
                     and p['genus_true'] == p['genus_pred']
                     and p['family_true'] == p['family_pred'])

    genus_family_correct = sum(1 for p in predictions
                               if p['genus_true'] == p['genus_pred']
                               and p['family_true'] == p['family_pred'])

    only_family_correct = sum(1 for p in predictions
                              if p['family_true'] == p['family_pred']
                              and p['genus_true'] != p['genus_pred'])

    print(f"  All levels correct: {all_correct} ({all_correct/len(predictions)*100:.2f}%)")
    print(f"  Genus & Family correct: {genus_family_correct} ({genus_family_correct/len(predictions)*100:.2f}%)")
    print(f"  Only Family correct: {only_family_correct} ({only_family_correct/len(predictions)*100:.2f}%)")

    # Species error analysis
    print("\n" + "-"*60)
    print("Species-Level Error Analysis (Top 10):")
    print("-"*60)

    species_errors = Counter()
    for p in predictions:
        if p['species_true'] != p['species_pred']:
            species_errors[(p['species_true'], p['species_pred'])] += 1

    print(f"{'True Species':<30} {'Predicted Species':<30} {'Count':<10}")
    print("-"*70)
    for (true_sp, pred_sp), count in species_errors.most_common(10):
        true_decoded = decode_species(true_sp)
        pred_decoded = decode_species(pred_sp)
        true_display = true_decoded[:27] + "..." if len(true_decoded) > 30 else true_decoded
        pred_display = pred_decoded[:27] + "..." if len(pred_decoded) > 30 else pred_decoded
        print(f"{true_display:<30} {pred_display:<30} {count:<10}")

    # Genus error analysis
    print("\n" + "-"*60)
    print("Genus-Level Error Analysis (Top 10):")
    print("-"*60)

    genus_errors = Counter()
    for p in predictions:
        if p['genus_true'] != p['genus_pred']:
            genus_errors[(p['genus_true'], p['genus_pred'])] += 1

    print(f"{'True Genus':<25} {'Predicted Genus':<25} {'Count':<10}")
    print("-"*60)
    for (true_g, pred_g), count in genus_errors.most_common(10):
        true_decoded = decode_genus(true_g)
        pred_decoded = decode_genus(pred_g)
        print(f"{true_decoded:<25} {pred_decoded:<25} {count:<10}")

    # Family statistics
    print("\n" + "-"*60)
    print("Family-Level Statistics:")
    print("-"*60)

    family_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    for p in predictions:
        family = p['family_true']
        family_stats[family]['total'] += 1
        if p['family_true'] == p['family_pred']:
            family_stats[family]['correct'] += 1

    print(f"{'Family':<30} {'Total':<10} {'Correct':<10} {'Accuracy':<10}")
    print("-"*60)
    sorted_families = sorted(family_stats.items(), key=lambda x: x[1]['total'], reverse=True)
    for family, stats in sorted_families[:15]:
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        family_decoded = decode_family(family)
        family_display = family_decoded[:27] + "..." if len(family_decoded) > 30 else family_decoded
        print(f"{family_display:<30} {stats['total']:<10} {stats['correct']:<10} {acc:<10.4f}")

    # Show example errors
    print("\n" + "-"*60)
    print("Example Species Errors (First 5):")
    print("-"*60)

    species_errors_list = [p for p in predictions if p['species_true'] != p['species_pred']]
    for i, p in enumerate(species_errors_list[:5]):
        print(f"\n{i+1}. {p['image_path']}")
        true_fam = decode_family(p['family_true'])
        true_gen = decode_genus(p['genus_true'])
        true_sp = decode_species(p['species_true'])
        pred_fam = decode_family(p['family_pred'])
        pred_gen = decode_genus(p['genus_pred'])
        pred_sp = decode_species(p['species_pred'])
        print(f"   True:      {true_fam} / {true_gen} / {true_sp}")
        print(f"   Predicted: {pred_fam} / {pred_gen} / {pred_sp}")


def load_predictions_streaming(filepath, max_predictions=None):
    """
    Load predictions in a memory-efficient way by reading line by line
    and parsing incrementally.

    Args:
        filepath: Path to JSON file
        max_predictions: Optional limit on number of predictions to load

    Returns:
        Dictionary with metadata and predictions list
    """
    print(f"Loading predictions from {filepath} (memory-efficient mode)")

    # Try to use ijson for streaming if available
    try:
        import ijson

        with open(filepath, 'rb') as f:
            parser = ijson.items(f, '')
            results = next(parser)

            # If max_predictions is set, truncate the predictions list
            if max_predictions and 'predictions' in results:
                results['predictions'] = results['predictions'][:max_predictions]
                print(f"Limited to first {max_predictions} predictions")

            return results

    except ImportError:
        # Fallback: try to load with standard json but with better error handling
        print("Note: Install 'ijson' for better memory efficiency: pip install ijson")
        print("Attempting to load with standard JSON parser...")

        try:
            with open(filepath, 'r') as f:
                # Try reading in chunks
                chunk_size = 1024 * 1024 * 100  # 100MB chunks
                content = ""
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    content += chunk

                results = json.loads(content)

                # If max_predictions is set, truncate
                if max_predictions and 'predictions' in results:
                    results['predictions'] = results['predictions'][:max_predictions]
                    print(f"Limited to first {max_predictions} predictions")

                return results

        except MemoryError:
            print("\nERROR: File is too large to load into memory.")
            print("\nSolutions:")
            print("1. Install ijson for streaming support: pip install ijson")
            print("2. Use --max-predictions flag to analyze a subset")
            print("3. Split your predictions file into smaller chunks")
            sys.exit(1)


def analyze_prediction_file(prediction_file: str, max_predictions: int = None):

    """
        Analyze a single prediction file with optional memory-efficient loading

        Args:
            prediction_file: Path to the prediction JSON file
            max_predictions: Optional limit on number of predictions to analyze (for memory efficiency)
    """

    results = load_predictions_streaming(prediction_file, max_predictions)

    print(f"Loaded {results.get('num_samples', len(results.get('predictions', [])))} samples")

    # Try to load label mappings, auto-detect based on prediction file name
    mappings_file = prediction_file.replace('.json', '_mappings.json')

    if mappings_file and os.path.exists(mappings_file):
        print(f"Loading label mappings from {mappings_file}")
        try:
            with open(mappings_file, 'r') as f:
                mappings = json.load(f)
            print(f"Loaded mappings for {mappings.get('type', 'unknown')} model")
        except Exception as e:
            print(f"Warning: Could not load mappings file: {e}")
            print("Will attempt to analyze with raw labels (backwards compatibility)")
    else:
        print("No mappings file found. Assuming labels are already decoded (backwards compatibility)")

    # Auto-generate report name 
    base_name = os.path.splitext(os.path.basename(prediction_file))[0]
    save_report = f"{base_name}.txt"

    # Determine if single-task or multi-task
    is_multi_task = 'species_accuracy' in results

    # Run analysis
    if is_multi_task:
        analyze_multi_task(results, mappings, save_report)
    else:
        analyze_single_task(results, mappings, save_report)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze prediction results")
    parser.add_argument('--prediction_dir', type=str, required=True, help='Path to dir with prediction JSONs')
    # parser.add_argument('--save_report', type=str, default=None,
    #                    help='Save analysis report to file (default: auto-generated based on prediction file name)')
    parser.add_argument('--max-predictions', type=int, default=None,
                       help='Maximum number of predictions to analyze (for memory efficiency)')
    args = parser.parse_args()

    # Load predictions with memory-efficient method
    PREDICTION_FILES = [f for f in os.listdir(args.prediction_dir) if f.endswith('.json') and "mappings" not in f]

    if not PREDICTION_FILES:
        print(f"No prediction JSON files found in directory: {args.prediction_dir}")
        sys.exit(1)
    print(f"Found {len(PREDICTION_FILES)} prediction files in directory: {args.prediction_dir}")

    for prediction_file in PREDICTION_FILES:
        print(f"\n __MAIN__: Analyzing file: {prediction_file}")
        analyze_prediction_file(os.path.join(args.prediction_dir, prediction_file), args.max_predictions)

if __name__ == "__main__":
    main()
