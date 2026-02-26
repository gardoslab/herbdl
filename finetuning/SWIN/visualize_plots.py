#!/usr/bin/env python3
"""
Interactive visualization tool for comparing model checkpoint plots.
Supports SWIN Base, SWIN Large, SWIN V2 Base, and SWIN V2 Large models
trained on different label set sizes (15k, 21k, 50k).
"""

import streamlit as st
import os
from pathlib import Path
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Model Checkpoint Comparison",
    page_icon="📊",
    layout="wide"
)

# Constants
PLOTS_DIR = Path("./PLOTS")
PLOT_TYPES = [
    "accuracy_distribution.png",
    "accuracy_vs_class_size.png",
    "accuracy_vs_f1_comparison.png",
    "boxplot_comparison.png",
    "f1_distribution.png"
]

# Model configurations
MODEL_CONFIGS = {
    "SWIN_BASE": {
        "name": "SWIN Base",
        "15k": "SWIN_BASE_UNFROZEN_15K",
        "21k": "SWIN_BASE_UNFROZEN_21K_checkpoint-210888",
        "50k": "SWIN_BASE_UNFROZEN_50K_checkpoint-184527"
    },
    "SWIN_LARGE": {
        "name": "SWIN Large",
        "15k": "SWIN_LARGE_UNFROZEN_15K_checkpoint-215209",
        "21k": "SWIN_LARGE_UNFROZEN_21K_checkpoint-210876",
        "50k": "SWIN_LARGE_UNFROZEN_50K_checkpoint-298741"
    },
    "SWINV2_BASE": {
        "name": "SWIN V2 Base",
        "15k": "SWINV2_BASE_UNFROZEN_15K_checkpoint-682370",
        "21k": "SWINV2_BASE_UNFROZEN_21K_checkpoint-667755",
        "50k": "SWINV2_BASE_UNFROZEN_50K_checkpoint-492030"
    },
    "SWINV2_LARGE": {
        "name": "SWIN V2 Large",
        "15k": "SWINV2_LARGE_UNFROZEN_15K_checkpoint-377928",
        "21k": "SWINV2_LARGE_UNFROZEN_21K_checkpoint-351450",
        "50k": "SWINV2_LARGE_UNFROZEN_50K_checkpoint-281160"
    }
}

LABEL_SIZES = ["15k", "21k", "50k"]


def get_checkpoint_path(model_key, label_size):
    """Get the directory path for a specific model checkpoint."""
    checkpoint_name = MODEL_CONFIGS[model_key][label_size]
    return PLOTS_DIR / checkpoint_name


def plot_exists(model_key, label_size, plot_type):
    """Check if a plot exists for the given configuration."""
    checkpoint_path = get_checkpoint_path(model_key, label_size)
    plot_path = checkpoint_path / plot_type
    return plot_path.exists()


def load_plot(model_key, label_size, plot_type):
    """Load a plot image."""
    checkpoint_path = get_checkpoint_path(model_key, label_size)
    plot_path = checkpoint_path / plot_type
    if plot_path.exists():
        return Image.open(plot_path)
    return None


def load_metrics_csv(model_key, label_size):
    """Load the per-class metrics CSV for a checkpoint."""
    checkpoint_path = get_checkpoint_path(model_key, label_size)
    csv_path = checkpoint_path / "per_class_metrics.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def main():
    st.title("🌿 Model Checkpoint Comparison Dashboard")
    st.markdown("""
    Compare performance metrics across different SWIN model variants and label set sizes.
    - **Models**: SWIN Base, SWIN Large, SWIN V2 Base, SWIN V2 Large
    - **Label Sets**: 15k, 21k, 50k species
    """)

    # Sidebar controls
    st.sidebar.header("Visualization Settings")

    viz_mode = st.sidebar.radio(
        "Visualization Mode",
        ["Single Model Progression", "Side-by-Side Comparison", "Metrics Data Explorer"]
    )

    if viz_mode == "Single Model Progression":
        show_single_model_progression()
    elif viz_mode == "Side-by-Side Comparison":
        show_side_by_side_comparison()
    else:
        show_metrics_explorer()


def show_single_model_progression():
    """Show how metrics change as label size increases for a single model."""
    st.header("Single Model Progression")
    st.markdown("View how metrics change with increasing label set sizes (15k → 21k → 50k)")

    # Model selection
    model_key = st.selectbox(
        "Select Model",
        list(MODEL_CONFIGS.keys()),
        format_func=lambda x: MODEL_CONFIGS[x]["name"]
    )

    # Plot type selection
    plot_type = st.selectbox(
        "Select Plot Type",
        PLOT_TYPES,
        format_func=lambda x: x.replace("_", " ").replace(".png", "").title()
    )

    # Display mode
    display_mode = st.radio(
        "Display Mode",
        ["Overlay (side-by-side)", "Stacked (vertical)"]
    )

    st.markdown(f"### {MODEL_CONFIGS[model_key]['name']} - {plot_type.replace('_', ' ').replace('.png', '').title()}")

    if display_mode == "Overlay (side-by-side)":
        cols = st.columns(3)
        for idx, label_size in enumerate(LABEL_SIZES):
            with cols[idx]:
                st.markdown(f"**{label_size.upper()} Labels**")
                img = load_plot(model_key, label_size, plot_type)
                if img:
                    st.image(img, use_container_width=True)
                else:
                    st.warning(f"Plot not found")
    else:  # Stacked
        for label_size in LABEL_SIZES:
            st.markdown(f"#### {label_size.upper()} Labels")
            img = load_plot(model_key, label_size, plot_type)
            if img:
                st.image(img, use_container_width=True)
            else:
                st.warning(f"Plot not found")
            st.markdown("---")


def show_side_by_side_comparison():
    """Compare two different models side by side."""
    st.header("Side-by-Side Model Comparison")
    st.markdown("Compare different model architectures at the same label set sizes")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model 1")
        model_key_1 = st.selectbox(
            "Select First Model",
            list(MODEL_CONFIGS.keys()),
            format_func=lambda x: MODEL_CONFIGS[x]["name"],
            key="model1"
        )

    with col2:
        st.subheader("Model 2")
        model_key_2 = st.selectbox(
            "Select Second Model",
            list(MODEL_CONFIGS.keys()),
            format_func=lambda x: MODEL_CONFIGS[x]["name"],
            key="model2",
            index=1
        )

    # Plot type selection
    plot_type = st.selectbox(
        "Select Plot Type",
        PLOT_TYPES,
        format_func=lambda x: x.replace("_", " ").replace(".png", "").title(),
        key="plot_type_comparison"
    )

    # Show comparisons for each label size
    for label_size in LABEL_SIZES:
        st.markdown(f"### {label_size.upper()} Labels")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**{MODEL_CONFIGS[model_key_1]['name']}**")
            img1 = load_plot(model_key_1, label_size, plot_type)
            if img1:
                st.image(img1, use_container_width=True)
            else:
                st.warning("Plot not found")

        with col2:
            st.markdown(f"**{MODEL_CONFIGS[model_key_2]['name']}**")
            img2 = load_plot(model_key_2, label_size, plot_type)
            if img2:
                st.image(img2, use_container_width=True)
            else:
                st.warning("Plot not found")

        st.markdown("---")


def show_metrics_explorer():
    """Explore the underlying metrics data."""
    st.header("Metrics Data Explorer")
    st.markdown("Explore and compare the raw per-class metrics data")

    # Model and label size selection
    col1, col2 = st.columns(2)

    with col1:
        model_keys = st.multiselect(
            "Select Models",
            list(MODEL_CONFIGS.keys()),
            default=[list(MODEL_CONFIGS.keys())[0]],
            format_func=lambda x: MODEL_CONFIGS[x]["name"]
        )

    with col2:
        label_sizes = st.multiselect(
            "Select Label Sizes",
            LABEL_SIZES,
            default=LABEL_SIZES
        )

    if not model_keys or not label_sizes:
        st.info("Please select at least one model and one label size")
        return

    # Load and display data
    for model_key in model_keys:
        st.subheader(MODEL_CONFIGS[model_key]["name"])

        tabs = st.tabs(label_sizes)

        for idx, label_size in enumerate(label_sizes):
            with tabs[idx]:
                df = load_metrics_csv(model_key, label_size)
                if df is not None:
                    # Display summary statistics
                    st.markdown("**Summary Statistics**")
                    cols = st.columns(4)

                    if 'accuracy' in df.columns:
                        with cols[0]:
                            st.metric("Avg Accuracy", f"{df['accuracy'].mean():.3f}")
                    if 'f1' in df.columns:
                        with cols[1]:
                            st.metric("Avg F1", f"{df['f1'].mean():.3f}")
                    if 'precision' in df.columns:
                        with cols[2]:
                            st.metric("Avg Precision", f"{df['precision'].mean():.3f}")
                    if 'recall' in df.columns:
                        with cols[3]:
                            st.metric("Avg Recall", f"{df['recall'].mean():.3f}")

                    st.markdown("**Per-Class Metrics**")
                    st.dataframe(df, use_container_width=True, height=400)

                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label=f"Download {label_size} metrics CSV",
                        data=csv,
                        file_name=f"{model_key}_{label_size}_metrics.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning(f"Metrics data not found for {label_size}")

        st.markdown("---")


if __name__ == "__main__":
    main()
