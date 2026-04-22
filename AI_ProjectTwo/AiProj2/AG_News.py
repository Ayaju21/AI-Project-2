import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]  # 0..3

# Control whether to display plots (for "Generate and display graph(s)" requirement)
SHOW_PLOTS = True


# 1) Text Preprocessing

def clean_text(text: str) -> str:
    """
    Basic cleaning:
    - lowercase
    - remove urls
    - keep letters/numbers/spaces
    - collapse extra spaces
    """
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)               # remove punctuation/special chars
    text = re.sub(r"\s+", " ", text).strip()               # collapse spaces
    return text


# 2) Load Dataset (AG News)
def load_ag_news_dataframe() -> pd.DataFrame:
    """
    Loads AG News from HuggingFace and returns a single DataFrame
    with columns: text, label, split
    """
    ds = load_dataset("ag_news")  # has train/test splits
    train_df = pd.DataFrame(ds["train"])
    test_df = pd.DataFrame(ds["test"])
    train_df["split"] = "train"
    test_df["split"] = "test"
    df = pd.concat([train_df, test_df], ignore_index=True)

    # Clean text
    df["text"] = df["text"].astype(str).apply(clean_text)
    return df


# 3) TF-IDF Feature Extraction
def build_vectorizer():
    """
    TF-IDF Vectorizer parameters (mention in report):
    - ngram_range: (1,2) -> unigrams + bigrams
    - max_features: limit vocabulary for speed/memory
    - min_df: ignore very rare terms
    - max_df: ignore very common terms
    """
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        min_df=2,
        max_df=0.9
    )


# 4) Train + Evaluate Helper

def evaluate_model(model, X_test, y_test, model_name: str, output_dir: str, show_plots: bool = True):
    """
    Evaluates a model:
    - Returns dict of metrics (accuracy, precision, recall, f1) using weighted average
    - Saves + displays confusion matrix plot
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    # Weighted metrics
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    print("\n" + "=" * 70)
    print(f"{model_name} ")
    print("=" * 70)
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)

    #  Create a figure/axes explicitly to avoid duplicate figures
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()

    cm_path = os.path.join(output_dir, f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png")
    fig.savefig(cm_path, dpi=200)

    #  Display plot if requested
    if show_plots:
        plt.show()

    plt.close(fig)

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision (weighted)": prec,
        "Recall (weighted)": rec,
        "F1-score (weighted)": f1,
        "ConfusionMatrixPath": cm_path
    }


# 5) Graph: Metrics Comparison
def plot_metrics_comparison(results_df: pd.DataFrame, output_dir: str, show_plots: bool = True):
    """
    Bar chart comparing Accuracy/Precision/Recall/F1 between models
    Saves + displays the plot.
    """
    metrics = ["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1-score (weighted)"]
    plot_df = results_df.set_index("Model")[metrics]

    ax = plot_df.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Model Performance Comparison (TF-IDF)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", rotation=0)

    fig = ax.get_figure()
    fig.tight_layout()

    out_path = os.path.join(output_dir, "metrics_comparison.png")
    fig.savefig(out_path, dpi=200)

    if show_plots:
        plt.show()

    plt.close(fig)
    return out_path


# 6) Decision Tree Visualization
def save_decision_tree_plot(tree_model, feature_names, output_dir: str, show_plots: bool = True):
    """
    Saves + displays a visualization of the decision tree.
    Since we train the tree with max_depth=3, what you see is the actual tree.
    """
    fig, ax = plt.subplots(figsize=(18, 10))
    plot_tree(
        tree_model,
        feature_names=feature_names,
        class_names=LABEL_NAMES,
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax
    )
    ax.set_title("Decision Tree")
    fig.tight_layout()

    out_path = os.path.join(output_dir, "decision_tree_plot.png")
    fig.savefig(out_path, dpi=200)

    if show_plots:
        plt.show()

    plt.close(fig)
    return out_path


# 7) Main Pipeline
def main():
    output_dir = "project2_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = load_ag_news_dataframe()

    # Use HuggingFace official splits
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    X_train_full = train_df["text"].values
    y_train_full = train_df["label"].values

    X_test = test_df["text"].values
    y_test = test_df["label"].values

    # Validation split from train
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.1,
        random_state=RANDOM_STATE,
        stratify=y_train_full
    )

    # TF-IDF
    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    # Model 1: Logistic Regression
    lr = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    )
    lr.fit(X_train_vec, y_train)

    # Validation check
    val_pred_lr = lr.predict(X_val_vec)
    val_acc_lr = accuracy_score(y_val, val_pred_lr)
    print(f"\nLogistic Regression Validation Accuracy: {val_acc_lr:.4f}")

    # Model 2: Decision Tree
    # 
    dt = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=5
    )
    dt.fit(X_train_vec, y_train)

    val_pred_dt = dt.predict(X_val_vec)
    val_acc_dt = accuracy_score(y_val, val_pred_dt)
    print(f"Decision Tree Validation Accuracy: {val_acc_dt:.4f}")

    # Evaluate on TEST set
    results = []
    results.append(evaluate_model(lr, X_test_vec, y_test, "Logistic Regression", output_dir, show_plots=SHOW_PLOTS))
    results.append(evaluate_model(dt, X_test_vec, y_test, "Decision Tree", output_dir, show_plots=SHOW_PLOTS))

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("Final Results (Test Set)")
    print("=" * 70)
    print(results_df[["Model", "Accuracy", "Precision (weighted)", "Recall (weighted)", "F1-score (weighted)"]])

    # Save results to CSV
    csv_path = os.path.join(output_dir, "results_summary.csv")
    results_df.to_csv(csv_path, index=False)

    # Metrics comparison plot (save + display)
    metrics_plot_path = plot_metrics_comparison(results_df, output_dir, show_plots=SHOW_PLOTS)

    # Decision tree plot (save + display)
    feature_names = vectorizer.get_feature_names_out()
    tree_plot_path = save_decision_tree_plot(dt, feature_names, output_dir, show_plots=SHOW_PLOTS)

    # Save parameters for report
    params_path = os.path.join(output_dir, "model_params.txt")
    with open(params_path, "w", encoding="utf-8") as f:
        f.write("TF-IDF params:\n")
        f.write(str(vectorizer.get_params()) + "\n\n")
        f.write("Logistic Regression params:\n")
        f.write(str(lr.get_params()) + "\n\n")
        f.write("Decision Tree params:\n")
        f.write(str(dt.get_params()) + "\n")

    print("\nSaved files:")
    print(f"- Results CSV: {csv_path}")
    print(f"- Metrics comparison plot: {metrics_plot_path}")
    print(f"- Decision Tree plot: {tree_plot_path}")
    print(f"- Model params: {params_path}")
    print(f"- Confusion matrices: {results_df['ConfusionMatrixPath'].tolist()}")
    print("\nDone ")


if __name__ == "__main__":
    main()
