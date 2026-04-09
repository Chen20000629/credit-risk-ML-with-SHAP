import os
import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from preprocess import load_data, preprocess, split_data
from model import train_models, evaluate

OUTPUT_DIR = "../outputs"

def save_roc_curve(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.savefig(f"{OUTPUT_DIR}/roc_curve.png")
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data("../data/credit.csv")
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    lr, rf = train_models(X_train, y_train)

    print("\n=== Logistic Regression ===")
    evaluate(lr, X_test, y_test)

    print("\n=== Random Forest ===")
    evaluate(rf, X_test, y_test)

    # =========================
    # SHAP
    # =========================
    explainer = shap.Explainer(rf, X_train)
    shap_values = explainer(X_test)

    # summary plot
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(f"{OUTPUT_DIR}/shap_summary.png", bbox_inches="tight")
    plt.close()

    # waterfall（取第一筆）
    shap.plots.waterfall(shap_values[0], show=False)
    plt.savefig(f"{OUTPUT_DIR}/shap_waterfall.png", bbox_inches="tight")
    plt.close()

    # ROC curve
    save_roc_curve(rf, X_test, y_test)

    print("\n All plots saved in outputs/")

if __name__ == "__main__":
    main()