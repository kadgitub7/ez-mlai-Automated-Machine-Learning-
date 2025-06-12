import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.decomposition import PCA

nb_model = None
nb_scaler = None
nb_features = None

def encode_image(fig):
    img_io = BytesIO()
    fig.savefig(img_io, format='png')
    plt.close(fig)
    img_io.seek(0)
    return base64.b64encode(img_io.read()).decode('utf-8')

def run_naive_bayes(file_stream, target_column):
    df = pd.read_csv(file_stream)

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found."}

    encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        if col != target_column:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    target_encoder = LabelEncoder()
    df[target_column] = target_encoder.fit_transform(df[target_column])

    X = df.drop(columns=[target_column])
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    y_proba = nb.predict_proba(X_test)[:, 1] if len(np.unique(y)) == 2 else None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "auc_roc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        "average_precision": average_precision_score(y_test, y_proba) if y_proba is not None else None
    }

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix")
    confusion_matrix_img = encode_image(fig)

    fig, ax = plt.subplots()
    ax.scatter(range(len(y_test)), y_test, label='Actual', alpha=0.5)
    ax.scatter(range(len(y_pred)), y_pred, label='Predicted', alpha=0.5)
    ax.legend()
    ax.set_title("Predicted vs Actual")
    predicted_vs_actual_img = encode_image(fig)

    roc_curve_img = None
    precision_recall_curve_img = None
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_title("ROC Curve")
        roc_curve_img = encode_image(fig)

        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        fig, ax = plt.subplots()
        ax.plot(recall, precision)
        ax.set_title("Precision-Recall Curve")
        precision_recall_curve_img = encode_image(fig)

    histogram_imgs = {}
    for col in X.columns:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Histogram of {col}")
        histogram_imgs[col] = encode_image(fig)

    fig, ax = plt.subplots()
    y_counts = df[target_column].value_counts()
    ax.pie(y_counts, labels=y_counts.index, autopct="%1.1f%%")
    ax.set_title("Target Variable Distribution")
    target_distribution_img = encode_image(fig)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca["target"] = y.values
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="target", palette="viridis", ax=ax)
    ax.set_title("PCA Visualization")
    pca_img = encode_image(fig)

    from models.registry import model_registry
    model_registry["naive_bayes"]["model"] = nb
    model_registry["naive_bayes"]["scaler"] = scaler
    model_registry["naive_bayes"]["features"] = list(X.columns)
    model_registry["naive_bayes"]["encoders"] = encoders
    model_registry["naive_bayes"]["target_encoder"] = target_encoder
    nb_features = list(X.columns)

    return {
        "metrics": metrics,
        "confusion_matrix": confusion_matrix_img,
        "predicted_vs_actual": predicted_vs_actual_img,
        "roc_curve": roc_curve_img,
        "precision_recall_curve": precision_recall_curve_img,
        "histograms": histogram_imgs,
        "target_distribution": target_distribution_img,
        "pca_plot": pca_img,
        "feature_names": nb_features,
        "python_code": open(__file__).read()
    }
