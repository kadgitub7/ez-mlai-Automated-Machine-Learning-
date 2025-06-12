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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA

lr_model = None
lr_scaler = None
lr_features = None

def encode_image(fig):
    img_io = BytesIO()
    fig.savefig(img_io, format='png')
    plt.close(fig)
    img_io.seek(0)
    return base64.b64encode(img_io.read()).decode('utf-8')

def run_linear_regression(file_stream, target_column):
    df = pd.read_csv(file_stream)

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found."}

    # Encode categorical features
    encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        if col != target_column:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    X = df.drop(columns=[target_column])
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    # Metrics
    metrics = {
        "mean_squared_error": mean_squared_error(y_test, y_pred),
        "mean_absolute_error": mean_absolute_error(y_test, y_pred),
        "r2_score": r2_score(y_test, y_pred)
    }

    # Predicted vs Actual Scatter Plot
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")
    predicted_vs_actual_img = encode_image(fig)

    # Residual Plot
    residuals = y_test - y_pred
    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_title("Residual Plot")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    residual_plot_img = encode_image(fig)

    # Feature histograms
    histogram_imgs = {}
    for col in X.columns:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Histogram of {col}")
        histogram_imgs[col] = encode_image(fig)

    # Target Distribution Histogram
    fig, ax = plt.subplots()
    sns.histplot(y, kde=True, ax=ax)
    ax.set_title("Target Variable Distribution")
    target_distribution_img = encode_image(fig)

    # PCA Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots()
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
    ax.set_title("PCA of Features Colored by Target")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    pca_img = encode_image(fig)

    from models.registry import model_registry
    model_registry["linear_regression"]["model"] = lr
    model_registry["linear_regression"]["scaler"] = scaler
    model_registry["linear_regression"]["features"] = list(X.columns)
    model_registry["linear_regression"]["encoders"] = encoders
    
    lr_features = list(X.columns)

    return {
        "metrics": metrics,
        "predicted_vs_actual": predicted_vs_actual_img,
        "residual_plot": residual_plot_img,
        "histograms": histogram_imgs,
        "target_distribution": target_distribution_img,
        "pca_plot": pca_img,
        "feature_names": lr_features,
        "python_code": open(__file__).read()
    }
