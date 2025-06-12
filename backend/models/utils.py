import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_image(fig):
    """Encode a matplotlib figure to a base64 string."""
    img_io = BytesIO()
    fig.savefig(img_io, format='png')
    plt.close(fig)
    img_io.seek(0)
    return base64.b64encode(img_io.read()).decode('utf-8')

def encode_dataframe(df: pd.DataFrame, target_column: str):
    """
    Encodes string columns in a dataframe using LabelEncoder.
    Returns:
    - dataframe with encoded features and target
    - dict of encoders for features
    - label encoder for target
    """
    df = df.copy()
    encoders = {}

    for col in df.select_dtypes(include=['object']).columns:
        if col != target_column:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    target_encoder = LabelEncoder()
    df[target_column] = target_encoder.fit_transform(df[target_column])

    return df, encoders, target_encoder

def generate_histograms(df: pd.DataFrame):
    """Generate histograms for all columns and return dict of base64 images."""
    histogram_imgs = {}
    for col in df.columns:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Histogram of {col}")
        histogram_imgs[col] = encode_image(fig)
    return histogram_imgs

def generate_pie_chart(series, title="Target Distribution"):
    """Generate a pie chart for a target series."""
    fig, ax = plt.subplots()
    counts = series.value_counts()
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%")
    ax.set_title(title)
    return encode_image(fig)
