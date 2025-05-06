import streamlit as st
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torchvision import models, transforms
from PIL import Image
import plotly.express as px

# Define constants
SKIN_TYPE_CLASSES = ['dry', 'normal', 'oily']
SKIN_ISSUE_CLASSES = ['acne', 'bags', 'redness', 'wrinkles', 'spots', 'scar']

class SkinClassificationModel(nn.Module):
    """
    A PyTorch model wrapper that supports different base CNN architectures (ResNet, EfficientNet)
    for skin type and skin issue classification tasks.
    """
    def __init__(self, num_classes, model_name='resnet18', use_pretrained=True):
        super(SkinClassificationModel, self).__init__()
        if model_name == 'resnet18':
            self.base_model = models.resnet18(weights='IMAGENET1K_V1' if use_pretrained else None)
            num_ftrs = self.base_model.fc.in_features
        elif model_name == 'resnet50':
            self.base_model = models.resnet50(weights='IMAGENET1K_V1' if use_pretrained else None)
            num_ftrs = self.base_model.fc.in_features
        elif model_name == 'efficientnet_b0':
            self.base_model = models.efficientnet_b0(weights='IMAGENET1K_V1' if use_pretrained else None)
            num_ftrs = self.base_model.classifier[1].in_features
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        if model_name.startswith('resnet'):
            self.base_model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, num_classes))
        elif model_name.startswith('efficientnet'):
            self.base_model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, num_classes))

    def forward(self, x):
        """Forward pass through the model."""
        return self.base_model(x)

@st.cache_resource
def load_data():
    """
    Loads and preprocesses skincare product datasets.
    Returns:
        Tuple of (exp_skincare, skincare) DataFrames
    """
    exp_skincare = pd.read_csv('export_skincare_recc.csv')
    skincare = pd.read_csv('skincare_recc.csv')

    exp_skincare['skintype'] = exp_skincare['skintype'].str.lower()
    exp_skincare['notable_effects'] = exp_skincare['notable_effects'].fillna("").str.lower()
    skincare['Skin_Type'] = skincare['Skin_Type'].str.lower()

    return exp_skincare, skincare

@st.cache_resource
def load_models():
    """
    Loads trained PyTorch models for skin type and skin issue classification.
    Returns:
        Tuple of (skin_type_model, skin_issues_model, device)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skin_type_model = SkinClassificationModel(num_classes=len(SKIN_TYPE_CLASSES), model_name='efficientnet_b0')
    skin_issues_model = SkinClassificationModel(num_classes=len(SKIN_ISSUE_CLASSES), model_name='resnet50')

    if os.path.exists('Apr_29_models/skin_type_model.pth'):
        skin_type_model.load_state_dict(torch.load('Apr_29_models/skin_type_model.pth', map_location=device))
    else:
        st.warning("Skin type model not found.")
    if os.path.exists('Apr_30_models/skin_issues_model.pth'):
        skin_issues_model.load_state_dict(torch.load('Apr_30_models/skin_issues_model.pth', map_location=device))
    else:
        st.warning("Skin issues model not found.")

    return skin_type_model.to(device), skin_issues_model.to(device), device

def predict_skin_conditions(skin_type_model, skin_issues_model, image, device='cpu'):
    """
    Predicts the skin type and top skin issues from an input image.
    Args:
        skin_type_model: Model to predict skin type
        skin_issues_model: Model to predict skin issues
        image: Input image (PIL or ndarray)
        device: Torch device (CPU or CUDA)
    Returns:
        Tuple of (predicted_skin_type, top_issues, type_pred_probs, issue_pred_probs)
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    input_tensor = transform(image).unsqueeze(0).to(device)

    skin_type_model.eval()
    skin_issues_model.eval()

    with torch.no_grad():
        type_outputs = skin_type_model(input_tensor)
        type_probs = torch.nn.functional.softmax(type_outputs, dim=1)
        type_pred_probs = type_probs[0].cpu().numpy()
        type_pred = torch.argmax(type_probs, dim=1).item()

        issue_outputs = skin_issues_model(input_tensor)
        issue_probs = torch.nn.functional.softmax(issue_outputs, dim=1)
        issue_pred_probs = issue_probs[0].cpu().numpy()

        top3_issues_values, top3_issues_indices = torch.topk(issue_probs, 3)

    predicted_skin_type = SKIN_TYPE_CLASSES[type_pred]

    top_issues = []
    for i in range(3):
        idx = top3_issues_indices[0][i].item()
        prob = top3_issues_values[0][i].item()
        if prob > 0.1:
            top_issues.append((SKIN_ISSUE_CLASSES[idx], prob))

    return predicted_skin_type, top_issues, type_pred_probs, issue_pred_probs

def recommend_products(skin_type, skin_issues, exp_skincare, skincare, top_n=5):
    """
    Recommends skincare products based on detected skin type and issues.
    Args:
        skin_type: Predicted skin type
        skin_issues: List of detected skin issues (tuples)
        exp_skincare: Expanded skincare DataFrame
        skincare: Backup skincare DataFrame
        top_n: Number of recommendations
    Returns:
        List of recommended product dictionaries
    """
    recs = []
    for _, row in exp_skincare.iterrows():
        skin_type_match = skin_type in row['skintype']
        issues_match = any(issue in row['notable_effects'] for issue, _ in skin_issues)

        if skin_type_match and issues_match:
            recs.append({
                "product_name": row['product_name'],
                "brand": row['brand'],
                "price": row['price'],
                "link": row['product_href'],
                "description": row['description'],
                "picture": row['picture_src'],
                "priority": 1
            })

    if len(recs) < top_n:
        for _, row in exp_skincare.iterrows():
            skin_type_match = skin_type in row['skintype']
            issues_match = any(issue in row['notable_effects'] for issue, _ in skin_issues)

            if (skin_type_match or issues_match) and not any(rec['product_name'] == row['product_name'] for rec in recs):
                recs.append({
                    "product_name": row['product_name'],
                    "brand": row['brand'],
                    "price": row['price'],
                    "link": row['product_href'],
                    "description": row['description'],
                    "picture": row['picture_src'],
                    "priority": 2
                })

    if len(recs) < top_n:
        for _, row in skincare.iterrows():
            if skin_type in row['Skin_Type'] or row['Skin_Type'] == 'all':
                if not any(rec['product_name'] == row['Title'].strip() for rec in recs):
                    recs.append({
                        "product_name": row['Title'].strip(),
                        "brand": row['Brand'],
                        "price": row['Price'],
                        "link": row['Link'],
                        "description": f"Category: {row['Category']}",
                        "picture": None,
                        "priority": 3
                    })

    seen = set()
    unique_recs = []
    for rec in recs:
        if rec['product_name'] not in seen:
            unique_recs.append(rec)
            seen.add(rec['product_name'])

    unique_recs = sorted(unique_recs, key=lambda x: x['priority'])

    return unique_recs[:top_n]

def plot_skin_type_probs(probs):
    """
    Plots predicted probabilities for skin types using Plotly.
    Args:
        probs: List of probabilities for each class
    Returns:
        Plotly bar chart
    """
    fig = px.bar(x=SKIN_TYPE_CLASSES, y=probs, labels={'x': 'Skin Type', 'y': 'Probability'}, title="Skin Type Probability")
    return fig

def plot_skin_issues_probs(probs):
    """
    Plots predicted probabilities for skin issues using Plotly.
    Args:
        probs: List of probabilities for each skin issue
    Returns:
        Plotly bar chart
    """
    fig = px.bar(x=SKIN_ISSUE_CLASSES, y=probs, labels={'x': 'Skin Issue', 'y': 'Probability'}, title="Skin Issues Probability")
    return fig

def main():
    """
    Main Streamlit app logic that handles image input, model prediction,
    visualization, and recommendation display.
    """
    st.set_page_config(page_title="Skin Analyzer", layout="wide")
    st.title("🔬 Skin Condition Analyzer")

    skin_type_model, skin_issues_model, device = load_models()
    exp_skincare, skincare = load_data()

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    camera_image = st.camera_input("Or Take a Photo") if st.checkbox("Use Camera") else None

    st.markdown("### Or try a sample image:")
    sample_cols = st.columns(4)
    sample_images = {
        "Dry Skin": "sample_images/dry_skin.jpg",
        "Oily Skin": "sample_images/oily_skin.jpg",
        "Acne": "sample_images/acne.jpg",
        "Normal Skin": "sample_images/normal_skin.jpg"
    }

    selected_sample = None
    for idx, (name, path) in enumerate(sample_images.items()):
        with sample_cols[idx % 4]:
            if os.path.exists(path):
                st.image(path, caption=name, width=150)
                if st.button(f"Use {name}"):
                    selected_sample = path

    image = None
    if selected_sample:
        image = Image.open(selected_sample).convert('RGB')
    elif uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
    elif camera_image:
        image = Image.open(camera_image).convert('RGB')

    if image:
        st.image(image, caption="Input Image", use_column_width=True)

        with st.spinner("Analyzing..."):
            predicted_skin_type, top_issues, type_probs, issue_probs = predict_skin_conditions(
                skin_type_model, skin_issues_model, image, device
            )

        tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Skin Type", "Skin Issues", "Recommendations"])

        with tab1:
            st.header("Summary")
            st.subheader(f"Detected Skin Type: **{predicted_skin_type.upper()}**")
            if top_issues:
                for issue, prob in top_issues:
                    st.write(f"- {issue.capitalize()} ({prob:.1%})")
            else:
                st.success("No major skin issues detected!")

        with tab2:
            st.header("Skin Type Analysis")
            st.plotly_chart(plot_skin_type_probs(type_probs))

        with tab3:
            st.header("Skin Issues Analysis")
            st.plotly_chart(plot_skin_issues_probs(issue_probs))

        with tab4:
            st.header("Personalized Product Recommendations")
            recommendations = recommend_products(predicted_skin_type, top_issues, exp_skincare, skincare)
            df_recc = pd.DataFrame(recommendations)
            st.dataframe(df_recc)

    else:
        st.info("Upload, capture, or select a sample image to start.")

if __name__ == "__main__":
    main()