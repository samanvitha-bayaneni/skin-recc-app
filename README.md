### 📄 `README.md`

```markdown
# 🧴 Skin Analyzer App

A Streamlit-powered web app to analyze facial skin from an image:
- 🧠 Predict your skin type (dry, normal, oily)
- 🔍 Detect skin issues (acne, redness, spots, etc.)
- 🛍️ Recommend personalized skincare products

## 🚀 Live App

👉 Try it now: https://skin-recc-app.streamlit.app/

## Medium Post: https://medium.com/@samanvitha9/skinvision-leveraging-deep-learning-for-skin-condition-analysis-6860d9c5add2

## 📁 Project Structure

├── app.py                        # Main Streamlit app
├── requirements.txt             # Python dependencies
├── export\_skincare\_recc.csv     # Skincare product dataset (detailed)
├── skincare\_recc.csv            # Backup product dataset
├── sample\_images/               # Sample test images
│   ├── dry\_skin.jpg
│   ├── oily\_skin.jpg
│   └── ...
├── Apr\_29\_models/
│   └── skin\_type\_model.pth      # Local model for skin type
├── Apr\_30\_models/
│   └── (downloaded at runtime)  # Large model auto-downloaded from GDrive
└── .streamlit/
└── config.toml              # UI config (optional)

## 🧠 Features

- 📷 Upload, capture, or choose a sample image
- 🧬 CNN-based skin classification (EfficientNet / ResNet)
- 📊 Interactive visualizations using Plotly
- 🛒 Smart skincare recommendations using CSV data

## ☁️ Notes on Deployment

* The `skin_issues_model.pth` is not in the repo due to GitHub size limits.
* It is automatically downloaded at runtime from Google Drive using `gdown`.

## 📃 License

MIT License. © 2025 

## 🙌 Acknowledgments

* Built with PyTorch + Streamlit
* Google Drive used for large model storage
* Product data processed from open skincare listings
