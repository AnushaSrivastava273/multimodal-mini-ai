# Multimodal Mini AI  

A lightweight multimodal AI application that combines **image captioning** and **text sentiment analysis** in a simple Streamlit interface. This project demonstrates how to build and deploy a multimodal AI system using **Hugging Face Transformers** and **Streamlit**.  

🔗 **Live Demo:** [Click here](https://anushasrivastava273-multimodal-mini-ai-app-ob9qkw.streamlit.app/)  

---

## ✨ Features  
- 🖼️ **Image Captioning:** Upload an image, and the model generates a descriptive caption using [BLIP (Salesforce/blip-image-captioning-base)](https://huggingface.co/Salesforce/blip-image-captioning-base).  
- 📝 **Sentiment Analysis:** Enter any text and get real-time positive/negative sentiment prediction using [DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).  
- 🌐 **Deployed on Streamlit Cloud** for instant access without local setup.  
- ⚡ **Lightweight & beginner-friendly** project that integrates both vision and NLP.  

---

## 🛠️ Tech Stack  
- **Frontend:** [Streamlit](https://streamlit.io/)  
- **Models:** Hugging Face Transformers  
  - Image Captioning: `Salesforce/blip-image-captioning-base`  
  - Sentiment Analysis: `distilbert-base-uncased-finetuned-sst-2-english`  
- **Backend:** Python  
- **Other Libraries:**  
  - `torch`  
  - `pandas`, `numpy`  
  - `Pillow` for image handling  

---

## 🚀 How to Run Locally  

1. Clone the repository:  
   ```bash
   git clone https://github.com/AnushaSrivastava273/multimodal-mini-ai.git
   cd multimodal-mini-ai
---
Create and activate a virtual environment:

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app.py
