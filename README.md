 
# CrisisWatch AI

## AIM 

The main aim of this project is to support individuals experiencing suicidal thoughts, mental health issues, or substance use challenges and try to seek help on social medias. Using advanced natural language processing and machine learning techniques, the system analysis posts in real time and provides actionable insights to the mental health services and professionals. This enables mental health organizations to launch targeted awareness campaigns and provide timely support and intervention to those peoples.

## OVERVIEW

CrisisWatch AI is an crisis monitoring system designed to detect and analyze crisis-related content on Reddit. By collecting and processing posts from subreddits focused on mental health, suicide thoughts, and substance use, the system performs risk classification, sentimental analysis, user behavior tracking, and geolocation mapping.

The data is visualized through an interactive dashboard that enables mental health professionals and researchers to monitor trends, identify at-risk user, and make data-driven decisions for outreach and support. The solution is modular, scalable, and focused on creating real-world impact in the field of mental health using Artificial Intelligence.


## Features

- Reddit data collection using PRAW
- Text preprocessing and keyword filtering
- Mental health risk classification using a fine-tuned DistilBERT model
- Sentiment analysis using VADER
- Hybrid geolocation extraction using spaCy, OpenAI GPT, and Nominatim
- User behavior analysis
- SQLite database for structured storage
- Data visualization and exploratory analysis
- Global heatmap generation
- Interactive Dashboard

---

# System Architecture

 <img width="1070" height="630" alt="Screenshot 2026-07-03 223243" src="https://github.com/user-attachments/assets/2a50597b-9a35-4b4a-89b8-877205b61369" />


---

# Technologies Used

| **Category** | **Technologies Used** |
|--------------|-----------------------|
| Programming Language | Python |
| Reddit Data Collection | PRAW (Reddit API) |
| Deep Learning Framework | PyTorch |
| Risk Classification | Fine-tuned DistilBERT (Hugging Face Transformers) |
| Sentiment Analysis | VADER Sentiment |
| Named Entity Recognition | spaCy |
| Contextual Location Inference | OpenAI (via LangChain) |
| Geocoding | Nominatim, GeoPy |
| Data Processing | Pandas, NumPy |
| Database | SQLite |
| Data Visualization | Matplotlib, WordCloud, Folium |
| Dash Board | Plotly |

---

# Dataset

Since no publicly available dataset directly matched the project's three-level mental health risk taxonomy (**High Risk**, **Moderate Risk**, and **Low Risk**), a custom dataset was constructed by merging, cleaning, relabeling, and standardizing two publicly available datasets.

## Dataset Sources

### 1. Sentiment Analysis for Mental Health

https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health

### 2. Suicidal Tweet Detection Dataset

https://www.kaggle.com/datasets/aunanya875/suicidal-tweet-detection-dataset

---

## Dataset Construction

The custom dataset was created through the following preprocessing pipeline:

- Combined both source datasets
- Removed missing and invalid records
- Removed duplicate samples
- Text normalization and standardization
- Unified label schema
- Mapped the original labels into three behavioral health risk categories

---

## Dataset Statistics

**Total Samples:** **47,894**

**Number of Classes:** **3**

| **Risk Level** | **Description** | **Samples** |
|----------------|-----------------|------------:|
| 🔴 High Risk | Explicit suicidal ideation, suicide planning, self-harm intent, or immediate crisis | 12,064 |
| 🟡 Moderate Risk | Depression, anxiety, emotional distress, stress, or substance abuse without explicit suicidal intent | 19,487 |
| 🟢 Low Risk | General discussions, daily life experiences, emotionally neutral, or non-crisis mental health conversations | 16,343 |

---

# Risk Classification Model

A **DistilBERT-based transformer model** was fine-tuned to classify Reddit posts into three behavioral health risk categories.

## Model Configuration

| Parameter | Value |
|------------|-------|
| Base Model | distilbert-base-uncased |
| Framework | Hugging Face Transformers |
| Deep Learning Library | PyTorch |
| Task | Multi-class Text Classification |
| Learning Rate | 2e-5 |
| Batch Size | 8 |
| Epochs | 3 |
| Optimizer | AdamW |
| Weight Decay | 0.01 |

---

## Model Performance

| Epoch | Training Loss | Validation Loss | Accuracy | Weighted F1 |
|-------:|--------------:|----------------:|----------:|------------:|
| 1 | 0.2657 | 0.4282 | 85.45% | 85.32% |
| 2 | 0.2367 | 0.5043 | **85.80%** | **85.71%** |
| 3 | 0.1542 | 0.6890 | 85.79% | 85.69% |

**Best Validation Accuracy:** **85.80%**

**Best Weighted F1 Score:** **85.71%**

---

# Geolocation Extraction

One of the major challenges in behavioral health monitoring is accurately identifying a user's location from unstructured Reddit posts.
 
1. **spaCy Named Entity Recognition (NER)** identifies posts containing potential location entities.
2. **Nominatim** converts inferred locations into latitude and longitude coordinates.


## DEMO
Due to cost compute constraints, we were unable to deploy the complete version of the application that includes the full post analysis pipeline(post extraction, model inference, and geolocation).
However, we have deployed the **interactive dashboard**, which showcases visualizations based on posts analysed and stored in the database.

Explore the deployed dashboard here

**Live Demo:** [CrisisWatchAI](https://crisis-watch-demo-gqib.onrender.com/)

## Prerequisites
1. **Create a Reddit API Key**:
   - Go to [Reddit Apps](https://www.reddit.com/prefs/apps)
   - Click **Create an App**
   - Select **script** as the app type
   - Note down the `client_id` and `client_secret`

2. **Set Up Environment Variables**:
   - Change the `.env` file in the project directory.
   - Add the following content:
     ```ini
     CLIENT_ID=your_reddit_client_id
     CLIENT_SECRET=your_reddit_client_secret
     USER_AGENT=your_app_name
     USER_NAME=your_username
     PASSWORD=your_password
     ```

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/gokulan006/CrisisWatch-AI.git
   cd CrisisWatch-AI
   ```
2. Create a virtual environment:
   ```sh
   python -m venv venv
   venv\Scripts\activate   # For Windows
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
1. **Run the main application**:
   ```sh
   python main.py
   ```
2. **View CrisisWatch AI**:
   - Open `http://localhost:5000/` in a browser.
   - Press `Launch dashboard` in the site.

## File Structure
```
.
├── main.py                     # Main application file
├── dash_app.py                 # Plotly Dashboard Python file
├── posts.db                    # SQLite database for storing posts
├── risk_analysis.csv           # Risk CLassification Dataset
├── requirements.txt            # Dependencies
├── templates/index.html        # HTML code for Home Page
├── static/styles.css           # Styling CSS File for Home Page
├── assets/style.css            # Styling CSS File for Dashboard Page
└── .env                        # Environment variables     
```

## Use of Large Language Models (LLMs)
- **Custom Keyword Lexicon Creation**: Chat-Gpt was used to generate a domain-specific keyword lexicon for filtering posts related to mental health, suicide, and substance use with including coded language.
- **Frontend Styling Assistance**: Claude also assited in designing and refining the homepage animatic styling, helping to choose appropriate color transformation and logos for a clean and user-friendly interface.

## Contributors

- [Gokulan M](https://github.com/gokulan006) – Model Training, Backend Development, Dashboard, Frontend Development
 
## Acknowledgments

- [Reddit](https://www.reddit.com/dev/api) for API access
- [Hugging Face Transformers](https://huggingface.co/) for the DistilBERT model
- [spaCy](https://spacy.io/) for NER model
- [VADER](https://github.com/cjhutto/vaderSentiment) for sentiment analysis
- [Plotly Dash](https://plotly.com/dash/) for dashboard visualizations
- [Anthropic](https://www.anthropic.com/) for Claude LLM assistance in lexicon generation and UI design
