 
# CrisisWatch AI

## AIM 

The main aim of this project is to support individuals experiencing suicidal thoughts, mental health issues, or substance use challenges and try to seek help on social medias. Using advanced natural language processing and machine learning techniques, the system analysis posts in real time and provides actionable insights to the mental health services and professionals. This enables mental health organizations to launch targeted awareness campaigns and provide timely support and intervention to those peoples.

 
## OVERVIEW

CrisisWatch AI is an crisis monitoring system designed to detect and analyze crisis-related content on Reddit. By collecting and processing posts from subreddits focused on mental health, suicide thoughts, and substance use, the system performs risk classification, sentimental analysis, user behavior tracking, and geolocation mapping.

The data is visualized through an interactive dashboard that enables mental health professionals and researchers to monitor trends, identify at-risk user, and make data-driven decisions for outreach and support. The solution is modular, scalable, and focused on creating real-world impact in the field of mental health using Artificial Intelligence.


## FLOWCHART

![image](https://github.com/user-attachments/assets/bc842a3e-def2-4511-8048-486610b4d73a)


## FEATURES

- Real-time Reddit Monitoring: Fetches posts from mental health, suicide prevention, and substance abuse subreddits with custom lexicon keyword filtering
- Risk Classification: Uses DistilBERT Bi-LSTM CNN model to classify posts into High, Medium, or Low risk categories  
- Sentiment Analysis: Analyzes sentiment of posts (Positive, Negative, Neutral)  
- Geolocation Mapping: Extracts locations and maps coordinates from the posts  
- User Behavior Tracking: Monitor each users to identify the patterns of at-risk users
- Interactive Dashboard: Visualizes data through charts, graphs, and heatmaps  

## AI Models

 - Sentiment Analysis:
    - VADER Sentimental Analysis Model
 - Risk Classification:
    - Fine Tuned DistilBERT Bi-LSTM CNN model trained on a modified dataset from [Kaggle Dataset](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) for classifying the risk levels with an accuracy of 84% on validation dataset.
    - Model Structure
    - 
       ![image](https://github.com/user-attachments/assets/fe8a2a0b-55ab-4cbf-9cfb-cd347948abbc)
      
 - Location Extraction:
    - spaCy Named Entity Recognition model
      
 - Coordinates Extraction:
    - Nominatim Geocode Model

## SCALABILITY

CrisisWatch AI with minimal changes, this system can be extented in the following ways:

#### 1. Multi-Platform Support
- **Extend to Other Social Media**: By expanding the data extraction module, this application can monitor additional platforms such as **X**, **Facebook**, **Tumblr**, or online forums related to mental health.
- **API Adaptability**: The current implementation uses praw for Reddit, similar API wrappers for X and Facebook can be integrated with cost and computational resources.

#### 2. Language Support
- **Multilingual Capabilities**: With multilingual models like mBERT, the app can be adapted to detect crisis posts in various language, broadening its global reach.

#### 3. Real-Time Alert System
- **Early Intervention System**: Integration with real-time alert mechanisms (email, SMS) to notify mental health professionals when highly risk user is detected (identified by a sudden increase in high-risk posts within a short time frame).

#### 4. Integration with Support Services
- **Link to Mental Health Services**: Detected high-risk users could be forwarded to local/national mental health services or NGOs for rapid response.

#### 5. Time-Series Risk Forecasting
- **Behavior Forecasting Models**: With more post data, time-based forecasting models can be built to predict future user behavior, risk levels and sentiment trends for early prevention.

## TECHNOLOGIES USED

### Backend

- Flask  
- SQLAlchemy  
- PRAW (Python Reddit API Wrapper)  
- TensorFlow/Keras  
- Transformers (DistilBERT)  
- VADER Sentiment Analysis  
- spaCy (NER)  
- Geopy (Geocoding)  

### Frontend

- Dash (Dashboard framework)  
- Plotly (Visualizations)  
- Folium (Geospatial mapping)  
- WordCloud
- HTML
- Tailwind CSS

### Database

- SQLite

## DEMO
Due to cost compute constraints, we were unable to deploy the complete version of the application that includes the full post analysis pipeline(post extraction, model inference, and geolocation).
However, we have deployed the **interactive dashboard**, which showcases visualizations based on posts ( 2026-07-01 to 2026-07-07 ) analysed and stored in the database.

Explore the deployed dashboard here

**Live Demo:** [CrisisWatchAI](https://crisiswatch-api-419808785746.asia-south1.run.app/)

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
├── templates/analyze.html      # HTML code for Analyze Page
├── static/styles.css           # Styling CSS File for Home Page and Analyze Page
├── assets/style.css            # Styling CSS File for Dashboard Page
├── risk_model_package/         # Risk Classification model package
├── HackOrbit.pptx              # Project Presentation 
└── .env                        # Environment variables     
```

## Use of Large Language Models (LLMs)
- **Custom Keyword Lexicon Creation**: Chat-Gpt was used to generate a domain-specific keyword lexicon for filtering posts related to mental health, suicide, and substance use with including coded language.
- **Frontend Styling Assistance**: Claude also assited in designing and refining the homepage animatic styling, helping to choose appropriate color transformation and logos for a clean and user-friendly interface.

## Contributors

- Gokulan M – Model Training, Backend Development
- Sri Jaai Meenakshi M – Dashboard, Frontend Development
 
## Acknowledgments

- [Reddit](https://www.reddit.com/dev/api) for API access
- [Hugging Face Transformers](https://huggingface.co/) for the DistilBERT model
- [spaCy](https://spacy.io/) for NER model
- [VADER](https://github.com/cjhutto/vaderSentiment) for sentiment analysis
- [Plotly Dash](https://plotly.com/dash/) for dashboard visualizations
- [Anthropic](https://www.anthropic.com/) for Claude LLM assistance in lexicon generation and UI design
