# YouTube Churn Prediction Dashboard

## Project Purpose

This project was developed as part of my placement preparation to demonstrate skills in API integration, sentiment analysis, data processing, and machine learning. It predicts viewer churn on YouTube videos using engagement metrics and audience sentiment.

## Live Demo

[View the live demo](https://churn-tube.streamlit.app/)

## Overview

The YouTube Churn Prediction Dashboard is an interactive web application built with Streamlit. It uses the YouTube Data API to collect video metrics, VADER for comment sentiment analysis, and an XGBoost model to predict whether a video will retain its audience or experience churn with 91% accuracy.

## Features

- **Video Engagement Analysis:** Analyze views, likes, and comments to identify engagement patterns.
- **Sentiment Analysis:** Evaluate audience sentiment using VADER on YouTube comments.
- **Churn Prediction:** Predict viewer churn using an optimized XGBoost model.
- **Interactive Dashboard:** Real-time predictions and visual insights via Streamlit.
- **Feature Importance:** Identify key factors influencing churn.

## Technology Stack

- **Backend:** Python, Streamlit
- **Data Processing:** Pandas, NumPy
- **API Integration:** YouTube Data API v3
- **Sentiment Analysis:** VADER
- **Machine Learning:** Scikit‑learn, XGBoost
- **Visualization:** Matplotlib, Seaborn

## Model Performance

| Model               | Accuracy (%) |
| ------------------- | ------------ |
| Logistic Regression | 85.0         |
| Decision Tree       | 78.0         |
| Random Forest       | 89.0         |
| SVM                 | 82.0         |
| Gradient Boosting   | 88.0         |
| KNN                 | 80.0         |
| **XGBoost**         | **91.0**     |

## Project Structure

```
YouTube-Churn-Prediction/
├── data/
│   ├── all_comments.csv
│   ├── final_dataset.csv
│   └── trending_videos.csv
├── models/
│   ├── best_model.pkl
│   └── feature_names.pkl
├── notebooks/
│   └── Churn_Prediction_Project.ipynb
├── reports/
│   └── model_performance.csv
├── scripts/
│   └── app.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Hamsika7/YouTube-Churn-Prediction.git
   cd YouTube-Churn-Prediction
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up YouTube API credentials:
   - Create a project in Google Developer Console and enable YouTube Data API v3.
   - Create an API key and save it.
   - Create a `.env` file in the project root and add:
     ```env
     YOUTUBE_API_KEY=your_api_key_here
     ```
    
## Usage

1. Run the application:
   ```bash
   streamlit run scripts/app.py
   ```
2. Enter a YouTube video URL in the input field.
3. Click "Predict Churn" to view the probability, sentiment scores, and feature insights.

## Key Insights

- High likes-to-views ratio correlates with lower churn risk.
- Negative comment sentiment is a strong churn indicator.
- Active creator engagement with comments can reduce churn.

## Future Enhancements

- Improve sentiment analysis using transformer-based models.
- Implement user authentication for personalized dashboards.
- Optimize API usage for higher data throughput.
- Add scheduling and notifications for creators.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributions

Contributions are welcome. Please fork the repository and submit a pull request with your enhancements.

## Contact
**Hamsika Suresh** - [hamsikassnn2004@gmail.com](mailto:hamsikassnn2004@gmail.com)

## Acknowledgments
* YouTube Data API v3
* Streamlit
* XGBoost
* VADER Sentiment Analysis
