# YouTube Churn Prediction Using Sentiment Analysis

## 📚 Project Overview

This project develops a YouTube Churn Prediction Dashboard that analyzes video engagement metrics and audience sentiment to predict whether a video will retain its audience or experience churn. The system uses YouTube API for data collection, VADER Sentiment Analysis to analyze comments, and Machine Learning (XGBoost) to predict churn with 91% accuracy.

### 🎯 Problem Statement

Content creators on YouTube struggle to identify early indicators of churn, which leads to a decline in audience engagement and overall channel performance. This project addresses the following challenges:

- Lack of real-time insights into audience sentiment
- No predictive mechanism to assess future audience retention
- Inability to identify the key drivers behind churn

## ⚙️ Features

- **Video Engagement Analysis**: Analyzes views, likes, and comments to identify engagement patterns
- **Sentiment Analysis**: Uses VADER to analyze audience sentiment from comments
- **Churn Prediction**: Implements XGBoost model to predict if a video will experience churn
- **Interactive Dashboard**: Streamlit-based interface for real-time predictions and insights
- **Feature Importance Analysis**: Identifies key factors influencing churn

## 🚀 Technology Stack

- **Backend**: Python, Streamlit
- **Data Processing**: Pandas, NumPy
- **API Integration**: YouTube Data API v3
- **Sentiment Analysis**: VADER
- **Machine Learning**: Scikit-learn, XGBoost
- **Visualization**: Matplotlib, Seaborn

## 📈 Model Performance

| Model | Accuracy (%) |
|-------|--------------|
| Logistic Regression | 85% |
| Decision Tree | 78% |
| Random Forest | 89% |
| SVM | 82% |
| Gradient Boosting | 88% |
| KNN | 80% |
| **XGBoost** | **91%** |

## 📁 Project Structure

```
/YouTube-Churn-Prediction
├── /data
│   └── final_dataset.csv
├── /models
│   ├── best_model.pkl
│   └── feature_names.pkl
├── /notebooks
│   └── Churn_Prediction_Project.ipynb
├── /reports
│   └── analysis_report.pdf
├── /scripts
│   └── app.py
├── requirements.txt
└── README.md
```

## 🛠️ Installation & Setup

1. Clone the repository:
   ```
   git clone https://github.com/YourUsername/YouTube-Churn-Prediction.git
   cd YouTube-Churn-Prediction
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up YouTube API credentials:
   - Create a project in the [Google Developer Console](https://console.developers.google.com/)
   - Enable the YouTube Data API v3
   - Create API credentials and save the API key

5. Run the Streamlit application:
   ```
   streamlit run scripts/app.py
   ```

## 💡 Key Insights & Recommendations

- **High Likes-to-Views Ratio**: Videos with higher likes-to-views ratios are less likely to experience churn
- **Negative Sentiment in Comments**: Strong indicator of potential churn
- **Comment Engagement**: Creators who engage with comments tend to have lower churn rates
- **Consistent Posting Schedule**: Helps maintain audience engagement and prevents churn

## 🔮 Future Enhancements

- Optimize API rate limiting for more efficient data collection
- Implement user authentication for dashboard access
- Enhance sentiment analysis with BERT or GPT models
- Develop personalized content strategies for creators

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Hamsika7/YouTube-Churn-Prediction/blob/main/LICENSE) file for details.

## 👥 Contributions

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Contact

For questions or feedback, please contact [hamsikassnn2004@gmail.com](mailto:hamsikassnn2004@gmail.com).

---

*Note: This project is for educational purposes only and is not affiliated with YouTube or Google.*
