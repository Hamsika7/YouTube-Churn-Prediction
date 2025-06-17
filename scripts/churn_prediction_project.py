import os
import pandas as pd
import numpy as np
import joblib
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Download required nltk data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("vader_lexicon")

# Set up project paths
project_path = "Churn_Prediction_Project"
folders = ["data", "models", "scripts", "reports"]
for folder in folders:
    os.makedirs(os.path.join(project_path, folder), exist_ok=True)

# Load YouTube API key securely (replace with os.environ["YOUTUBE_API_KEY"] in production)
API_KEY = "AIzaSyBR2nLc8DJTsnbwT4PJrfR2JdScgD0ntms"
youtube = build("youtube", "v3", developerKey=API_KEY)

# Function to fetch trending videos
def get_trending_videos(region_code="US", max_results=10):
    request = youtube.videos().list(
        part="snippet,statistics",
        chart="mostPopular",
        regionCode=region_code,
        maxResults=max_results
    )
    response = request.execute()

    video_data = []
    for video in response.get("items", []):
        video_id = video["id"]
        title = video["snippet"]["title"]
        views = int(video["statistics"].get("viewCount", 0))
        likes = int(video["statistics"].get("likeCount", 0))
        comments = int(video["statistics"].get("commentCount", 0))
        category_id = video["snippet"]["categoryId"]

        video_data.append({
            "Video ID": video_id,
            "Title": title,
            "Views": views,
            "Likes": likes,
            "Comments": comments,
            "Category ID": category_id
        })

    return pd.DataFrame(video_data)

# Fetch trending videos and save
video_df = get_trending_videos(region_code="US", max_results=20)
video_csv_path = os.path.join(project_path, "data", "trending_videos.csv")
video_df.to_csv(video_csv_path, index=False)

# Function to fetch comments
def get_comments_from_videos(video_df, max_comments=50):
    all_comments = []
    for _, row in video_df.iterrows():
        video_id = row["Video ID"]
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=max_comments
            )
            response = request.execute()
            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                all_comments.append({"Video ID": video_id, "Comment": comment})
        except Exception as e:
            print(f"Error fetching comments for {video_id}: {e}")

    return pd.DataFrame(all_comments)

# Extract comments and save
comments_df = get_comments_from_videos(video_df)
comments_csv_path = os.path.join(project_path, "data", "all_comments.csv")
comments_df.to_csv(comments_csv_path, index=False)

# Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(text):
    return analyzer.polarity_scores(text)["compound"]

comments_df["Sentiment Score"] = comments_df["Comment"].astype(str).apply(get_sentiment_score)
comments_df.to_csv(comments_csv_path, index=False)

# Merge Sentiment Scores with Video Data
video_df = pd.read_csv(video_csv_path)
comments_df = pd.read_csv(comments_csv_path)

avg_sentiment = comments_df.groupby("Video ID")["Sentiment Score"].mean().reset_index()
avg_sentiment.rename(columns={"Sentiment Score": "Avg Sentiment"}, inplace=True)

final_df = video_df.merge(avg_sentiment, on="Video ID", how="left").fillna(0)
final_csv_path = os.path.join(project_path, "data", "final_dataset.csv")
final_df.to_csv(final_csv_path, index=False)

# Define churn classification
def classify_churn(row):
    if (row["Likes"] / row["Views"] < 0.02) or \
       (row["Comments"] / row["Views"] < 0.005) or \
       (row["Avg Sentiment"] < -0.2):
        return 1
    return 0

final_df["Churn"] = final_df.apply(classify_churn, axis=1)
final_df.to_csv(final_csv_path, index=False)

# Prepare Data for ML
features = ["Views", "Likes", "Comments", "Avg Sentiment"]
target = "Churn"

X = final_df[features]
y = final_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and Evaluate Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "KNN": KNeighborsClassifier(),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

# Save Best Model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
joblib.dump(best_model, os.path.join(project_path, "models", "best_model.pkl"))

# Model Performance Visualization
results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"]).sort_values(by="Accuracy", ascending=False)
plt.figure(figsize=(10, 5))
sns.barplot(x=results_df["Accuracy"], y=results_df["Model"], palette="viridis")
plt.xlabel("Accuracy")
plt.ylabel("Model")
plt.title("Model Comparison - Accuracy Scores")
plt.show()

# Load Model for Prediction
loaded_model = joblib.load(os.path.join(project_path, "models", "best_model.pkl"))
new_data = np.array([[5000, 200, 50, 0.7]])
prediction = loaded_model.predict(new_data)
print(f"Predicted Churn: {'Yes' if prediction[0] == 1 else 'No'}")
