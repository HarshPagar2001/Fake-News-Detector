import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline

# Loading the dataset
data = pd.read_csv('Dataset/News.csv')

# Preprocessing the text data
data['text'] = data['text'].str.lower().fillna('')
data['title'] = data['title'].str.lower().fillna('')

# Combine title and text into one column
data['content'] = data['title'] + " " + data['text']

# Splitting the dataset into features and labels
X = data['content']
y = data['class']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a TF-IDF vectorizer and a Logistic Regression model pipeline
pipeline = make_pipeline(
    TfidfVectorizer(stop_words='english', max_features=5000),
    LogisticRegression(max_iter=1000)
)

# Training the model
pipeline.fit(X_train, y_train)

# Making predictions on the test set
y_pred = pipeline.predict(X_test)

# Printing title and prediction
for title, prediction in zip(X_test, y_pred):
    print(f"Title: {title[:50]}... | Prediction: {'Real' if prediction == 1 else 'Fake'}")

# Counting total real and fake news
real_count = sum(y_pred == 1)
fake_count = sum(y_pred == 0)

# Printing total count of real and fake news
print(f"\nNews Report:\n")
print(f"Total Real News: {real_count}")
print(f"Total Fake News: {fake_count}")

# Printing accuracy in percentage
print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Printing classification report
print(f"\nClassification Report:\n")
print(classification_report(y_test, y_pred))