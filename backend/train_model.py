import pandas as pd
import re
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

# Load datasets
df_fake = pd.read_csv("dataset/Fake.csv")
df_true = pd.read_csv("dataset/True.csv")

# Label the classes
df_fake["class"] = 0
df_true["class"] = 1

# Reserve 10 rows for manual testing
df_manual_testing = pd.concat([df_fake.tail(10), df_true.tail(10)], axis=0)
df_manual_testing.to_csv("dataset/manual_testing.csv", index=False)

# Remove reserved rows
df_fake = df_fake.iloc[:-10]
df_true = df_true.iloc[:-10]

# Combine and shuffle
df = pd.concat([df_fake, df_true], axis=0)
df = df.drop(["title", "subject", "date"], axis=1)
df = df.sample(frac=1).reset_index(drop=True)

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

df["text"] = df["text"].apply(clean_text)

# Features and Labels
X = df["text"]
y = df["class"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
Xv_train = vectorizer.fit_transform(X_train)
Xv_test = vectorizer.transform(X_test)

# Model training and evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(random_state=0),
    "Gradient Boosting": GradientBoostingClassifier(random_state=0)
}

scores = {}

print("🔍 Evaluating Models...\n")
for name, model in models.items():
    model.fit(Xv_train, y_train)
    preds = model.predict(Xv_test)
    acc = accuracy_score(y_test, preds)
    scores[name] = acc
    # print(f" {name} Accuracy: {acc:.4f}")
    # print(f"{classification_report(y_test, preds)}\n")

# Select best model
best_model_name = max(scores, key=scores.get)
best_model = models[best_model_name]
# print(f"Best Model: {best_model_name} with Accuracy: {scores[best_model_name]:.4f}")

# Save the best model and vectorizer
with open("models/model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Best model and vectorizer saved successfully to 'models/'")
