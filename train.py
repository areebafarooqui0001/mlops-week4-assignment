import pandas as pd
import feast
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib
import os
from datetime import datetime, timedelta

MODEL_DIR = 'artifacts'
MODEL_NAME = 'model.joblib'
METRICS_FILE = 'metrics.txt'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

FEATURE_REPO_PATH = 'feature_repo/'

os.makedirs(MODEL_DIR, exist_ok=True)

print("Connecting to Feast Feature Store at", FEATURE_REPO_PATH)
try:
    fs = feast.FeatureStore(repo_path=FEATURE_REPO_PATH)
    print("Connection successful.")
except Exception as e:
    print(f"Error connecting to Feast Feature Store: {e}")
    exit(1)

features_to_get = [
    "iris_features:sepal_length",
    "iris_features:sepal_width",
    "iris_features:petal_length",
    "iris_features:petal_width",
]
target_feature = "iris_target:species"

print("Loading entity list from local prepared data...")
try:
    entity_df = pd.read_csv("data/iris_feast.csv")[['iris_id', 'event_timestamp']]
    entity_df['event_timestamp'] = pd.to_datetime(entity_df['event_timestamp'])
    print(f"Found {len(entity_df)} entities.")

except FileNotFoundError:
    print("Error: data/iris_feast.csv not found.")
    print("Please run prep_data.py first.")
    exit(1)

print("Retrieving historical features from BigQuery via Feast...")
training_df = fs.get_historical_features(
    entity_df=entity_df,
    features=features_to_get + [target_feature],
).to_df()
print("Feature retrieval complete.")
print("\n--- Training Sample ---")
print(training_df.head())
print("----------------------------\n")



print("Splitting data...")
X = training_df.drop(columns=['iris_id', 'event_timestamp', 'species'])
y = training_df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
print("Data split complete.")


print("Training Decision Tree model...")
mod_dt = DecisionTreeClassifier(max_depth=3, random_state=1)
mod_dt.fit(X_train, y_train)
print("Model training complete.")

print("Evaluating model...")
prediction = mod_dt.predict(X_test)
accuracy = metrics.accuracy_score(prediction, y_test)
print(f'The accuracy of the Decision Tree is: {accuracy:.3f}')

print(f"Saving metrics to {METRICS_FILE}...")
with open(METRICS_FILE, "w") as f:
    f.write(f"Accuracy: {accuracy:.3f}\n")
print("Metrics saved.")

print(f"Saving model to {MODEL_PATH}...")
joblib.dump(mod_dt, MODEL_PATH)
print(f"Model saved successfully to {MODEL_PATH}")

print("train.py script finished.")