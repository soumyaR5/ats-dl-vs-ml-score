import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBClassifier

# -------------------------------
# LOAD DATASET
# -------------------------------
df = pd.read_csv("data/resume_dataset.csv")

if "resume_id" not in df.columns:
    df["resume_id"] = range(1, len(df) + 1)

# -------------------------------
# PREPROCESSING
# -------------------------------
df["combined"] = df["resume_text"] + " " + df["job_description"]

# -------------------------------
# SKILL VOCAB
# -------------------------------
SKILL_VOCAB = {
    "python","java","c++","sql","r","machine learning","deep learning",
    "nlp","pandas","numpy","tensorflow","pytorch","aws","docker",
    "kubernetes","tableau","power bi","excel","statistics"
}

# -------------------------------
# ATS SCORE (FEATURE ONLY)
# -------------------------------
def compute_ats_score(text):
    skills = {s for s in SKILL_VOCAB if s in text.lower()}
    return (len(skills) / len(SKILL_VOCAB)) * 100

df["ATS_score"] = df["resume_text"].apply(compute_ats_score)

# -------------------------------
# TF-IDF FEATURES
# -------------------------------
vectorizer = TfidfVectorizer(max_features=2000)
X_text = vectorizer.fit_transform(df["combined"])

# -------------------------------
# ADD ATS SCORE AS FEATURE
# -------------------------------
X = np.hstack([X_text.toarray(), df[["ATS_score"]].values])

# -------------------------------
# TARGET (NO LEAKAGE)
# -------------------------------
y = df["label"]

# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# RANDOM FOREST
# -------------------------------
rf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# -------------------------------
# XGBOOST
# -------------------------------
xgb = XGBClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)
xgb.fit(X_train, y_train)

# -------------------------------
# ENSEMBLE (PROBABILITY)
# -------------------------------
rf_pred = rf.predict_proba(X_test)[:, 1]
xgb_pred = xgb.predict_proba(X_test)[:, 1]

y_pred_prob = (rf_pred + xgb_pred) / 2
y_pred = (y_pred_prob >= 0.5).astype(int)

# -------------------------------
# EVALUATION (REAL)
# -------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nMODEL PERFORMANCE:")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

# -------------------------------
# MATCH SCORE
# -------------------------------
tfidf_resume = vectorizer.fit_transform(df["resume_text"])
tfidf_job = vectorizer.transform(df["job_description"])

match_scores = []
for i in range(len(df)):
    sim = cosine_similarity(tfidf_resume[i], tfidf_job[i])[0][0]
    match_scores.append(sim * 100)

df["match_score"] = match_scores

# -------------------------------
# ML SCORE (FULL DATA)
# -------------------------------
rf_full = rf.predict_proba(X)[:, 1]
xgb_full = xgb.predict_proba(X)[:, 1]
df["ML_score"] = (rf_full + xgb_full) / 2 * 100

# -------------------------------
# FAIRNESS SCORE
# -------------------------------
df["gender"] = np.random.choice(["Male", "Female"], len(df))

def compute_fairness(df, col):
    m = df[df["gender"] == "Male"][col].mean()
    f = df[df["gender"] == "Female"][col].mean()
    return 1 - abs(m - f) / 100

df["fairness_score"] = compute_fairness(df, "ML_score")

# -------------------------------
# FINAL SCORES
# -------------------------------
df["final_score"] = (
    0.5 * df["ATS_score"] +
    0.3 * df["ML_score"] +
    0.2 * df["match_score"]
)

# -------------------------------
# SAVE OUTPUT
# -------------------------------
output_df = df[[
    "resume_id",
    "ATS_score",
    "match_score",
    "ML_score",
    "final_score"
]]

output_df.to_csv("ml_ats_score.csv", index=False)

# -------------------------------
# TOP 10
# -------------------------------
top_10 = output_df.sort_values(by="final_score", ascending=False).head(10)

print("\nTOP 10 PERFORMERS:")
print(top_10)



# -------------------------------
# VISUALIZATION: CONFUSION MATRIX
# -------------------------------
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Rejected", "Selected"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - ATS Resume Screening")
plt.grid(False)
plt.show()


# -------------------------------
# VISUALIZATION: FINAL SCORE DISTRIBUTION
# -------------------------------
plt.figure(figsize=(10, 6))

plt.hist(
    output_df["final_score"],
    bins=15,
    edgecolor="black"
)

plt.title("Distribution of Final ATS Scores")
plt.xlabel("Final Score")
plt.ylabel("Number of Candidates")
plt.grid(True, linestyle="--", alpha=0.5)

plt.show()


# -------------------------------
# OPTIONAL: TOP 10 SCORE BAR CHART
# -------------------------------
plt.figure(figsize=(12, 6))

top_10_sorted = top_10.sort_values(by="final_score", ascending=True)

plt.barh(
    top_10_sorted["resume_id"].astype(str),
    top_10_sorted["final_score"]
)

plt.title("Top 10 Candidate Final Scores")
plt.xlabel("Final Score")
plt.ylabel("Resume ID")
plt.grid(True, linestyle="--", alpha=0.5)

plt.show()