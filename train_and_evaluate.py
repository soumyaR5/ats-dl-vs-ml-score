import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer, util

# -------------------------------
# LOAD DATASET
# -------------------------------
df = pd.read_csv("data/resume_dataset.csv")

# Add resume_id if not present
if "resume_id" not in df.columns:
    df["resume_id"] = range(1, len(df) + 1)

# Combine text
df["combined"] = df["resume_text"] + " " + df["job_description"]

# -------------------------------
# TRAIN ATS CLASSIFIER
# -------------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["combined"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# -------------------------------
# EVALUATION METRICS (REAL)
# -------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

print("\nMODEL PERFORMANCE:")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

# -------------------------------
# SEMANTIC MATCHING (MiniLM)
# -------------------------------
sim_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_similarity(resume, job):
    emb1 = sim_model.encode(resume, convert_to_tensor=True)
    emb2 = sim_model.encode(job, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2)) * 100

# -------------------------------
# GENERATE SCORES
# -------------------------------
ats_scores = []
similarities = []

for i in range(len(df)):
    resume = df.loc[i, "resume_text"]
    job = df.loc[i, "job_description"]

    vec = vectorizer.transform([resume + " " + job])
    ats_score = model.predict_proba(vec)[0][1]

    sim_score = get_similarity(resume, job)

    ats_scores.append(ats_score)
    similarities.append(sim_score)

df["ATS_score"] = ats_scores
df["match_score"] = similarities

# Final hybrid score
df["final_score"] = (df["ATS_score"] * 0.5) + (df["match_score"] / 100 * 0.5)

# -------------------------------
# SAVE OUTPUT FILES
# -------------------------------
df.to_csv("results.csv", index=False)
print("\nResults saved to results.csv")

scores_df = df[[
    "resume_id",
    "ATS_score",
    "match_score",
    "final_score"
]]

scores_df.to_csv("dl_ats_scores.csv", index=False)
print("ATS, Match, and Final scores saved to ml_ats_scores.csv")

# -------------------------------
# TOP PERFORMERS
# -------------------------------
top_candidates = df.sort_values(by="final_score", ascending=False).head(10)

print("\nTOP PERFORMERS:")
print(top_candidates[[
    "resume_id",
    "ATS_score",
    "match_score",
    "final_score"
]])