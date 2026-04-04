### 📄 ATS Score Generation using Deep Learning vs Machine Learning

📌 Overview

This project focuses on building an **AI-powered Applicant Tracking System (ATS) score generator** by comparing **Deep Learning (DL)** and **Machine Learning (ML)** approaches.

The system evaluates resumes against job descriptions and generates ATS scores, helping analyze which approach performs better in real-world resume screening scenarios.

---

## 🎯 Objectives

* Generate ATS scores using both ML and DL models
* Compare performance between traditional and deep learning techniques
* Analyze effectiveness in resume-job description matching
* Provide insights for automated hiring systems

---

## 🧠 Methodology

### 🔹 Machine Learning Approach

* Text preprocessing (cleaning, tokenization)
* Feature extraction (TF-IDF / Count Vectorization)
* Similarity calculation / model-based scoring

### 🔹 Deep Learning Approach

* NLP-based embeddings
* Semantic similarity using advanced models
* Context-aware scoring mechanism

---

## 📂 Project Structure


ats-dl-vs-ml-score/
│
├── dl_method.py        # Deep Learning implementation
├── ml_method.py        # Machine Learning implementation
├── utils.py            # Helper functions
├── dl_score.csv        # DL generated scores
├── ml_score.csv        # ML generated scores
├── base_paper.pdf      # Reference research paper
└── README.md


---

## ⚙️ Tech Stack

* Python
* Natural Language Processing (NLP)
* Machine Learning Algorithms
* Deep Learning Models

---

## 📊 Output

* ATS scores generated using both approaches
* CSV files containing score comparisons
* Performance differences between ML and DL models

---

🚀 How to Run

1. Clone the Repository


git clone https://github.com/your-username/ats-dl-vs-ml-score.git
cd ats-dl-vs-ml-score


2. Install Dependencies


pip install -r requirements.txt


*(Create requirements.txt if not added)*

3. Run Models

python ml_method.py
python dl_method.py



📈 Key Insights

* ML models provide faster and simpler scoring
* DL models capture better semantic meaning
* DL generally performs better for complex resume matching


🔮 Future Work

* Integrate real-time resume upload system
* Build a web-based ATS dashboard
* Improve model accuracy using transformer models
* Deploy as a scalable API

---

## 📚 Reference

* Base research paper included in repository

---

## 👤 Author

**Soumya Ramchandran**
Aspiring Data Analyst | BI & AI Enthusiast

---

## ⭐ If you found this useful

Give this repository a star ⭐
