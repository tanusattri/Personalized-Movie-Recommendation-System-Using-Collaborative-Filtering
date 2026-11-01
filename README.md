# Personalized Movie Recommendation System Using Collaborative Filtering
### HCLTech Career Shaper Programme | IV Year Industry Project #28 

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask 3.1](https://img.shields.io/badge/Flask-3.1.3-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange.svg)](https://scikit-learn.org/)
[![Bootstrap 5](https://img.shields.io/badge/Bootstrap-5.3-purple.svg)](https://getbootstrap.com/)
[![Compliance: DPDP Act 2023](https://img.shields.io/badge/Compliance-DPDP%20Act%202023-brightgreen.svg)](#-ethical-compliance-declaration--governance)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Executive Summary & Project Overview

This repository hosts the complete, production-ready **Personalized Movie Recommendation System** developed under the **HCLTech Career Shaper Programme (Track #28: Digital-Data-AI / AIML)** 

Modern digital streaming architectures often suffer from the **"Filter Bubble" dilemma**: standard machine learning recommenders over-index on historical popularity and repetitive mainstream genres, trapping users in echo chambers while marginalizing rich regional cinema.

This project delivers a multi-algorithmic collaborative filtering platform combining:
1. **SVD Matrix Factorization** (Latent Factor Model with dynamic TruncatedSVD fallback).
2. **User-Based Collaborative Filtering** (Cosine Similarity neighborhoods).
3. **Item-Based Collaborative Filtering** (Cosine Similarity co-rated item patterns).
4. **Diversity & Novelty Re-Ranking Post-Processor** to actively dissolve filter bubbles:
   $$\text{Adjusted Score}(i) = \frac{\hat{r}_{u,i}}{1.0 + 0.3 \times \text{Count}(\text{Genre}(i) \in \text{TopRecommendations})}$$
5. **Real-Time Online Learning Pipeline** (`POST /api/rate`) with dynamic matrix re-computation.
6. **Strict Ethical AI Governance** compliant with the **Digital Personal Data Protection (DPDP) Act, 2023**.

---

## ⚖️ Ethical Compliance Declaration & Governance

This project incorporates formal ethical governance adhering to the statutory requirements of the **HCLTech Career Shaper Programme** and the **Government of India DPDP Act, 2023**.

### Key Ethical Confirmations:
- **Zero Personally Identifiable Information (PII):** In compliance with the **DPDP Act 2023** data minimization principles, no user names, emails, IP addresses, or demographic identifiers are stored. Users are represented solely by synthetic alphanumeric tokens (`U0001`–`U0200`).
- **Filter Bubble & Echo Chamber Dissolution:** Unconstrained recommendation engines reinforce cognitive biases. Our active diversity re-ranking mechanism penalizes consecutive genre repetitions, ensuring multi-genre and cross-cultural discovery.
- **Multilingual Cultural Equity:** The dataset intentionally balances **7 Indian regional languages** (Hindi, Tamil, Telugu, Malayalam, Bengali, Kannada, Marathi) alongside global cinema (English, Korean, Japanese), eliminating linguistic centralisation.
- **Anomaly Noise Auditing:** 224 synthetic rating anomalies (4.48% bot/noise ratings) are transparently labeled via the `is_anomaly` column and excluded from training pipelines.
- **Source Licensing & AI Assistance:** Base dataset references the public MovieLens dataset (GroupLens Research). AI assistance used for dataset scaling prompt design is fully disclosed.

---

## 📊 Empirical Benchmarks & Model Evaluation (80/20 Train-Test Split)

All algorithms were benchmarked across 5,000 ratings using a rigorous 80/20 train-test partition:

| Algorithm | RMSE ↓ | MAE ↓ | Precision@5 ↑ | Recall@5 ↑ | Catalog Coverage ↑ | Diversity Score ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVD Matrix Factorization** | 1.7439 | 1.5162 | 0.0460 | 0.0985 | 14.81% | **0.9070** |
| **User-Based CF (Cosine)** | 1.0090 | 0.8094 | 0.0370 | 0.0748 | **100.00%** | 0.8875 |
| **Item-Based CF (Cosine)** | **0.8884** | **0.7211** | 0.0290 | 0.0672 | **100.00%** | 0.8845 |

- **Top Accuracy:** Item-Based Collaborative Filtering achieved the lowest error (**RMSE: 0.8884**, **MAE: 0.7211**).
- **Zero Blind Spots:** Both neighborhood models achieved **100.0% catalog coverage**, ensuring zero catalogue starvation.
- **Max Diversity:** SVD latent factors combined with our re-ranking algorithm achieved a **0.9070 diversity score**, surfacing critically acclaimed regional cinema.

---

## 📂 Repository File Structure

```text
├── app.py                              # Production Flask REST API & Web Server
├── recommender.py                      # RecommenderEngine (SVD, User-CF, Item-CF, Diversity Re-ranker)
├── generate_dataset.py                 # Reproducible dataset generator (5,000 rows, Seed: 42)
├── eda_notebook.py                     # Statistical Exploratory Data Analysis & plot generator
├── requirements.txt                    # Project dependency specification
├── dataset.csv                         # Scaled 5,000-row movie ratings dataset (CSV format)
├── dataset.xlsx                        # Scaled 5,000-row movie ratings dataset (Excel format)
├── data_dictionary.csv                 # Detailed schema documentation for all 8 fields (CSV format)
├── data_dictionary.xlsx                # Detailed schema documentation for all 8 fields (Excel format)
│
├── eda_rating_distribution.png         # Rating distribution histogram with KDE overlay (Figure 1)
├── eda_genre_distribution.png          # Genre volume breakdown bar chart (Figure 2)
├── eda_language_breakdown.png          # Multilingual catalog breakdown donut chart (Figure 3)
│
├── templates/
│   └── index.html                      # Bootstrap 5 dark-mode responsive single-page web app
│
├── Ethical_Compliance_Declaration.docx # Signed Ethical Compliance Declaration document
├── Project_Report.docx                 # Complete 8+ page formal academic/technical project report
├── Project_Synopsis_Completed.docx     # Official filled submission synopsis document
├── Source_Code_and_Video_Appendix.docx # Complete source code listing & 90-second video demo script
├── run_app.bat                         # One-click Windows launch script
└── README.md                           # Comprehensive GitHub documentation
```

---

## 🚀 Quickstart & Execution Guide

### 1. Prerequisites
- Python 3.10 or higher
- PowerShell, Command Prompt, or Terminal

### 2. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/movie-recommender-hcl28.git
cd movie-recommender-hcl28
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Dataset Generation & EDA Visualizations (Pre-generated)
```bash
python generate_dataset.py
python eda_notebook.py
```

### 5. Launch the Web Platform
```bash
python app.py
```
*Or on Windows: simply double-click **`run_app.bat`**.*

Navigate to **`http://127.0.0.1:5000`** in your web browser.

---

## 🌐 REST API Specification

| Endpoint | Method | Query / Payload Parameters | Response Description |
| :--- | :---: | :--- | :--- |
| **`/`** | `GET` | None | Serves the interactive Bootstrap 5 Web Dashboard |
| **`/api/recommend`** | `GET` | `user_id` (e.g. `U0001`), `algo` (`svd`/`user`/`item`), `k` (`3`–`10`), `diversity` (`true`/`false`) | Returns JSON ranked recommendations with genres, languages, and star ratings |
| **`/api/metrics`** | `GET` | None | Returns JSON performance metrics (RMSE, MAE, Precision@5, Coverage, Diversity) |
| **`/api/rate`** | `POST` | JSON: `{"user_id": "U0001", "movie_title": "KGF", "rating": 4.5}` | Records new rating and triggers real-time online model retraining |

---

## 🎬 90-Second Demo Video Walkthrough

The demo video script and storyboard are detailed in [`Source_Code_and_Video_Appendix.docx`](./Source_Code_and_Video_Appendix.docx):
- **[0:00–0:15] Title & Problem Statement:** Highlighting filter bubbles and regional cinema neglect.
- **[0:15–0:40] Dashboard Walkthrough:** Navigating controls across SVD, User-CF, and Item-CF models.
- **[0:40–1:05] Diversity Re-Ranking in Action:** Toggling diversity to observe regional films (*Manjummel Boys*, *Kantara*, *Sairat*) surfacing in top ranks.
- **[1:05–1:20] Benchmarks & Online Learning:** Real-time metrics review and live interactive movie rating submission.
- **[1:20–1:30] Governance & Conclusion:** DPDP Act 2023 compliance confirmation.

---

## 📚 References
1. Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets. *ACM TiiS*, 5(4), 1-19.
2. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. *IEEE Computer*, 42(8), 30-37.
3. Sarwar, B., et al. (2001). Item-Based Collaborative Filtering Recommendation Algorithms. *WWW '01*, 285-295.
4. Pariser, E. (2011). *The Filter Bubble: What the Internet Is Hiding from You*. Penguin Press.
5. Hug, N. (2020). Surprise: A Python library for recommender systems. *JOSS*, 5(52), 2174.
6. Ministry of Law and Justice, Government of India. (2023). *The Digital Personal Data Protection Act, 2023*.
