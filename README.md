# RankSmart Search System

RankSmart is a lightweight, local Search Ranking system built entirely with **FastAPI**, **XGBoost**, and **Dynamic Text Feature Extraction**. 

Designed to be completely self-contained and run locally, this architecture does not require complicated, pre-computed external MSLR web-metrics nor does it rely on external API keys to function. It uses live semantic computing to score and rank documents accurately the moment a user types a query.

## 🌟 Key Features

- **FastAPI Backend:** Blazing fast serving of search results and easy-to-use REST endpoints.
- **XGBoost LambdaMART:** Employs advanced Learning-to-Rank (LTR) models to intelligently re-rank text documents for the absolute best performance.
- **20 Dynamic Text Features:** RankSmart dynamically evaluates roughly 20 native text metrics on-the-fly when a query hits the server. Features include *TF-IDF Cosine Similarity*, *Jaccard Similarity*, *Term Density*, *N-Gram Overlaps*, and *Exact Match Bonuses*. 
- **100% Native:** You do not need to upload models from Kaggle or configure Hugging Face tokens. The entire LTR system can be synthesized, trained, and executed perfectly inside this very repository!

---

## 🏗️ Project Architecture

* `api/main_native.py` – The core FastAPI application. This hosts the `/search` endpoints and automatically mounts the `/static` Web UI frontend.
* `src/features_native.py` – The text parsing engine that performs semantic matching and outputs the 20 realtime scoring metrics.
* `src/train_native.py` – The XGBoost training sequence. It parses your configured text documents, applies synthetic simulated query rankings, trains the LambdaMART system, and natively exports the models locally.
* `models/` – Contains the exported artifact files needed for production (`ranker_native.json`, `features_native.pkl`, `extractor_native.pkl`). 
* `api/static/index.html` – The clean Web User Interface to test and query the system.

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Installation
Ensure you have Python 3.9+ installed and run:
```powershell
pip install -r requirements.txt
```

### 2. Train the Model Globally
Because the system employs native text features, you must generate the ranking model by training it locally on the provided documents.
```powershell
cd src
python train_native.py
cd ..
```
*This will execute seamlessly and export all required `.json` and `.pkl` artifacts into the active `models/` directory.*

### 3. Launching the Backend Server
Once your models are spun up, start the FastAPI engine and serve the Web application locally:
```powershell
uvicorn api.main_native:app --reload --host 0.0.0.0 --port 8000
```

### 4. Experience the Search
Open your browser and navigate to:
**`http://localhost:8000/static/index.html`**

You can also view the documented Swagger API endpoints natively here:
**`http://localhost:8000/docs`**
