# AI-Powered Customer Retention System for Telecom

## Overview
This project is an end-to-end AI-powered system designed to predict telecom customer churn, identify at-risk customers, and support data-driven retention strategies. It integrates data engineering, machine learning, deep learning, MLOps, and deployment components to simulate a real-world telecom analytics platform.

The system includes a complete workflow starting from raw data ingestion to automated ETL pipelines, predictive modeling, API deployment, web interface, and business intelligence reporting.

---

## Key Features

### Data Engineering
- Raw to processed data pipeline.
- PostgreSQL relational database with normalized tables.
- Fact table generation and feature store creation.
- Automated ETL pipeline using Airflow.
- Modular Python scripts for splitting, cleaning, loading, and exporting data.

### Machine Learning & Deep Learning
- Multiple ML models: XGBoost, Random Forest, CatBoost.
- Deep Learning model using a feed-forward neural network.
- Feature engineering and preprocessing pipelines.
- Model evaluation and comparison using standard metrics.
- Saved models for inference (pickle and H5 formats).

### MLOps Integration
- MLflow for experiment tracking and model registry.
- Reproducible training workflows.
- Versioned models stored in the repository.

### Deployment
- FastAPI backend for real-time churn prediction.
- Streamlit web application for interactive user interface.
- Dockerized deployment using Docker Compose.
- Predictive API and UI running in isolated containers.

### Business Intelligence
- Power BI dashboard for churn analytics.
- Insights into customer behavior, churn drivers, and revenue impact.
- Data aggregation from the fact table and processed dataset.

---

## Technologies Used

### Programming and Data Processing
- Python  
- Pandas, NumPy  
- SQL, PostgreSQL

### Machine Learning and Deep Learning
- Scikit-learn  
- XGBoost  
- TensorFlow / Keras

### MLOps & Workflow Automation
- MLflow  
- Apache Airflow

### Deployment
- FastAPI  
- Streamlit  
- Docker, Docker Compose

### Visualization & BI
- Power BI  
- Matplotlib, Seaborn

---

## How It Works

### 1. Data Engineering Pipeline
- Load raw dataset into the pipeline.
- Clean and transform the data using Jupyter Notebook.
- Split into normalized tables (customers, billing, services, etc.).
- Load tables into PostgreSQL using Python and SQL.
- Create a unified fact table for ML features.
- Export final ML dataset to the processed folder.

### 2. Model Training
- Use notebooks for exploratory analysis.
- Train ML models (XGBoost, Random Forest, CatBoost).
- Train a deep learning model using Keras.
- Evaluate using accuracy, precision, recall, F1 score.
- Save trained models inside the models folder.

### 3. Deployment
- FastAPI provides a prediction endpoint.
- Streamlit offers a simple front-end interface.
- Docker Compose launches both services along with MLflow.

### 4. MLOps
- MLflow tracks model parameters, metrics, and versions.
- The best model is saved to the registry and used in FastAPI.

### 5. Business Intelligence
- Power BI dashboard connects to PostgreSQL or processed dataset.
- Provides actionable insights for customer retention.

---

## Power BI Dashboard
- Open the file: `dashboard/churn_dashboard.pbix`
- Refresh dataset connection (PostgreSQL or CSV)
- Explore churn insights, customer behavior, and revenue metrics

---

## Future Improvements
- Add model explainability using SHAP  
- Real-time data ingestion pipeline  
- Model retraining automation with Airflow  
- Integration with cloud services (AWS/GCP/Azure)

---

