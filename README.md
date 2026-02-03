# Mental Health Treatment Predictor

A machine learning web application that predicts whether an individual usually requires mental health treatment based on their workplace and personal history.

This project demonstrates an end-to-end ML pipeline: from data cleaning and feature engineering to model training and deployment.

## 📊 Dataset

The model is trained on data based on the **[Mental Health in Tech Survey 2016](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-2016)** from OSMI (Open Sourcing Mental Illness). 

*Acknowledgement: This dataset is available on Kaggle and provided by OSMI.*

For a detailed history of the project's development and fixes, see the [Project Walkthrough](docs/project_walkthrough.md).

> **Note:** This repository includes a data generation script (`generate_sample_data.py`) that creates a synthetic dataset mirroring the schema and data quality issues (missing values, typos) of the original Kaggle dataset to demonstrate the robustness of the cleaning pipeline.

## 🛠️ Tech Stack

- **Frontend:** HTML5, CSS3, JavaScript
- **Backend:** Python, Flask
- **Machine Learning:** Scikit-learn, Pandas, NumPy
- **Containerization:** Docker
- **Deployment:** Google Cloud Platform (Cloud Run)

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- `pip`

### Installation

1.  **Clone the repository**
    ```bash
    git clone <your-repo-url>
    cd mh-kaggledata-predict-web
    ```

2.  **Create a virtual environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Run the Pipeline** (Generate data -> Train -> Start App)
    You can run the entire workflow with the provided script which handles data generation, processing, and training steps.
    ```bash
    # Generate synthetic data, train the model, and start the app
    python run_pipeline.py --generate && python train_model.py && python app.py
    ```

    *Alternatively, run step-by-step:*
    ```bash
    python run_pipeline.py --generate   # Generate/Clean Data
    python train_model.py               # Train Model & Save Artifacts
    python app.py                       # Start Flask Server
    ```

2.  **Access the App**
    Open your browser and navigate to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

## ☁️ Deployment to Google Cloud Platform (GCP)

This application is Dockerized and ready to be deployed to **Google Cloud Run**, a fully managed serverless platform.

### Prerequisites
- Google Cloud SDK (`gcloud`) installed and authenticated.
- A GCP Project with billing enabled.

### Steps

1.  **Enable Required APIs**
    ```bash
    gcloud services enable run.googleapis.com cloudbuild.googleapis.com
    ```

2.  **Build and Submit the Container**
    Replace `PROJECT_ID` with your actual GCP project ID and `APP_NAME` with your desired app name (e.g., `mh-predictor`).
    ```bash
    gcloud builds submit --tag gcr.io/PROJECT_ID/APP_NAME
    ```

3.  **Deploy to Cloud Run**
    ```bash
    gcloud run deploy APP_NAME \
      --image gcr.io/PROJECT_ID/APP_NAME \
      --platform managed \
      --region us-central1 \
      --allow-unauthenticated
    ```

4.  **Access Production App**
    Upon success, the terminal will display a Service URL (e.g., `https://mh-predictor-xyz-uc.a.run.app`). Click it to see your live ML app!

---


