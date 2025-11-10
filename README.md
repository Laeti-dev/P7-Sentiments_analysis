# P7-Sentiments_analysis


## 1. Academic goals
In this project, the academic goals revolve around developing and deploying machine learning models, specifically for sentiment analysis using deep learning techniques. The key objectives include:
- **Model Development**: Learning to create and optimize various predictive models ranging from logistic regression to deep neural networks.
- **MLOps Methodologies**: Understanding and implementing MLOps practices, including continuous integration and deployment pipelines, automated unit testing, and model management.
- **Cloud Deployment**: Deploying the selected model as an API in the cloud, which involves learning cloud services and how to integrate them with applications.
- **Communication Skills**: Preparing presentation materials for non-technical audiences and writing a blog article to effectively communicate the modeling and deployment process.
- **Experimentation Management**: Using tools like MLFlow to manage and track experiments that involve model training and evaluations.
These goals align to enhance an expertise in AI development, focusing on both the technical aspects and the ability to share your findings effectively.

## 2. Project structure
```txt
P7-sentiments_analysis/
├── notebooks/              
│   ├── ikusawa_laetitia_1_API_092025.ipynb
│   ├── ikusawa_laetitia_2_scripts_notebook_modélisation_092025/
│   │   ├── P7_basic_ML_models.ipynb
│   │   ├── P7_eda.ipynb
│   │   ├── P7_advanced_model.ipynb
│   │   ├── P7_BERT.ipynb
│   └── ikusawa_laetitia_3_dossier_code_092025 #TODO
│   ├── ikusawa_laetitia_4_interface_test_API_092025
│   ├── ikusawa_laetitia_5_blog_092025 #TODO
├── requirements.txt
├── README.md
├── .gitignore
```

## 3. Client needs
Our customer, Air Paradis, is looking to anticipate and manage potential negative feedback (bad buzz) on social media. By accurately predicting the sentiment related to tweets, the company aims to proactively address customer concerns and enhance its reputation. This involves developing an AI solution that can analyze social media sentiment effectively, allowing the company to respond promptly to any issues that may arise.
To do so, we are going to use open source tweets to train our models as Air Paradis doesn't have enough data to provide.

## 4. Steps
Before we address the problem, we are going to setup a virtual environment using [UV](https://docs.astral.sh/uv/getting-started/).

Then we will be able to start the project following these steps:
- **Data Collection**: Gather a dataset of tweets, including both positive and negative sentiments. The dataset is available [here](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Data Preprocessing**: Clean the data by removing noise (such as special characters and stop words) and preparing it for analysis (e.g., tokenization).
- **Exploratory Data Analysis**: Analyze the dataset to understand the sentiment distribution, common words, and patterns that can inform our model development.
- **Model Development**: Experiment with different machine learning models, including logistic regression, and deep learning approaches like neural networks, to predict sentiment.
- **Model Evaluation**: Use appropriate metrics (like accuracy, precision, recall, and F1 score) to evaluate the performance of our models, and select the best-performing one.
- **Deployment**: Create an API for the selected model and deploy it in a cloud environment, ensuring it is accessible for real-time predictions.
- **MLOps Integration**: Implement a pipeline for continuous integration and continuous deployment (CI/CD), manage experiments with MLFlow, and automate testing procedures.
- **Testing**: Conduct local tests of our API to ensure it functions correctly before finalizing it for deployment.
- **Feedback Loop**: After deployment, monitor the model’s performance and gather feedback for future improvements.

## 4. Setup

You should have a .env looking like .env_sample

To ensure a clean and isolated environment for this project, it is recommended to use a virtual environment. This project uses `uv`, a fast Python package installer and resolver.

### Steps:

1.  **Install `uv`** (if you don't have it):
    ```bash
    pip install uv
    ````

2.  **Create the virtual environment**:
    Navigate to the project's root directory and run:
    ```bash
    uv venv --python 3.12
    ```

    For the transformers, if you use a mac, you will need to downgrade yout python version du 3.10
    ```bash
      uv venv --python 3.10
    ```


3.  **Activate the virtual environment**:
    On macOS and Linux:
    ```bash
    source .uv-venv/bin/activate
    ```
    On Windows:
    ```bash
    .venv\Scripts\activate
    ```


4.  **Install dependencies**:
    Once the virtual environment is activated, install the required packages from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    or a single libraby
    ```bash
    uv pip install mlflow
    ```

Now you are ready to run the notebooks and scripts in this project.

5. **Streamlit App**
  To start streamlit local server
  ```bash
    streamit run main.py
    ```

6. **Run unit test**
```bash
pytest
```

[![Deploy to Cloud Run](https://github.com/Laeti-dev/P7-sentiments_analysis/actions/workflows/deploy.yml/badge.svg)](https://github.com/YOUR-USERNAME/YOUR-REPO/actions/workflows/deploy.yml)
