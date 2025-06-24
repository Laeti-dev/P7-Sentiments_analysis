# P7-Sentiments_analysis

## 1. Academic goals
In this project, the academic goals revolve around developing and deploying machine learning models, specifically for sentiment analysis using deep learning techniques. The key objectives include:
- **Model Development**: Learning to create and optimize various predictive models ranging from logistic regression to deep neural networks.
- **MLOps Methodologies**: Understanding and implementing MLOps practices, including continuous integration and deployment pipelines, automated unit testing, and model management.
- **Cloud Deployment**: Deploying the selected model as an API in the cloud, which involves learning cloud services and how to integrate them with applications.
- **Communication Skills**: Preparing presentation materials for non-technical audiences and writing a blog article to effectively communicate the modeling and deployment process.
- **Experimentation Management**: Using tools like MLFlow to manage and track experiments that involve model training and evaluations.
These goals align to enhance an expertise in AI development, focusing on both the technical aspects and the ability to share your findings effectively.

## 2. Client needs
Our customer, Air Paradis, is looking to anticipate and manage potential negative feedback (bad buzz) on social media. By accurately predicting the sentiment related to tweets, the company aims to proactively address customer concerns and enhance its reputation. This involves developing an AI solution that can analyze social media sentiment effectively, allowing the company to respond promptly to any issues that may arise.
To do so, we are going to use open source tweets to train our models as Air Paradis doesn't have enough data to provide.

## 3. Steps
To address the customer problem, we will follow these steps:
- **Data Collection**: Gather a dataset of tweets, including both positive and negative sentiments. The dataset is available [here](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Data Preprocessing**: Clean the data by removing noise (such as special characters and stop words) and preparing it for analysis (e.g., tokenization).
- **Exploratory Data Analysis**: Analyze the dataset to understand the sentiment distribution, common words, and patterns that can inform our model development.
- **Model Development**: Experiment with different machine learning models, including logistic regression, and deep learning approaches like neural networks, to predict sentiment.
- **Model Evaluation**: Use appropriate metrics (like accuracy, precision, recall, and F1 score) to evaluate the performance of our models, and select the best-performing one.
- **Deployment**: Create an API for the selected model and deploy it in a cloud environment, ensuring it is accessible for real-time predictions.
- **MLOps Integration**: Implement a pipeline for continuous integration and continuous deployment (CI/CD), manage experiments with MLFlow, and automate testing procedures.
- **Testing**: Conduct local tests of our API to ensure it functions correctly before finalizing it for deployment.
- **Feedback Loop**: After deployment, monitor the model’s performance and gather feedback for future improvements.
