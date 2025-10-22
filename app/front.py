import streamlit as st
import requests
import json
import plotly.graph_objects as go
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="centered",
)

# App title and description
st.title("Sentiment Analysis Tool")
st.markdown("""
This app analyzes the sentiment of text using a machine learning model.
Enter your text below and click 'Analyze' to see the results.
""")

# API URL - change this if your API is hosted elsewhere
API_URL = "http://localhost:8000/predict"

# Function to call API and get prediction
def get_sentiment_prediction(text):
    try:
        response = requests.post(
            API_URL,
            json={"text": text},
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            return response.json()
        else:
            # Try to extract error message from response
            try:
                error_detail = response.json().get("detail", response.text)
            except Exception:
                error_detail = response.text
            st.error(f"API Error ({response.status_code}): {error_detail}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

# Text input area
text_input = st.text_area(
    "Enter text to analyze:",
    height=150,
    placeholder="Type or paste your text here...",
    key="text_input"
)

# Analysis button
if st.button("Analyze Sentiment"):
    if not text_input:
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            # Call API
            result = get_sentiment_prediction(text_input)

            if result:
                # Display results
                sentiment = result["sentiment"]
                confidence = result["confidence"]
                probabilities = result["probabilities"]

                # Main result with emoji
                emoji = "😊" if sentiment == "positive" else "😔"
                st.markdown(f"## {emoji} Sentiment: {sentiment.upper()}")

                # Confidence meter
                st.markdown(f"### Confidence: {confidence:.2%}")
                st.progress(confidence)

                # Visualization of probabilities
                st.markdown("### Probability Distribution")

                # Create a DataFrame for the probabilities
                prob_df = pd.DataFrame({
                    'Sentiment': list(probabilities.keys()),
                    'Probability': list(probabilities.values())
                })

                # Create a bar chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=prob_df['Sentiment'],
                        y=prob_df['Probability'],
                        marker_color=['#ff9999' if s == 'negative' else '#99ccff' for s in prob_df['Sentiment']],
                        text=[f"{p:.2%}" for p in prob_df['Probability']],
                        textposition='auto',
                    )
                ])

                fig.update_layout(
                    title='Sentiment Probabilities',
                    xaxis_title='Sentiment',
                    yaxis_title='Probability',
                    yaxis=dict(range=[0, 1]),
                )

                st.plotly_chart(fig)

                # Show raw API response in expandable section
                with st.expander("See API Response Details"):
                    st.json(result)

# Add information about how the model works
with st.expander("How does this work?"):
    st.markdown("""
    ### How Sentiment Analysis Works

    This application uses a machine learning model to analyze the sentiment of text:

    1. You enter text in the input field
    2. The text is sent to our API
    3. The model processes the text and predicts the sentiment
    4. Results are displayed showing if the text is positive or negative

    The confidence score indicates how sure the model is about its prediction.
    """)

# Footer
st.markdown("---")
st.markdown("© 2023 Sentiment Analysis Tool | Built with Streamlit and FastAPI")

