import streamlit as st
import requests
import json
import plotly.graph_objects as go
import pandas as pd
import os

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
API_URL = os.environ.get("API_URL", "http://localhost:8000/predict")
FEEDBACK_URL = API_URL.replace("/predict", "/feedback")
STATS_URL = API_URL.replace("/predict", "/feedback/stats")


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

def submit_feedback(prediction_id, is_correct, actual_sentiment=None, comments=""):
    """Submit user feedback"""
    st.write(f"Submitting feedback: id={prediction_id}, correct={is_correct}, actual={actual_sentiment}")
    try:
        response = requests.post(
            FEEDBACK_URL,
            json={
                "prediction_id": prediction_id,
                "is_correct": is_correct,
                "actual_sentiment": actual_sentiment,
                "comments": comments
            },
            headers={"Content-Type": "application/json"}
        )
        st.write(f"Feedback status: {response.status_code} | Response: {response.text}")
        if response.status_code == 200:
            return True
        else:
            st.error(f"Failed to submit feedback: {response.text}")
            return False
    except Exception as e:
        st.error(f"Error submitting feedback: {str(e)}")
        return False
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
                prediction_id = result.get("prediction_id", "")
                st.session_state["prediction_id"] = prediction_id
                st.session_state["last_sentiment"] = result["sentiment"]

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


prediction_id = st.session_state.get("prediction_id", "")
if prediction_id:
    sentiment = st.session_state.get("last_sentiment", "")
    confidence = st.session_state.get("last_confidence", 0)
    probabilities = st.session_state.get("last_probabilities", None)

    st.markdown("---")
    st.markdown("### 📝 Was this prediction correct?")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Correct", key="feedback_correct"):
            if submit_feedback(prediction_id, is_correct=True):
                st.success("Thank you for your feedback!")
                st.balloons()
                st.rerun()
    with col2:
        if st.button("❌ Incorrect", key="feedback_incorrect"):
            st.session_state['show_correction_form'] = True
            st.rerun()

    if st.session_state.get('show_correction_form', False):
        st.markdown("#### Help us improve!")
        actual_sentiment = st.radio("What should the sentiment be?",
                                    options=["positive", "negative"],
                                    index=0 if sentiment == "negative" else 1)
        comments = st.text_area(
            "Additional comments (optional):",
            placeholder="Why do you think the prediction was wrong?",
            key="feedback_comments"
        )
        if st.button("Submit Correction", key="submit_correction"):
            if submit_feedback(prediction_id, is_correct=False, actual_sentiment=actual_sentiment, comments=comments):
                st.success("Thank you for helping us improve!")
                st.session_state['show_correction_form'] = False
                st.rerun()

# === NEW: Show Model Performance Stats in Sidebar ===
with st.sidebar:
    st.markdown("## 📊 Model Performance")

    try:
        stats_response = requests.get(STATS_URL)
        if stats_response.status_code == 200:
            stats = stats_response.json()

            st.metric("Total Feedback", stats['total_feedback'])
            st.metric("Accuracy", f"{stats['accuracy']}%")

            if stats['total_feedback'] > 0:
                correct = stats['correct_predictions']
                incorrect = stats['incorrect_predictions']

                fig_stats = go.Figure(data=[
                    go.Pie(
                        labels=['Correct', 'Incorrect'],
                        values=[correct, incorrect],
                        marker_colors=['#99ccff', '#ff9999']
                    )
                ])

                fig_stats.update_layout(height=250, margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig_stats, use_container_width=True)
    except:
        st.info("Stats will appear after feedback is collected")

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
