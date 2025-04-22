import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np

# Load pre-trained emotion detection model from Hugging Face
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Function to classify emotion
def detect_emotion(text):
    result = emotion_classifier(text)
    emotion = result[0]['label']
    return emotion

# Tips based on emotion
def get_tips(emotion):
    tips = {
        'joy': "You're feeling great! Keep up the positive energy, and remember to take a break to recharge.",
        'anger': "It seems like you're feeling frustrated. Take a deep breath, step away for a moment, and try to relax.",
        'fear': "You may be feeling anxious. Try some mindfulness exercises or take a break to calm down.",
        'sadness': "It looks like you're feeling down. Talk to someone, and don't hesitate to reach out for support.",
        'surprise': "You're feeling surprised! Embrace the unexpected and take some time to process what's happening.",
        'disgust': "You might be feeling uncomfortable. Try to focus on positive aspects and take a short break."
    }
    return tips.get(emotion, "Emotion not recognized. Take a moment to reflect.")

# Function to check if stress level is high
def is_stress_high(emotion):
    high_stress_emotions = ['anger', 'fear', 'sadness']
    if emotion in high_stress_emotions:
        return True
    return False

# Streamlit app
def app():
    st.title("Employee Emotion Detection")
    st.write("Enter a sentence that reflects how you're feeling today!")

    # User input
    user_input = st.text_area("How are you feeling?", "")

    # Keep track of emotional history
    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = []

    # Add current emotion to history
    if user_input:
        emotion = detect_emotion(user_input)
        st.session_state.emotion_history.append(emotion)

        st.write(f"Detected Emotion: {emotion}")
        tips = get_tips(emotion)
        st.write(f"Tip: {tips}")

        # Check if stress level is high and alert HR
        if is_stress_high(emotion):
            st.warning("High stress level detected! Alerting HR for further action.")
            st.session_state.emotion_history.append("HR Alert: High Stress")

    # Show emotional history
    st.subheader("Emotional History")
    if st.session_state.emotion_history:
        st.write("Here’s a record of your emotions today:")
        st.write(st.session_state.emotion_history)

    # Visualizing emotions with a bar chart
    st.subheader("Emotion Breakdown")
    emotion_counts = {emotion: st.session_state.emotion_history.count(emotion) for emotion in set(st.session_state.emotion_history)}
    
    if emotion_counts:
        emotions = list(emotion_counts.keys())
        counts = list(emotion_counts.values())
        
        fig, ax = plt.subplots()
        ax.bar(emotions, counts, color='skyblue')
        ax.set_xlabel('Emotions')
        ax.set_ylabel('Count')
        ax.set_title('Emotion Breakdown')
        st.pyplot(fig)

if __name__ == "__main__":
    app()
