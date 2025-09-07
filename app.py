# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E90FF;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2E8B57;
        margin-bottom: 1rem;
    }
    .prediction-pass {
        font-size: 1.8rem;
        color: #2E8B57;
        background-color: #F0FFF0;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #2E8B57;
    }
    .prediction-fail {
        font-size: 1.8rem;
        color: #DC143C;
        background-color: #FFF0F5;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #DC143C;
    }
    .stButton>button {
        background-color: #1E90FF;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #0066CC;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<h1 class="main-header">🎓 Student Performance Predictor</h1>', unsafe_allow_html=True)
st.write("""
This app predicts whether a student is likely to **pass** or **fail** based on their study habits, 
attendance, and other factors. Adjust the values in the sidebar and see the prediction in real-time!
""")

# Function to generate sample data
@st.cache_data
def generate_data():
    np.random.seed(42)
    n_samples = 500
    
    data = {
        'study_hours': np.random.normal(15, 5, n_samples),
        'attendance': np.random.normal(85, 10, n_samples),
        'previous_score': np.random.normal(70, 15, n_samples),
        'extracurricular': np.random.randint(0, 5, n_samples),
        'sleep_hours': np.random.normal(7, 1.5, n_samples),
        'pass': np.zeros(n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create a realistic prediction logic
    df['pass'] = (
        (df['study_hours'] > 12) & 
        (df['attendance'] > 75) & 
        (df['previous_score'] > 60) &
        (df['sleep_hours'] > 6)
    ).astype(int)
    
    # Add some noise to make it more realistic
    noise = np.random.choice([0, 1], size=n_samples, p=[0.15, 0.85])
    df['pass'] = df['pass'] ^ noise
    
    return df

# Load or generate data
df = generate_data()

# Split data into features and target
X = df[['study_hours', 'attendance', 'previous_score', 'extracurricular', 'sleep_hours']]
y = df['pass']

# Train a simple model
@st.cache_resource
def train_model():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# Sidebar for user input
st.sidebar.header("📊 Student Information")
st.sidebar.write("Adjust the values to see how they affect the prediction:")

study_hours = st.sidebar.slider("Weekly Study Hours", 0.0, 40.0, 15.0, 0.5)
attendance = st.sidebar.slider("Attendance Percentage", 0.0, 100.0, 85.0, 1.0)
previous_score = st.sidebar.slider("Previous Exam Score", 0.0, 100.0, 70.0, 1.0)
extracurricular = st.sidebar.slider("Extracurricular Activities (hours/week)", 0, 20, 2)
sleep_hours = st.sidebar.slider("Sleep Hours per Night", 3.0, 12.0, 7.0, 0.5)

# Create input dataframe for prediction
input_data = pd.DataFrame({
    'study_hours': [study_hours],
    'attendance': [attendance],
    'previous_score': [previous_score],
    'extracurricular': [extracurricular],
    'sleep_hours': [sleep_hours]
})

# Make prediction
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0]

# Display the prediction
st.markdown('<h2 class="sub-header">📈 Prediction Result</h2>', unsafe_allow_html=True)

if prediction == 1:
    st.markdown(f'<div class="prediction-pass">✅ This student is likely to PASS! (Confidence: {probability[1]*100:.1f}%)</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="prediction-fail">❌ This student is likely to FAIL! (Confidence: {probability[0]*100:.1f}%)</div>', unsafe_allow_html=True)

# Display probability gauge
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(px.bar(x=['Fail', 'Pass'], y=probability, 
                          title='Prediction Probability', 
                          color=['Fail', 'Pass'],
                          color_discrete_map={'Fail': 'red', 'Pass': 'green'},
                          labels={'x': 'Outcome', 'y': 'Probability'}),
                   use_container_width=True)

with col2:
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': ['Study Hours', 'Attendance', 'Previous Score', 'Extracurricular', 'Sleep Hours'],
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    st.plotly_chart(px.bar(feature_importance, x='Importance', y='Feature', 
                          title='Feature Importance in Prediction',
                          orientation='h'),
                   use_container_width=True)

# Display student profile
st.markdown('<h2 class="sub-header">👨‍🎓 Student Profile</h2>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Study Hours", f"{study_hours} hrs/week")
col2.metric("Attendance", f"{attendance}%")
col3.metric("Previous Score", f"{previous_score}%")
col4.metric("Extracurricular", f"{extracurricular} hrs/week")
col5.metric("Sleep Hours", f"{sleep_hours} hrs/night")

# Improvement suggestions
st.markdown('<h2 class="sub-header">💡 Improvement Suggestions</h2>', unsafe_allow_html=True)

if prediction == 0:
    suggestions = []
    if study_hours < 12:
        suggestions.append("📚 Increase study hours to at least 12 hours per week")
    if attendance < 75:
        suggestions.append("🕒 Improve attendance to above 75%")
    if previous_score < 60:
        suggestions.append("📝 Focus on improving fundamental knowledge through extra practice")
    if sleep_hours < 6:
        suggestions.append("😴 Ensure at least 6 hours of sleep per night for better concentration")
    if extracurricular > 10:
        suggestions.append("⚖️ Balance extracurricular activities with study time")
    
    for suggestion in suggestions:
        st.write(f"- {suggestion}")
else:
    st.write("🎉 Great job! The student is on track to pass. Keep up the good habits!")

# Data visualization
st.markdown('<h2 class="sub-header">📊 Data Distribution</h2>', unsafe_allow_html=True)

viz_option = st.selectbox("Select visualization", ["Study Hours vs Attendance", "Previous Score Distribution", "Sleep Hours vs Study Hours"])

if viz_option == "Study Hours vs Attendance":
    fig = px.scatter(df, x='study_hours', y='attendance', color='pass',
                    title='Study Hours vs Attendance',
                    labels={'study_hours': 'Study Hours per Week', 'attendance': 'Attendance Percentage'},
                    color_discrete_map={0: 'red', 1: 'green'})
    st.plotly_chart(fig, use_container_width=True)
    
elif viz_option == "Previous Score Distribution":
    fig = px.histogram(df, x='previous_score', color='pass', barmode='overlay',
                      title='Distribution of Previous Scores',
                      labels={'previous_score': 'Previous Exam Score'},
                      color_discrete_map={0: 'red', 1: 'green'})
    st.plotly_chart(fig, use_container_width=True)
    
else:
    fig = px.scatter(df, x='sleep_hours', y='study_hours', color='pass',
                    title='Sleep Hours vs Study Hours',
                    labels={'sleep_hours': 'Sleep Hours per Night', 'study_hours': 'Study Hours per Week'},
                    color_discrete_map={0: 'red', 1: 'green'})
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("### About This App")
st.write("""
This student performance predictor uses a machine learning model (Random Forest) trained on synthetic data. 
The model considers factors like study hours, attendance, previous scores, extracurricular activities, and sleep patterns.
**Note:** This is a demonstration app and should not be used for actual student evaluation.
""")
