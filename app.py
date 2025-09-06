import streamlit as st
import pandas as pd
import numpy as np
import joblib
from model import generate_sample_data

# Set page config
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# Load or train model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('student_model.pkl')
        st.sidebar.success("Model loaded successfully!")
        return model
    except:
        st.sidebar.warning("Model not found. Please run model.py first.")
        return None

# Main app
def main():
    st.title("🎓 Student Performance Predictor")
    st.write("Predict whether a student is likely to pass based on their study habits and attendance.")
    
    # Load model
    model = load_model()
    
    # Sidebar for user input
    st.sidebar.header("Student Information")
    
    study_hours = st.sidebar.slider("Weekly Study Hours", 0.0, 40.0, 15.0, 0.5)
    attendance = st.sidebar.slider("Attendance Percentage", 0.0, 100.0, 85.0, 1.0)
    previous_score = st.sidebar.slider("Previous Exam Score", 0.0, 100.0, 70.0, 1.0)
    extracurricular = st.sidebar.slider("Extracurricular Activities (hours/week)", 0, 20, 2)
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'study_hours': [study_hours],
        'attendance': [attendance],
        'previous_score': [previous_score],
        'extracurricular': [extracurricular]
    })
    
    # Display input data
    st.subheader("Student Profile")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Study Hours", f"{study_hours} hrs/week")
        st.metric("Attendance", f"{attendance}%")
    
    with col2:
        st.metric("Previous Score", f"{previous_score}%")
        st.metric("Extracurricular", f"{extracurricular} hrs/week")
    
    # Prediction
    if model is not None:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        st.subheader("Prediction Result")
        
        if prediction == 1:
            st.success("✅ This student is likely to PASS the exam!")
            st.write(f"Confidence: {probability[1]*100:.1f}%")
        else:
            st.error("❌ This student is likely to FAIL the exam!")
            st.write(f"Confidence: {probability[0]*100:.1f}%")
        
        # Progress bars for probabilities
        st.progress(probability[1])
        st.caption(f"Pass probability: {probability[1]*100:.1f}%")
        
        # Tips for improvement
        if prediction == 0:
            st.subheader("💡 Improvement Suggestions")
            if study_hours < 12:
                st.write("- Increase study hours to at least 12 hours per week")
            if attendance < 75:
                st.write("- Improve attendance to above 75%")
            if previous_score < 60:
                st.write("- Focus on improving fundamental knowledge")
    
    # Sample data visualization
    if st.checkbox("Show Sample Data Distribution"):
        sample_data = generate_sample_data(100)
        st.subheader("Sample Data Distribution")
        st.dataframe(sample_data.head(10))
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.bar_chart(sample_data['pass_exam'].value_counts())
            st.caption("Pass/Fail Distribution")
        
        with col2:
            st.line_chart(sample_data[['study_hours', 'attendance']].head(20))
            st.caption("Study Hours vs Attendance")
    
    # Footer
    st.markdown("---")
    st.caption("This is a simple prediction model for demonstration purposes. Actual results may vary.")

if __name__ == "__main__":
    main()
