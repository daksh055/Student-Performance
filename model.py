import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Generate sample data
def generate_sample_data(n_samples=1000):
    np.random.seed(42)
    
    data = {
        'study_hours': np.random.normal(15, 5, n_samples),
        'attendance': np.random.normal(85, 10, n_samples),
        'previous_score': np.random.normal(70, 15, n_samples),
        'extracurricular': np.random.randint(0, 5, n_samples),
        'pass_exam': np.zeros(n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create prediction logic
    df['pass_exam'] = (
        (df['study_hours'] > 12) & 
        (df['attendance'] > 75) & 
        (df['previous_score'] > 60)
    ).astype(int)
    
    # Add some noise
    noise = np.random.choice([0, 1], size=n_samples, p=[0.1, 0.9])
    df['pass_exam'] = df['pass_exam'] ^ noise
    
    return df

# Train and save model
def train_model():
    # Generate sample data
    df = generate_sample_data()
    
    # Features and target
    X = df[['study_hours', 'attendance', 'previous_score', 'extracurricular']]
    y = df['pass_exam']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    
    # Save model
    joblib.dump(model, 'student_model.pkl')
    print("Model saved as 'student_model.pkl'")
    
    return model, accuracy

if __name__ == "__main__":
    train_model()
