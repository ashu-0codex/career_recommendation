import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# Load Trained Model
# -----------------------------
model = joblib.load("job_role_model.pkl")
role_encoder = joblib.load("role_encoder.pkl")

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(page_title="Student Job Role Prediction System")

st.title("Student Job Role Prediction System")
st.write("Select your skill levels and click the Predict button.")

# -----------------------------
# Skill Mapping
# -----------------------------
skill_mapping = {
    "Not Interested": 0,
    "Poor": 1,
    "Beginner": 2,
    "Average": 3,
    "Intermediate": 4,
    "Professional": 5
}

options = list(skill_mapping.keys())

# -----------------------------
# User Inputs
# -----------------------------
db = st.selectbox("Database Fundamentals", options)
ca = st.selectbox("Computer Architecture", options)
dcs = st.selectbox("Distributed Computing Systems", options)
cs = st.selectbox("Cyber Security", options)
net = st.selectbox("Networking", options)
sd = st.selectbox("Software Development", options)
ps = st.selectbox("Programming Skills", options)
pm = st.selectbox("Project Management", options)
cff = st.selectbox("Computer Forensics Fundamentals", options)
tc = st.selectbox("Technical Communication", options)
aiml = st.selectbox("AI ML", options)
se = st.selectbox("Software Engineering", options)
ba = st.selectbox("Business Analysis", options)
comm = st.selectbox("Communication Skills", options)
ds = st.selectbox("Data Science", options)
ts = st.selectbox("Troubleshooting Skills", options)
gd = st.selectbox("Graphics Designing", options)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Job Role"):

    input_data = pd.DataFrame([{
        "Database Fundamentals": skill_mapping[db],
        "Computer Architecture": skill_mapping[ca],
        "Distributed Computing Systems": skill_mapping[dcs],
        "Cyber Security": skill_mapping[cs],
        "Networking": skill_mapping[net],
        "Software Development": skill_mapping[sd],
        "Programming Skills": skill_mapping[ps],
        "Project Management": skill_mapping[pm],
        "Computer Forensics Fundamentals": skill_mapping[cff],
        "Technical Communication": skill_mapping[tc],
        "AI ML": skill_mapping[aiml],
        "Software Engineering": skill_mapping[se],
        "Business Analysis": skill_mapping[ba],
        "Communication skills": skill_mapping[comm],
        "Data Science": skill_mapping[ds],
        "Troubleshooting skills": skill_mapping[ts],
        "Graphics Designing": skill_mapping[gd]
    }])

    prediction = model.predict(input_data)
    predicted_role = role_encoder.inverse_transform(prediction)

    st.subheader("Prediction Result")
    st.success(predicted_role[0])