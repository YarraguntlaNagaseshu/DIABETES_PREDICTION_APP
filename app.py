import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Diabetes Prediction Dashboard",
    page_icon="🩺",
    layout="wide"
)

# ==========================================================
# LOAD MODEL & DATA
# ==========================================================
model = joblib.load("diabetes_logistic_model.pkl")
columns = joblib.load("columns.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")

df = load_data()

# ==========================================================
# SIDEBAR INPUTS
# ==========================================================
st.sidebar.title("🩺 Patient Health Details")

HighBP = st.sidebar.selectbox("High Blood Pressure", [0,1])
HighChol = st.sidebar.selectbox("High Cholesterol", [0,1])
CholCheck = st.sidebar.selectbox("Cholesterol Check", [0,1])
BMI = st.sidebar.slider("BMI",10,60,25)
Smoker = st.sidebar.selectbox("Smoker", [0,1])
Stroke = st.sidebar.selectbox("Stroke History", [0,1])
HeartDiseaseorAttack = st.sidebar.selectbox("Heart Disease or Attack", [0,1])
PhysActivity = st.sidebar.selectbox("Physical Activity", [0,1])
HvyAlcoholConsump = st.sidebar.selectbox("Heavy Alcohol Consumption", [0,1])
GenHlth = st.sidebar.slider("General Health (1=Best,5=Worst)",1,5,3)
MentHlth = st.sidebar.slider("Mental Health Days",0,30,5)
PhysHlth = st.sidebar.slider("Physical Health Days",0,30,5)
DiffWalk = st.sidebar.selectbox("Difficulty Walking", [0,1])
Sex = st.sidebar.selectbox("Sex (0=Female,1=Male)", [0,1])
Age = st.sidebar.slider("Age Category",1,13,6)

predict_clicked = st.sidebar.button("🔍 Predict Diabetes")

# ==========================================================
# NAVIGATION
# ==========================================================
if "page" not in st.session_state:
    st.session_state.page = "overview"

col1, col2, col3, col4 = st.columns(4)

if col1.button("🏠 Overview"):
    st.session_state.page = "overview"

if col2.button("📊 Dataset"):
    st.session_state.page = "dataset"

if col3.button("📈 EDA"):
    st.session_state.page = "eda"

if col4.button("🧠 Prediction"):
    st.session_state.page = "prediction"

st.markdown("---")

# ==========================================================
# OVERVIEW PAGE
# ==========================================================
if st.session_state.page == "overview":

    st.title("🩺 Diabetes Prediction Dashboard")

    st.write("""
This machine learning application predicts whether a patient is **likely to have diabetes**.

### Project Workflow
✔ Data Cleaning  
✔ Outlier Handling  
✔ Feature Scaling  
✔ Model Training  
✔ Hyperparameter Tuning  
✔ Model Deployment using Streamlit
""")

# ==========================================================
# DATASET PAGE
# ==========================================================
elif st.session_state.page == "dataset":

    st.title("📊 Dataset Overview")

    st.dataframe(df.head(), use_container_width=True)

    st.write("Dataset Shape:", df.shape)

    st.write("Column Information")

    st.write(df.dtypes)

# ==========================================================
# EDA PAGE
# ==========================================================
elif st.session_state.page == "eda":

    st.title("📈 Exploratory Data Analysis")

    fig1 = px.histogram(
        df,
        x="BMI",
        nbins=30,
        title="BMI Distribution"
    )

    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(
        df,
        x="Age",
        nbins=15,
        title="Age Distribution"
    )

    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.bar(
        df.groupby("Sex")["Diabetes_012"].mean().reset_index(),
        x="Sex",
        y="Diabetes_012",
        title="Diabetes Rate by Gender"
    )

    st.plotly_chart(fig3, use_container_width=True)

# ==========================================================
# PREDICTION PAGE
# ==========================================================
elif st.session_state.page == "prediction":

    st.title("🧠 Diabetes Prediction")

    if predict_clicked:

        input_dict = {
            "HighBP": HighBP,
            "HighChol": HighChol,
            "CholCheck": CholCheck,
            "BMI": BMI,
            "Smoker": Smoker,
            "Stroke": Stroke,
            "HeartDiseaseorAttack": HeartDiseaseorAttack,
            "PhysActivity": PhysActivity,
            "HvyAlcoholConsump": HvyAlcoholConsump,
            "GenHlth": GenHlth,
            "MentHlth": MentHlth,
            "PhysHlth": PhysHlth,
            "DiffWalk": DiffWalk,
            "Sex": Sex,
            "Age": Age
        }

        input_data = pd.DataFrame([input_dict])

        # Match model columns
        input_data = input_data.reindex(columns=columns, fill_value=0)

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"⚠ High Risk of Diabetes ({probability*100:.2f}% probability)")
        else:
            st.success(f"✅ Low Risk of Diabetes ({(1-probability)*100:.2f}% probability)")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability*100,
            title={"text": "Diabetes Risk (%)"},
            gauge={"axis": {"range": [0,100]}}
        ))

        st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;font-weight:600;'>Diabetes Prediction App | Machine Learning & Streamlit 🩺</p>",
    unsafe_allow_html=True
)