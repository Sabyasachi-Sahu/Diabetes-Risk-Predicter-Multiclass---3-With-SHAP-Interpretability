import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# ... (Paste the rest of your code here)

st.set_page_config(page_title='Diabetes Risk Predictor',layout='wide',page_icon="🏥")
st.title('Diabetes Risk Assessment Tool')
st.markdown('Powered by CDC Survey Records')
st.divider()

# If not foud condition can be added. Ignored while practice
model_path = 'models/best_model.pkl'
model=joblib.load(model_path)
feature_cols=joblib.load('models/feature_cols.pkl')

col1,col2,col3 = st.columns(3)

with col1:
    st.subheader("🩺 Health condition")
    high_bp = st.selectbox('High Blood Pressure ?',['No','Yes'])
    high_chol = st.selectbox('High Cholestrol ?',['No','Yes'])
    chol_check = st.selectbox('Cholestrol Check (Last 5 years)?',['No','Yes'])
    stroke = st.selectbox('Ever had a stroke ?',['No','Yes'])
    heart_disease = st.selectbox('Heart Disease / Attack ?',['No','Yes'])
    diff_walk = st.selectbox('Difficulty Walking ?',['No','Yes'])


with col2:
    st.subheader("📊 Vitals & Lifestyle")
    bmi = st.slider("BMI",12,60,27)
    smoker = st.selectbox('Smoking Habits ?',['No','Yes'])
    phys_activity = st.selectbox('Physical Activity (Last 30 Days)?',['No','Yes'])
    fruits = st.selectbox('Eat fruits daily ?',['No','Yes'])
    veggies = st.selectbox('Eat veggies daily ?',['No','Yes'])
    heavy_alcohol = st.selectbox('Heavy alcohol consumption ?',['No','Yes'])

with col3:
    st.subheader("👤 Demographics")
    gen_health = st.selectbox("General Health", ["Excellent", "Very Good", "Good", "Fair", "Poor"])
    ment_hlth = st.slider("Poor Mental Health Days (month)", 0, 30, 0)
    phys_hlth = st.slider("Poor Physical Health Days (month)", 0, 30, 0)
    sex = st.selectbox("Sex", ["Female", "Male"])
    age_cat = st.selectbox("Age Group", [
        '18-24', '25-29', '30-34', '35-39', '40-44', '45-49',
        '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+'])
    education = st.selectbox("Education", [
        "Never attended", "Elementary", "Some high school",
        "High school graduate", "Some college", "College graduate"])
    income = st.selectbox("Income", [
        '<$10K', '$10-15K', '$15-20K', '$20-25K',
        '$25-35K', '$35-50K', '$50-75K', '$75K+'])
    healthcare = st.selectbox("Have Healthcare Coverage?", ["Yes", "No"])
    no_doc_cost = st.selectbox("Couldn't See Doctor (Cost)?", ["No", "Yes"])

st.divider()
yn = lambda s: 1 if s=="Yes" else 0
age_map = {'18-24':1,'25-29':2,'30-34':3,'35-39':4,'40-44':5,'45-49':6,'50-54':7,'55-59':8,'60-64':9,'65-69':10,'70-74':11,'75-79':12,'80+':13}
health_map = {"Excellent":1,"Very Good":2,"Good":3,"Fair":4,"Poor":5}
edu_map = {"Never attended":1,"Elementary":2,"Some high school":3,"High school graduate":4,"Some college":5,"College graduate":6}
inc_map = {'<$10K':1,'$10-15K':2,'$15-20K':3,'$20-25K':4,'$25-35K':5,'$35-50K':6,'$50-75K':7,'$75K+':8}

if st.button("🔍 Predict Diabetes Risk", type="primary", use_container_width=True):
    input_dict = {
        'HighBP': yn(high_bp), 'HighChol': yn(high_chol),
        'CholCheck': yn(chol_check), 'BMI': float(bmi),
        'Smoker': yn(smoker), 'Stroke': yn(stroke),
        'HeartDiseaseorAttack': yn(heart_disease),
        'PhysActivity': yn(phys_activity), 'Fruits': yn(fruits),
        'Veggies': yn(veggies), 'HvyAlcoholConsump': yn(heavy_alcohol),
        'AnyHealthcare': yn(healthcare), 'NoDocbcCost': yn(no_doc_cost),
        'GenHlth': health_map[gen_health],
        'MentHlth': float(ment_hlth), 'PhysHlth': float(phys_hlth),
        'DiffWalk': yn(diff_walk),
        'Sex': 1 if sex == "Male" else 0,
        'Age': age_map[age_cat],
        'Education': edu_map[education],
        'Income': inc_map[income],
    }
    input_df = pd.DataFrame([input_dict])
    # Align columns with training
    for c in feature_cols:
        if c not in input_df.columns:
            input_df[c] = 0
    input_df = input_df[feature_cols]

    try:
        probs = model.predict_proba(input_df)[0]
        prob_no = probs[0]
        prob_pre = probs[1]
        prob_diab = probs[2]
        pred_class = np.argmax(probs)
        
        st.divider()
        st.markdown("### 📋 Prediction Results")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("🟢 No Diabetes", f"{prob_no:.1%}")
        with c2:
            st.metric("🟡 Prediabetes", f"{prob_pre:.1%}")
        with c3:
            st.metric("🔴 Diabetes", f"{prob_diab:.1%}")
            
        st.markdown("---")
        
        if pred_class == 2:
            st.error(f"⚠️ **HIGH RISK (DIABETES INDICATED)** - Please consult a healthcare provider.")
        elif pred_class == 1:
            st.warning(f"⚡ **MODERATE RISK (PREDIABETES INDICATED)** - Lifestyle adjustments may be recommended.")
        else:
            st.success(f"✅ **LOW RISK (NO DIABETES INDICATED)** - Keep up the healthy habits!")
            st.subheader("📋 Key Risk Factors Detected")
            risks = []
            if yn(high_bp): risks.append("• High Blood Pressure")
            if yn(high_chol): risks.append("• High Cholesterol")
            if bmi >= 30: risks.append(f"• Obesity (BMI={bmi})")
            if age_map[age_cat] >= 9: risks.append(f"• Age 60+")
            if not yn(phys_activity): risks.append("• No Physical Activity")
            if yn(heart_disease): risks.append("• Heart Disease History")
            if health_map[gen_health] >= 4: risks.append("• Fair/Poor General Health")
            st.markdown("\n".join(risks) if risks else "No major risk factors detected.")
    except Exception as e:
        st.error(f"Error: {e}")
