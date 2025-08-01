# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# إعداد واجهة التطبيق
st.set_page_config(page_title="🎯 توقع السرعة", layout="centered")
st.title("🚀 توقع السرعة بناءً على رقم الجولة")

# رفع ملف البيانات
uploaded_file = st.file_uploader("📂 ارفع ملف Excel يحتوي على البيانات", type=["xlsx"])
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    # معالجة البيانات
    df.columns = ['row_id', 'round_number', 'speed']
    df.drop(columns=['row_id'], inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)

    st.write("📊 بيانات التدريب:")
    st.dataframe(df.head())

    # تحديد المدخلات والمخرجات
    X = df[['round_number']]
    y = df['speed']

    # تدريب النماذج
    lr = LinearRegression().fit(X, y)
    dt = DecisionTreeRegressor().fit(X, y)
    rf = RandomForestRegressor().fit(X, y)

    # عرض نتائج دقة النماذج
    st.subheader("📈 تقييم أداء النماذج")
    st.write(f"🔹 Linear Regression R² Score: {r2_score(y, lr.predict(X)):.2f}")
    st.write(f"🔸 Decision Tree R² Score: {r2_score(y, dt.predict(X)):.2f}")
    st.write(f"🔹 Random Forest R² Score: {r2_score(y, rf.predict(X)):.2f}")

    # إدخال رقم الجولة للتوقع
    round_num = st.number_input("🎯 ادخل رقم الجولة للتوقع", min_value=1)
    if st.button("احسب السرعة المتوقعة"):
        input_df = pd.DataFrame({'round_number': [round_num]})
        prediction_lr = lr.predict(input_df)[0]
        prediction_dt = dt.predict(input_df)[0]
        prediction_rf = rf.predict(input_df)[0]

        st.success(
            f"✅ توقعات السرعة للجولة {round_num}:\n"
            f"🔹 Linear: {prediction_lr:.2f} | 🔸 Tree: {prediction_dt:.2f} | 🔹 Forest: {prediction_rf:.2f}"
        )

        # حفظ التوقعات في ملف Excel
        result_df = pd.DataFrame([
            {"model": "Linear", "round_number": round_num, "predicted_speed": prediction_lr},
            {"model": "DecisionTree", "round_number": round_num, "predicted_speed": prediction_dt},
            {"model": "RandomForest", "round_number": round_num, "predicted_speed": prediction_rf}
        ])
        result_df.to_excel("prediction_result.xlsx", index=False)
        st.info("💾 تم حفظ التوقعات في ملف prediction_result.xlsx")
        تنظيف الكود وتحديث النموذج
