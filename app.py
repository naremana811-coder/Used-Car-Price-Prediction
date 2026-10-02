import streamlit as st
import pandas as pd
import joblib

# عنوان التطبيق
st.title("🚗 Used Car Price Prediction App")
st.write("أدخلي بيانات السيارة للحصول على السعر المتوقع:")

# تحميل الموديل والبيانات النظيفة
@st.cache_resource
def load_assets():
    model = joblib.load('car_price_model.pkl')
    df = pd.read_csv('cleaned_used_cars.csv')
    return model, df

model, df = load_assets()

# مدخلات المستخدم
milage = st.number_input("المشي / الممشى (Mileage):", min_value=0, value=50000, step=1000)
car_age = st.number_input("عمر السيارة بالسنوات (Car Age):", min_value=0, value=5, step=1)

# زر التوقع
if st.button("توقع السعر 💰"):
    try:
        # تجهيز الأعمدة التي تدرب عليها الموديل
        features = ['milage_clean', 'car_age', 'brand', 'fuel_type', 'accident', 'clean_title']
        features_df = pd.get_dummies(df[features], drop_first=True)

        # أخذ صف افتراضي لنسخ الأعمدة والتأكد من تطابق الشكل
        sample_input = features_df.iloc[[0]].copy()

        # تحديث قيم الممشى وعمر السيارة بالقيم التي أدخلها المستخدم
        if 'milage_clean' in sample_input.columns:
            sample_input['milage_clean'] = milage
        if 'car_age' in sample_input.columns:
            sample_input['car_age'] = car_age

        # تنفيذ التوقع
        prediction = model.predict(sample_input)[0]
        st.success(f"السعر المتوقع للسيارة: **${prediction:,.2f}**")
    except Exception as e:
        st.error(f"حدث خطأ أثناء التوقع: {e}")