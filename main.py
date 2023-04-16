import streamlit as st
import pandas as pd
import joblib


def get_data():
    pipeline = joblib.load('titanic_pipeline_RF.joblib')
    return pipeline


def main():
    st.sidebar.title('Titanic Survival Prediction')

    sex = st.sidebar.selectbox(label='Sex', options=['male', 'female'])
    age = st.sidebar.slider(label='Age', min_value=0, max_value=100, value=25)
    sibsp = st.sidebar.slider(label='Sibling/Spouse', min_value=0, max_value=10, value=0)
    fare = st.sidebar.number_input(label='Fare', min_value=0, max_value=100, value=25)
    _class = st.sidebar.selectbox(label='Class', options=['First', 'Second', 'Third'])
    alone = st.sidebar.radio(label='Alone', options=[1, 0])
    embarked = st.sidebar.selectbox(label='Embarked', options=['S', 'C', 'Q'])

    user_input = pd.DataFrame({
        'sex': sex,
        'age': age,
        'sibsp': sibsp,
        'fare': fare,
        'class': _class,
        'alone': alone,
        'embarked': embarked
    }, index=[0])

    st.write(user_input)

    pipeline = get_data()

    if st.button('Predict'):
        prediction = pipeline.predict(user_input)
        st.write(prediction[0])


if __name__ == '__main__':
    main()

