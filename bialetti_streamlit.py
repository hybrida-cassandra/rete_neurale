


import streamlit as st
import pandas as pd

import plotly.graph_objects as go


@st.cache(allow_output_mutation=True)
def load_model():
    print('loading model')
    return load('Rete_Neurale_Bialetti.joblib')


@st.cache()
def load_data():
    print('loading data')
    df = pd.read_csv('/Users/gabriele/PycharmProjects/pythonProject1/Df_Bialetti.csv')
    return df


model = load_model()
df = load_data()
st.write(model)
st.title('Cassandra - Bialetti ')


user_input= {}




categorical = ['day_week', 'month']

for feat in categorical:
    unique_values = df[feat].unique()
    user_input[feat]=st.sidebar.selectbox(feat, unique_values)

numerical = ['Spesa Facebook','google','Organico']

for feat in numerical:
    v_min = float(df[feat].min())
    v_max =float(df[feat].max())
    user_input[feat]=st.sidebar.slider(
        feat,
        min_value= v_min,
        max_value=v_max,
        value= (v_min+v_max)/2
    )
X = pd.DataFrame([user_input])
st.write(X)

prediction =model.predict(X)
st.write(prediction)

fig= go.Figure(
    go.Indicator(
        mode= 'gauge+number',
        value=prediction[0]
    )
)

st.plotly_chart(fig)
