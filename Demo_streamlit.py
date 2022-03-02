import streamlit as st
import pandas as pd
import seaborn as sns
from joblib import load
import plotly.graph_objects as go
from PIL import Image

@st.cache(allow_output_mutation=True)
def load_model():
    print('Loading model...')

    return load('Rete_Neurale_Bialetti.joblib')

@st.cache()
def load_data():
    print('Loading data...')
    df = pd.read_csv('Df_Bialetti.csv')

    return df

model = load_model()
df = load_data()

st.title('Cassandra - Scenario Simulator')

st.sidebar.image(Image.open(f"./references/logo.png"), width=200)
st.sidebar.title('1. Seasonal Variables')

user_input= {}
categorical = ['day_week', 'month']

for feat in categorical:
    unique_values = df[feat].unique()
    unique_values.sort()

    label = ""
    if feat == "day_week":
        label = "Day of the week"
    elif feat == 'month':
        label = "Month"

    if feat == 'day_week':
        display = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
    elif feat == 'month':
        display = ("Si", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "Dicember")

    user_input[feat] = st.sidebar.selectbox(label, unique_values, format_func=lambda x: display[x])

st.sidebar.title('2. Media Variables')

numerical = ['Spesa Facebook', 'google', 'Organico']

for feat in numerical:
    v_min = float(df[feat].min())
    v_max =float(df[feat].max())

    label = ""
    if feat == "Spesa Facebook":
        label = "Facebook Ads Spend"
    elif feat == "google":
        label = "Google Ads Spend"
    else:
        label = "Organic Sessions"

    user_input[feat]=st.sidebar.slider(
        label,
        min_value = v_min,
        max_value = v_max,
        value = (v_min+v_max)/2
    )
X = pd.DataFrame([user_input])
#st.write(X)

z=(X['Spesa Facebook']+X['google'])
cpa= z/model.predict(X)

prediction =model.predict(X)

st.title('Predicted sales')

fig = go.Figure(
    go.Indicator(
        mode= 'gauge+number',
        value=prediction[0]
    )
)

st.plotly_chart(fig)

aov= 60
fatturato=model.predict(X)*aov
profit=fatturato*0.6-z

st.subheader("Revenue: " + str(round(fatturato[0], 2)) + " $")
st.subheader("Profit: " + str(round(profit[0], 2)) + " $")
st.subheader("CPA: " + str(round(cpa[0], 2)) + " $")
