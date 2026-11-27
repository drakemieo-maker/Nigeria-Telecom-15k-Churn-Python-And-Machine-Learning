import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. Page Configuration
st.set_page_config(page_title="Telecom 15k Churn & ML Predictor", page_icon="📡", layout="wide")

st.title("📡 Nigerian Telecoms: 15,000-Subscriber Churn Analytics & ML Predictor")
st.markdown("**Built by Oghenemarho Ojagbedavwe** | Python (`Pandas`, `Scikit-Learn`, `Streamlit`) | Live Executive Decision App")

# 2. Load 15,000-Row Cleaned Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Telecom_15k_Customer_Churn.csv")
    df['City'] = df['City'].str.strip().str.title()
    df['Monthly_Spend_Naira'] = df['Monthly_Spend_Naira'].fillna(df['Monthly_Spend_Naira'].median())
    return df

df = load_data()

# 3. Train Live Scikit-Learn Random Forest Model
@st.cache_resource
def train_model(data):
    features = ['Tenure_Months', 'Monthly_Spend_Naira', 'Dropped_Calls_Monthly', 'Network_Downtime_Hours', 'Customer_Support_Calls']
    X = data[features]
    y = np.where(data['Account_Status'] == 'Churned', 1, 0)
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X, y)
    return model

model = train_model(df)

# 4. Sidebar Filters for the Executive Dashboard
st.sidebar.header("🔎 Filter 15,000 Subscribers")
selected_network = st.sidebar.multiselect("Select Network Provider:", options=df['Network_Provider'].unique(), default=df['Network_Provider'].unique())
selected_city = st.sidebar.multiselect("Select City:", options=df['City'].unique(), default=df['City'].unique())

filtered_df = df[(df['Network_Provider'].isin(selected_network)) & (df['City'].isin(selected_city))]

# 5. Top Executive KPI Cards
total_subs = len(filtered_df)
churned_subs = (filtered_df['Account_Status'] == 'Churned').sum()
churn_rate = (churned_subs / total_subs * 100) if total_subs > 0 else 0
lost_mrr = filtered_df[filtered_df['Account_Status'] == 'Churned']['Monthly_Spend_Naira'].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Subscribers", f"{total_subs:,}")
col2.metric("Churned Subscribers", f"{churned_subs:,}")
col3.metric("Churn Rate (%)", f"{churn_rate:.2f}%")
col4.metric("Monthly Revenue Lost", f"₦{lost_mrr:,.0f}")

st.divider()

# 6. Interactive Charts Side-by-Side
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.subheader("📉 Monthly Revenue Lost to Churn by City (₦)")
    city_loss = filtered_df[filtered_df['Account_Status'] == 'Churned'].groupby('City')['Monthly_Spend_Naira'].sum().sort_values(ascending=False)
    st.bar_chart(city_loss, color="#e63946")

with chart_col2:
    st.subheader("⏱️ Avg Network Downtime (Hours): Active vs Churned")
    downtime_avg = filtered_df.groupby('Account_Status')['Network_Downtime_Hours'].mean()
    st.bar_chart(downtime_avg, color="#2a9d8f")

st.divider()

# 7. LIVE MACHINE LEARNING CHURN SIMULATOR (Interactive Sliders!)
st.header("🤖 Live AI Churn Risk Predictor (Scikit-Learn Random Forest)")
st.markdown("Move the sliders below to simulate any telecom subscriber and predict their real-time probability of churning:")

s1, s2, s3, s4, s5 = st.columns(5)
tenure = s1.slider("Tenure (Months)", 1, 60, 12)
spend = s2.selectbox("Monthly Spend (₦)", [3500, 6000, 12000, 25000, 50000], index=2)
dropped = s3.slider("Dropped Calls / Mo", 0, 15, 4)
downtime = s4.slider("Network Downtime (Hrs)", 0.0, 40.0, 15.0)
support = s5.slider("Support Calls / Mo", 0, 10, 2)

input_data = pd.DataFrame([[tenure, spend, dropped, downtime, support]],
                          columns=['Tenure_Months', 'Monthly_Spend_Naira', 'Dropped_Calls_Monthly', 'Network_Downtime_Hours', 'Customer_Support_Calls'])
churn_prob = model.predict_proba(input_data)[0][1] * 100

if churn_prob >= 50:
    st.error(f"🚨 HIGH CHURN RISK: **{churn_prob:.1f}% Probability of Churning!** (Recommended Action: Issue Proactive 5GB Apology Data Bonus & Escalate Network Ticket)")
else:
    st.success(f"✅ LOW CHURN RISK: **{churn_prob:.1f}% Probability of Churning.** (Subscriber is healthy and likely to remain Active.)")
