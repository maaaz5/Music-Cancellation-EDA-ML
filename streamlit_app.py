import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression

def load_data():
  customers_path = "data/maven_music_customers.csv"
  history_path = "data/maven_music_listening_history.xlsx"

  customers = pd.read_csv(customers_path)
  history = pd.read_excel(history_path)

  return customers, history


def build_features(customers,history):
  customers = customers.copy()

  #convert dates 
  customers['Member Since'] = pd.to_datetime(customers['Member Since'])
  customers["Cancellation Date"] = pd.to_datetime(customers["Cancellation Date"])

  customers["Cancelled"] = customers["Cancellation Date"].notna().astype(int)

  customers['Discount?'] = np.where(customers['Discount?'] == 'Yes', 1, 0)

  sessions_per_customer = (
    history.groupby('Customer ID')['Session ID']
    .nunique()
    .rename('Number of Sessions')
    .reset_index()
  )

  #merge into customers
  customers = customers.merge(sessions_per_customer,how='left',on='Customer ID')


  return customers


def train_simple_model(model_df):
    # Use Number of Sessions to predict Cancelled
    X = model_df[["Number of Sessions"]]
    y = model_df["Cancelled"]

    model = LogisticRegression()
    model.fit(X, y)

    return model



def main():
  st.title('Music Subscription Cancellation App')
  st.write('testing this streamlit app')

  st.subheader('Raw data preview')

  #loading the data
  customers, history = load_data()

  st.write('Customers:')
  st.dataframe(customers.head())

  st.write('Listening history:')
  st.dataframe(history.head())

  #build the features
  customers_features = build_features(customers, history)

  st.subheader('Customers with engineered features')
  st.dataframe(customers_features.head())

  #simple eda 
  st.subheader('Cancellation rate')
  cancellation_rate = customers_features["Cancelled"].mean()
  st.write(f"Overall cancellation rate: {cancellation_rate:.2%}")


  #building the model df
  model_df = customers_features[['Customer ID','Cancelled','Discount?','Number of Sessions']].dropna()

  st.subheader('Modeling Data (sample)')
  st.dataframe(model_df.head())

  #training the model
  model = train_simple_model(model_df)

  st.subheader('Try as simple prediction')

  #widgets
  min_sessions = int(model_df['Number of Sessions'].min())
  max_sessions = int(model_df['Number of Sessions'].max())
  default_sessions = int(model_df['Number of Sessions'].mean())

  chosen_sessions = st.slider('Number of sessions in last 3 months',
  min_value=min_sessions,
  max_value=max_sessions,
  value=default_sessions)

  if st.button('Predict the cancellation probability'):
    X_new = pd.DataFrame({'Number of Sessions': [chosen_sessions]})
    prob_cancel = model.predict_proba(X_new)[0][1]

    st.write(f"Predicted probability of cancellation: {prob_cancel:.2%}")




 



if __name__ == '__main__':
  main()