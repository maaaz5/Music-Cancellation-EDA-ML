import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression

def load_data():
  customers_path = "data/maven_music_customers.csv"
  history_path = "data/maven_music_listening_history.xlsx"

  customers = pd.read_csv(customers_path)
  history = pd.read_excel(history_path,sheet_name=0)
  audio = pd.read_excel(history_path,sheet_name=1)

  return customers, history, audio


def build_features(customers,history, audio):
  customers = customers.copy()

  #convert dates 
  customers['Member Since'] = pd.to_datetime(customers['Member Since'])
  customers["Cancellation Date"] = pd.to_datetime(customers["Cancellation Date"])

  #cancelled flag
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

  #audio
  audio = audio.copy()
  audio['Genre'] = np.where(audio['Genre'] == 'Pop Music',"Pop",audio['Genre'])

  audio_id_split = (
        pd.DataFrame(audio["ID"].str.split("-").to_list())
        .rename(columns={0: "Type", 1: "Audio ID"})
    )
  
  audio_id_split["Audio ID"] = audio_id_split["Audio ID"].astype(int)

  audio_all = pd.concat([audio_id_split, audio],axis=1)

  #joining the history with audio info
  df = history.merge(audio_all, how='left', on='Audio ID')

  genres = (
    pd.concat([df['Customer ID'],pd.get_dummies(df['Genre'],dtype='int')],axis=1)
    .groupby('Customer ID')
    .sum()
    .reset_index()
  )
  
  total_audio = (
    history.groupby('Customer ID')['Audio ID']
    .count()
    .rename('Total Audio')
    .reset_index()
  )

  df_audio = genres.merge(total_audio, how="left", on="Customer ID")

  #Percent Pop
  customers = customers.merge(df_audio,how='left',on='Customer ID')
  customers["Percent Pop"] = customers["Pop"] / customers["Total Audio"] * 100

  #podcasts 
  comedy = customers.get("Comedy", 0)
  true_crime = customers.get("True Crime", 0)
  customers["Percent Podcasts"] = (comedy + true_crime) / customers["Total Audio"] * 100


  return customers


def train_simple_model(model_df):
    X = model_df[["Number of Sessions", "Discount?", "Percent Pop", "Percent Podcasts"]]
    y = model_df["Cancelled"]

    model = LogisticRegression()
    model.fit(X, y)

    return model



def main():
  st.title('Music Subscription Cancellation App')
  st.write('testing this streamlit app')

  st.subheader('Raw data preview')

  #loading the data
  customers, history, audio = load_data()

  st.write('Customers:')
  st.dataframe(customers.head())

  st.write('Listening history:')
  st.dataframe(history.head())

  #build the features
  customers_features = build_features(customers, history, audio)

  st.subheader('Customers with engineered features')
  st.dataframe(customers_features.head())

  #simple eda 
  st.subheader('Cancellation rate')
  cancellation_rate = customers_features["Cancelled"].mean()
  st.write(f"Overall cancellation rate: {cancellation_rate:.2%}")


  #building the model df
  model_df = customers_features[['Customer ID','Cancelled','Discount?','Number of Sessions','Percent Pop','Percent Podcasts']].dropna()

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