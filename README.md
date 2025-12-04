## Music Subscription Cancellation – Streamlit App

This project is a small web app built with **Streamlit** that predicts whether a music streaming customer is likely to cancel their subscription.  
It is based on a course final project where I did data cleaning, feature engineering, and a simple machine learning model.

### What the app does

- Loads customer and listening history data.
- Creates features such as:
  - Whether the customer cancelled
  - Whether they had a discount
  - Number of listening sessions
  - Percent of listening that is Pop and Podcasts
- Trains a logistic regression model.
- Lets you interactively:
  - Explore the data
  - Change inputs (discount, sessions, listening mix)
  - See the predicted cancellation probability.
