# %% [markdown]
# ### **About Dataset**
# Source: https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset
# 
# **Features:-**
# * step: represents a unit of time where 1 step equals 1 hour
# * type: type of online transaction
# * amount: the amount of the transaction
# * nameOrig: customer starting the transaction
# * oldbalanceOrg: balance before the transaction
# * newbalanceOrig: balance after the transaction
# * nameDest: recipient of the transaction
# * oldbalanceDest: initial balance of recipient before the transaction
# * newbalanceDest: the new balance of recipient after the transaction
# * isFraud: fraud transaction

# %%
import networkx as nx
import os
import joblib
from huggingface_hub import hf_hub_download

HF_TOKEN = os.getenv("HF_TOKEN")

# %%
def feature_engineering(dataframe):
  print("\n\n=> FEATURE ENGINEERING")
  print("-------------------------------")

  print("\n-> Encode categorcal variables")
  dataframe['type_CASH_OUT'] = (dataframe['type'] == 'CASH_OUT').astype(int)
  dataframe['type_DEBIT'] = (dataframe['type'] == 'DEBIT').astype(int)
  dataframe['type_PAYMENT'] = (dataframe['type'] == 'PAYMENT').astype(int)
  dataframe['type_TRANSFER'] = (dataframe['type'] == 'TRANSFER').astype(int)
  print(dataframe.head())

  dataframe['txn_freq_user'] = dataframe.groupby('nameOrig')['amount'].transform('count')
  dataframe['avg_amount_user'] = dataframe.groupby('nameOrig')['amount'].transform('mean')
  dataframe['txn_freq_dest'] = dataframe.groupby('nameDest')['amount'].transform('count')
  dataframe['avg_amount_dest'] = dataframe.groupby('nameDest')['amount'].transform('mean')
  print("\n-> (1/7) Added user and destination transaction frequency and average amount features.")

  # Network-based features
  G = nx.from_pandas_edgelist(dataframe, "nameOrig", "nameDest", create_using=nx.DiGraph())
  dataframe['org_sender_degree'] = dataframe['nameOrig'].map(lambda x: G.out_degree(x))
  dataframe['org_receiver_degree'] = dataframe['nameDest'].map(lambda x: G.in_degree(x))
  dataframe['dest_sender_degree'] = dataframe['nameDest'].map(lambda x: G.out_degree(x))
  dataframe['dest_receiver_degree'] = dataframe['nameDest'].map(lambda x: G.in_degree(x))
  print("\n-> (2/7) Added network-based features.")

  dataframe['sender_to_receiver_ratio'] = dataframe['org_sender_degree'] / (dataframe['org_receiver_degree'] + 1)
  dataframe['unique_receivers'] = dataframe.groupby('nameOrig')['nameDest'].transform('nunique')
  dataframe['unique_senders'] = dataframe.groupby('nameDest')['nameOrig'].transform('nunique')
  print("\n-> (3/7) Added sender to receiver ratio and unique sender/receiver features.")

  # Night transactions
  dataframe['hour'] = dataframe['step'] % 24  # Extract hour from step (since 1 step = 1 hour)
  dataframe['is_night'] = dataframe['hour'].apply(lambda x: 1 if (x < 6 or x > 20) else 0)

  dataframe['amount_ratio'] = dataframe['amount'] / (dataframe['oldbalanceOrg'] + 1)  # Avoid division by zero
  dataframe['sender_balance_change'] = dataframe['oldbalanceOrg'] - dataframe['newbalanceOrig']
  dataframe['receiver_balance_change'] = dataframe['newbalanceDest'] - dataframe['oldbalanceDest']
  print("\n-> (4/7) Added night transaction and balance change features.")

  # Behavioral flags
  dataframe['orig_balance_zero'] = (dataframe['oldbalanceOrg'] == 0).astype(int)
  dataframe['balance_drained'] = (dataframe['newbalanceOrig'] <= 0).astype(int)
  dataframe['dest_balance_zero'] = (dataframe['oldbalanceDest'] == 0).astype(int)
  dataframe['high_risk_type'] = ((dataframe['type'] == 'TRANSFER') | (dataframe['type'] == 'CASH_OUT')).astype(int)
  print("\n-> (5/7) Added behavioral flag features.")

  # Amount buckets
  dataframe['amt_less_than_10000'] = (dataframe['amount'] < 10000).astype(int)
  dataframe['amt_greater_than_10000'] = ((dataframe['amount'] >= 10000) & (dataframe['amount'] < 50000)).astype(int)
  dataframe['amt_greater_than_50000'] = ((dataframe['amount'] >= 50000) & (dataframe['amount'] < 100000)).astype(int)
  dataframe['amt_greater_than_100000'] = ((dataframe['amount'] >= 100000) & (dataframe['amount'] < 500000)).astype(int)
  dataframe['amt_greater_than_500000'] = (dataframe['amount'] > 500000).astype(int)
  print("\n-> (6/7) Added amount bucket features.")

  # Transaction consistency check
  dataframe['org_balance_mismatch'] = (
      (dataframe['oldbalanceOrg'] - dataframe['amount']) != dataframe['newbalanceOrig']
  ).astype(int)
  dataframe['dest_balance_mismatch'] = (
      (dataframe['oldbalanceDest'] + dataframe['amount']) != dataframe['newbalanceDest']
  ).astype(int)
  print("\n-> (7/7) Added transaction consistency check features.")

  extra_features = ['nameOrig', 'nameDest', 'newbalanceOrig', 'newbalanceDest', 'org_sender_degree', 'org_receiver_degree', 'step', 'type']

  features_to_drop = [col for col in extra_features if col in dataframe.columns]

  if features_to_drop:
      dataframe.drop(features_to_drop, axis=1, inplace=True)

  print("\n-> Removed unnecessary features - ", features_to_drop)

  print("\n\nFinished feature engineering -")
  print(dataframe.head())

  return dataframe

# %%
def data_preparation(
    dataframe,
    scaler_path=hf_hub_download(
        repo_id="sandalisingh/upi-fraud-models",
        filename="scaler.pkl",
        token=HF_TOKEN
    )
):
    print("\n\n=> DATA PREPARATION")
    print("-------------------------------")

    scaler = joblib.load(scaler_path)

    # Split features and labels
    feature_names = dataframe.columns.tolist()
    dataframe = scaler.transform(dataframe)

    return dataframe, feature_names
# %%
