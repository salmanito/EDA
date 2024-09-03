import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_categorical(df, column):
    if df[column].dtype == object or df[column].dtype == 'category':
        le = LabelEncoder()
        df[column + '_encoded'] = le.fit_transform(df[column])
        return df, le.classes_
    else:
        raise ValueError("Not eligible for encoding categorical values. More than 10 unique values.")
