import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def encode_categorical(df):
    categorical_cols = df.select_dtypes(include=['object']).columns.to_list()
    ordinal_encoder = OrdinalEncoder()
    encoded_categorical_cols = ordinal_encoder.fit_transform(df[categorical_cols])
    encoded_categorical_df = pd.DataFrame(
        encoded_categorical_cols,
        columns=categorical_cols
    )
    numerical_df = df.drop(columns=categorical_cols, axis=1)
    encoded_df = pd.concat([numerical_df, encoded_categorical_df], axis=1)
    return encoded_df

def normalize(df):
    normalizer = StandardScaler()
    return normalizer.fit_transform(df)

def split_data(df, test_size=0.3, random_state=1, shuffle=True):
    X, y = df[:, 1:], df[:, 0]
    return train_test_split(X, y, 
                            test_size=test_size,
                            random_state=random_state,
                            shuffle=shuffle)