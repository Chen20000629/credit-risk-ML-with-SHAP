import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess(df):
    df = df.dropna()

    X = df.drop("SeriousDlqin2yrs", axis=1)
    y = df["SeriousDlqin2yrs"]

    return X, y


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=100)