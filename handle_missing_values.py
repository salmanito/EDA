import pandas as pd
from sklearn.tree import DecisionTreeRegressor


def fill_with_mean(df, column):
    df[column].fillna(df[column].mean(), inplace=True)
    return df


def fill_with_median(df, column):
    df[column].fillna(df[column].median(), inplace=True)
    return df


def fill_with_mode(df, column):
    df[column].fillna(df[column].mode()[0], inplace=True)
    return df


def fill_with_custom_value(df, column, value):
    df[column].fillna(value, inplace=True)
    return df


def fill_with_decision_tree(df, column):
    df_copy = df.copy()
    missing_values = df_copy[column].isnull()

    # Ensure the column is numeric
    if not pd.api.types.is_numeric_dtype(df_copy[column]):
        raise ValueError("Decision tree method can only be applied to numeric columns.")

    # Use only numeric columns for training
    numeric_cols = df_copy.select_dtypes(include=['number']).columns.tolist()
    numeric_cols.remove(column)

    df_train = df_copy.loc[~missing_values]
    df_predict = df_copy.loc[missing_values]

    X_train = df_train[numeric_cols]
    y_train = df_train[column]

    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    X_predict = df_predict[numeric_cols]
    df_copy.loc[missing_values, column] = model.predict(X_predict)

    return df_copy


def drop_missing_values(df, column):
    df.dropna(subset=[column], inplace=True)
    return df
