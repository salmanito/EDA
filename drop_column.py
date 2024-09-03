def drop_column(dataframe, column_name):
    if column_name in dataframe.columns:
        return dataframe.drop(columns=[column_name])
    else:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")
