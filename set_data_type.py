def set_data_type(df, column, new_dtype):
    try:
        df[column] = df[column].astype(new_dtype)
        return df
    except Exception as e:
        raise ValueError(f"Error converting column {column} to {new_dtype}: {e}")
