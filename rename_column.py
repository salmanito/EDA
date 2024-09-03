def rename_column(dataframe, old_name, new_name):
    if old_name in dataframe.columns:
        return dataframe.rename(columns={old_name: new_name}, inplace=False)
    else:
        raise ValueError(f"Column name '{old_name}' not found in DataFrame.")
