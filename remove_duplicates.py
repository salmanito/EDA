def remove_duplicates(df):
    # Check for duplicates
    duplicated_rows = df[df.duplicated(keep=False)]

    if not duplicated_rows.empty:
        # Drop duplicates
        df = df.drop_duplicates()
        return df, True  # Return True indicating duplicates were found and removed
    else:
        return df, False  # Return False indicating no duplicates were found
