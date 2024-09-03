def correct_spelling(df, column, misspelled, corrected):
    df[column] = df[column].replace(misspelled, corrected)
    return df
