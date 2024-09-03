import pandas as pd

def discretize_column(df, column, categories, ranges):
    if len(categories) != len(ranges) - 1:
        raise ValueError("Number of categories should be equal to the number of ranges minus one.")

    # Ensure the last bin includes the maximum value
    bins = ranges[:-1] + [float('inf')]  # Use infinity to ensure the max value is included in the last bin
    labels = categories

    # Discretize the data
    df[column + '_category'] = pd.cut(df[column], bins=bins, labels=labels, include_lowest=True, right=False)

    return df
