import pandas as pd

def split_data_ranges(df, column, split_options):
    def split_range(value, option):
        try:
            lower, upper = map(int, value.split('-'))
            if option == "Lower":
                return lower
            elif option == "Upper":
                return upper
            elif option == "Average":
                return (lower + upper) / 2
        except:
            return None

    for option in split_options:
        new_column = f"{column}_{option.lower()}"
        df[new_column] = df[column].apply(lambda x: split_range(x, option))

    return df
