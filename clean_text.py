import pandas as pd
import string


def remove_punctuation(text):
    if isinstance(text, str):
        return text.translate(str.maketrans('', '', string.punctuation))
    return text


def to_lowercase(text):
    if isinstance(text, str):
        return text.lower()
    return text


def remove_numbers(text):
    if isinstance(text, str):
        return ''.join([i for i in text if not i.isdigit()])
    return text


def remove_characters(text):
    if isinstance(text, str):
        return ''.join([i for i in text if i.isdigit()])
    return text


def clean_text(df, column, cleaning_option):
    original_dtype = df[column].dtype  # Store the original data type
    df[column] = df[column].astype(str)  # Convert column to string type for cleaning

    if cleaning_option == "Removing Punctuation":
        df[column] = df[column].apply(remove_punctuation)
    elif cleaning_option == "Lowercasing":
        df[column] = df[column].apply(to_lowercase)
    elif cleaning_option == "Removing Numbers":
        df[column] = df[column].apply(remove_numbers)
    elif cleaning_option == "Removing Characters":
        df[column] = df[column].apply(remove_characters)

    # Convert the column back to its original data type if possible
    try:
        df[column] = df[column].astype(original_dtype)
    except ValueError:
        # If conversion back to original data type fails, keep it as string
        pass

    return df
