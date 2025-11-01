# -*- coding: utf-8 -*-
"""Data preprocessing module for text cleaning and transformation."""

import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pathlib import Path
from src.logger import logging
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)


def preprocess_dataframe(df, col='text'):
    """
    Preprocess a DataFrame by applying text cleaning to a column.

    Args:
        df (pd.DataFrame): The DataFrame to preprocess.
        col (str): The name of the column containing text.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    def preprocess_text(text):
        """Clean and normalize text data."""
        if pd.isna(text):
            return text

        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = ''.join([char for char in text if not char.isdigit()])
        text = text.lower()
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = text.replace('؛', "")
        text = re.sub(r'\s+', ' ', text).strip()

        text = " ".join(
            [word for word in text.split() if word not in stop_words]
        )
        text = " ".join(
            [lemmatizer.lemmatize(word) for word in text.split()]
        )
        return text

    df = df.copy()
    df[col] = df[col].apply(preprocess_text)
    df = df.dropna(subset=[col])

    logging.info("Data preprocessing completed for column '%s'", col)
    return df


def main():
    """Load, preprocess, and save train and test datasets."""
    try:
        project_root = Path(__file__).resolve().parents[2]
        train_path = project_root / 'data' / 'raw' / 'train.csv'
        test_path = project_root / 'data' / 'raw' / 'test.csv'

        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logging.info('Raw data loaded successfully')

        train_processed = preprocess_dataframe(train_data, 'review')
        test_processed = preprocess_dataframe(test_data, 'review')

        output_path = project_root / 'data' / 'interim'
        output_path.mkdir(parents=True, exist_ok=True)

        train_processed.to_csv(
            output_path / 'train_processed.csv',
            index=False
        )
        test_processed.to_csv(
            output_path / 'test_processed.csv',
            index=False
        )

        logging.info('Processed data saved to %s', output_path)

    except FileNotFoundError as e:
        logging.error('Data file not found: %s', e)
        raise
    except Exception as e:
        logging.error('Data preprocessing failed: %s', e)
        raise


if __name__ == '__main__':
    main()
