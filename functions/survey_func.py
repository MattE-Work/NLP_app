import pandas as pd

def apply_sentiment_filters(df, sentiment_column, sentiment_score_range=None):
    """
    Applies filters to the DataFrame based on user selections related to sentiment score.

    Args:
    - df (pd.DataFrame): The DataFrame containing the text data and sentiment scores.
    - sentiment_column (str): The name of the column containing sentiment scores.
    - sentiment_score_range (tuple, optional): Tuple of (min, max) sentiment score selected by user.

    Returns:
    - filtered_df (pd.DataFrame): The DataFrame filtered based on user selections.
    """
    
    # Filter by Sentiment Score
    if sentiment_score_range:
        df = df[df[sentiment_column].between(*sentiment_score_range)]
    
    return df
