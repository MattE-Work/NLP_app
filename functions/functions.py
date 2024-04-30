

import streamlit as st
from textblob import TextBlob
import pandas as pd
import altair as alt
import random
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import spacy
import math 

#translation library - needs installing into the environment
#from googletrans import Translator

#alternative preferred translation option on local machine
#from argostranslate import package, translate

#testing translation
#import argostranslate.package
#import argostranslate.translate

#from argostranslate.tags import DEFAULT

#language checking
from langdetect import detect, LangDetectException
import pycountry #library to conver language codes to language names

#removing stop words
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from gensim.utils import simple_preprocess


#text classification
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

#libraries for anomaly detection
from sklearn.ensemble import IsolationForest

#NLP sentiment analysis libraries
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#download lexicon if not already downloaded
nltk.download('vader_lexicon')


def generate_summary_dataframe(df, column_names):
    """
    Generate a summary dataframe showing the count of rows for each unique combination of values in the specified columns.

    Parameters:
    - df: pandas DataFrame, the source dataframe
    - column_names: list of strings or lists, the column names or lists of column names to be included in the summary

    Returns:
    - summary_df: pandas DataFrame, the summary dataframe
    """
    if not column_names:
        raise ValueError("Column names list cannot be empty.")

    # Ensure all elements in column_names are either strings or lists of strings
    valid_column_names = []
    for item in column_names:
        if isinstance(item, str):
            valid_column_names.append(item)
        elif isinstance(item, list) and all(isinstance(subitem, str) for subitem in item):
            valid_column_names.extend(item)
        else:
            raise ValueError("Invalid element in column_names list. Each element should be a string or a list of strings.")

    # Extract relevant columns from the dataframe
    selected_columns = df[valid_column_names]

    # Group by the selected columns and count the number of rows for each unique combination
    summary_df = selected_columns.groupby(valid_column_names).size().reset_index(name='Count')

    return summary_df



def get_all_testing_scenarios(var1_list_options, var2_list_options, var3_list_options):
    list_unique_test_combinations = []
    for x in var1_list_options:
        for y in var2_list_options:
            for z in var3_list_options:
                list_unique_test_combinations.append([x,y,z])
    
    df_testing_options = pd.DataFrame(list_unique_test_combinations)

    return list_unique_test_combinations, df_testing_options




def convert_to_df(sentiment):
    sentiment_dict = {
        'polarity': sentiment.polarity,
        'subjectivity': sentiment.subjectivity
    }
    sentiment_df = pd.DataFrame(sentiment_dict.items(), columns=['metric', 'value'])
    return sentiment_df


def analyse_token_sentiment(docx):
    analyser = SentimentIntensityAnalyzer()
    pos_list = []
    neg_list = []
    neu_list = []

    for i in docx.split():
        res = analyser.polarity_scores(i)['compound']
        if res >= 0.1:
            pos_list.append(i)
            pos_list.append(res)
        elif res <= -0.1:
            neg_list.append(i)
            neg_list.append(res)
        else:
            neu_list.append(i)
            #neu_list.append(res)

    result_dict = {
        'positives': pos_list,
        'negatives': neg_list,
        'neutral': neu_list
        }
    return result_dict

def overview_to_app():
    st.title(':green[Natural Language Processing Qualitative Analysis]')
    st.subheader('Quick overview to this app:')
    st.write('This app provides functionality to undertake qualitative analysis, using techniques such as sentiment analysis, clustering, topic modelling, keyword extraction etc.')
    st.write("These might all be new terms to you, but don't worry! There is an example page that illustrates some of the principles.")
    
    st.subheader('Your responsibilities in using this tool:')
    st.write('This tool contains functionality to apply NLP qualitative analysis methods to text. You are free to use this but **do so at your own risk, and it is your responsibility to ensure it complies with your own organisational policies prior to your use of it.** Test data is loaded by default when no file is uploaded, so you can easily interact with the tool / see the nature of the outputs. You can of course download the code from the repo and run the tool locally.')

    st.subheader('How to use the app:')
    st.write("The app consists of several pages. Each page provides a specific function or task with regards to qualitative analysis of text. The only core or mandatory step is to run through the data prep page first, after that you can pick and choose to run 1 / some / all of the aother pages' functionality as you require.")
    
    st.subheader("Got an idea to improve the app?")
    st.write("Feature suggestions, offers to collaborate etc. all welcome - get in touch! 🫱🏼‍🫲🏾😀")

def how_nlp_works():
    #st.set_page_config(initial_sidebar_state='collapsed', layout='wide')

    #st.title('Sentiment Analysis NLP App')
    st.subheader(':green[Illustrating how NLP sentiment analysis works]')
    st.write('Adapted from: https://www.youtube.com/watch?v=3eaUFqB6Xfo&t=335s')
    
    st.write("Each word in a given string of text is analysed and given a polarity and a subjectivity score. The polarity score is used to indicate whether the word is deemed to be positive, negative or neutral in tone (or sentiment). If the polarity score is >=0.5, the word is classed as positive. If it is <= -0.5 it is classed as negative. If it is inbetween, it is deemed to be neutral. The positive and negative polarity scores are averaged to give an overall score which is used to determine the sentiment of the text as a whole. Enter some text below to see this in action.")

   
    #st.subheader('Text Entry')
    with st.form(key='nlpform'):
        raw_text = st.text_area('Enter some text here...')
        submit_button = st.form_submit_button(label='Analyse')

    #layout
    col1, col2 = st.columns(2)
    if submit_button:
        
        with col1:
            st.info('Results')
            #use text blob to determine sentiment polarity and subjectivity
            sentiment = TextBlob(raw_text).sentiment
            st.write(sentiment)

            #emoji feedback message
            if sentiment.polarity > 0:
                st.markdown('Sentiment: Positive :smiley: ')
            elif sentiment.polarity < 0:
                st.markdown('Sentiment: Negative :angry: ')
            else:
                st.markdown('Sentiment: Neutral :neutral_face: ')
            # Dataframe
            result_df = convert_to_df(sentiment)
            st.dataframe(result_df)

            #plot result visualisation
            c = alt.Chart(result_df).mark_bar().encode(
                x='metric',
                y='value',
                color='metric'
            )
            st.altair_chart(c, use_container_width=True)


        with col2:
            st.info('Token Sentiment')

            token_sentiments = analyse_token_sentiment(raw_text)
            st.write(token_sentiments)



def example_page():
    st.set_page_config(initial_sidebar_state='collapsed', layout='wide')

    st.title('Sentiment Analysis NLP App')
    st.subheader('Illustrating how NLP sentiment analysis works')
    st.write('Adapted from: https://www.youtube.com/watch?v=3eaUFqB6Xfo&t=335s')
    
    menu = ['Home', 'About']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home':
        st.subheader('Text Entry')
        with st.form(key='nlpform'):
            raw_text = st.text_area('Enter text here...')
            submit_button = st.form_submit_button(label='Analyse')

        #layout
        col1, col2 = st.columns(2)
        if submit_button:
            with col1:
                st.info('Results')
                #use text blob to determine sentiment polarity and subjectivity
                sentiment = TextBlob(raw_text).sentiment
                st.write(sentiment)

                #emoji feedback message
                if sentiment.polarity > 0:
                    st.markdown('Sentiment: Positive :smiley: ')
                elif sentiment.polarity < 0:
                    st.markdown('Sentiment: Negative :angry: ')
                else:
                    st.markdown('Sentiment: Neutral :neutral_face: ')
                # Dataframe
                result_df = convert_to_df(sentiment)
                st.dataframe(result_df)

                #plot result visualisation
                c = alt.Chart(result_df).mark_bar().encode(
                    x='metric',
                    y='value',
                    color='metric'
                )
                st.altair_chart(c, use_container_width=True)


            with col2:
                st.info('Token Sentiment')

                token_sentiments = analyse_token_sentiment(raw_text)
                st.write(token_sentiments)


#------------------------------------------
# Sentiment Analysis Tool - page functions
#------------------------------------------

def create_pie_chart(dataframe, column_label, chart_title):
    """
    Create a pie chart to show the count of unique labels in a specified column.

    Parameters:
    - dataframe: Pandas DataFrame
    - column_label: String, the label of the column to create the pie chart from.
    - chart_title: String, the title of the pie chart.

    Returns:
    - None (displays the pie chart)
    """
    fig, ax = plt.subplots()
    # Get the counts of unique values in the specified column
    counts = dataframe[column_label].value_counts()

    # Create the pie chart
    #ax.figure(figsize=(8, 8))
    
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
    #ax.title(chart_title)
    ax.set_title(chart_title)
    ax.axis('equal')

    # Display the pie chart
    #plt.show()
    return fig

# --------------------------------------------

def subset_df_for_specific_langage(df, lang_col, language):
    df_filtered = df[df[lang_col] == language]
    return df_filtered

# --------------------------------------------

def redact_review_text(df, col_with_text, redact_text='[REDACTED]'):
  """
  Function to identify name entities in a col in a df and redact these.
  Returns updated version of the source df with redacted names.
  """
  df_updated = df.copy(deep=True)

  # Load the language model
  nlp = spacy.load("en_core_web_sm")

  # Create a list to store the redacted reviews
  redacted_reviews = []

  # Process and analyze each review
  for index, row in df.iterrows():
      review = row[col_with_text]
      doc = nlp(review)

      # Initialize a redacted version of the review
      redacted_review = review

      # Identify Person entities and redact them
      for ent in doc.ents:
          if ent.label_ == "PERSON":
              redacted_review = redacted_review.replace(ent.text, redact_text)

      # Append the redacted review to the list
      redacted_reviews.append(redacted_review)

  # Update the DataFrame with the redacted reviews
  df_updated.insert(0, f'redacted_{col_with_text}', redacted_reviews)

  #delete original un-redacted column
  df_updated.drop(col_with_text, axis=1, inplace=True)

  # return the DataFrame with redacted reviews
  return df_updated

# --------------------------------------------

def create_word_cloud(df, col_name_containing_text, title, color_map):
    """
    Function to create a word cloud
    """

    # Combine all feedback text into a single string
    text = " ".join(df[col_name_containing_text])

    # Create a WordCloud object
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=color_map, random_state=42)

    # Generate the word cloud from the text data
    wordcloud.generate(text)
    fig, ax = plt.subplots()

    # Display the word cloud using Matplotlib
    #ax.figure(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(title)
    #ax.show()
    st.pyplot(fig)

# --------------------------------------------

def categorical_create_word_cloud(df, col_name_containing_text, title, color_map):
    """
    Function to create a word cloud
    """

    # Combine all feedback text into a single string
    text = " ".join(df[col_name_containing_text])

    # Create a WordCloud object
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=color_map, random_state=42)

    # Generate the word cloud from the text data
    wordcloud.generate(text)
    fig, ax = plt.subplots()

    # Display the word cloud using Matplotlib
    #ax.figure(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(title)

    #ax.show()
    #st.pyplot(fig)
    return fig

# --------------------------------------------

def detect_anomalies(df, col_name_containing_text, contamination_param_threshold = 0.05, num_outliers_to_return=10):
  """
  Function to detect text strings that appear as anomalies or outliers compared to the rest in the dataset.
  Return these as a df and a list containing the strings detected.

  Notes on the function:
  can adjust the threshold for anomaly detection based on use case specific 
  requirements and the characteristics of the data in scope. 
  If not detecting outliers or only 1 / few, can experiment with the threshold 
  or consider checking the distribution of anomaly scores to set a suitable threshold.
  """

  if st.session_state['has_redact_col'] == 'Yes':
      subset_df = df[df[st.session_state['redact_col'] != st.session_state['redact_text']]]
      df = subset_df.copy(deep=True)

  random_seed = 42
  # Text Vectorization
  tfidf_vectorizer = TfidfVectorizer(max_features=5000)
  X = tfidf_vectorizer.fit_transform(df[col_name_containing_text])

  # Apply Isolation Forest
  clf = IsolationForest(contamination=contamination_param_threshold, random_state=random_seed)  # Adjust the contamination parameter
  clf.fit(X)

  #predict method uses pre-set threshold (-1) to determine whether a text is an outlier
  # Predict anomalies (1 for inliers, -1 for outliers)
  #anomaly_scores = clf.predict(X)
  # Identify and print anomalies
  #anomalies = df[anomaly_scores == -1]

  #decision_function method used here provides a continuous anomaly score, so
  #allows more flexibility in choosing a custom threshold based on the data in use
  #(hard coded below to be less than zero)
  #Predict anomaly scores (the lower, the more abnormal) - test line
  anomaly_scores = clf.decision_function(X)
  
  # Identify and print anomalies based on the anomaly scores
  #anomalies = df[anomaly_scores < 0]
  #list_of_anomaly_text = [row[col_name_containing_text] for index, row in anomalies.iterrows()]

  # Create a DataFrame with anomaly scores and original data
  anomalies_df = df.copy(deep=True)
  anomalies_df['anomaly_score'] = anomaly_scores

  # Sort anomalies by anomaly score in descending order
  anomalies_sorted = anomalies_df.sort_values(by='anomaly_score', ascending=True) #was false, but lower scores = more abnormal

  # Limit the number of outliers returned to the top X
  top_outliers = anomalies_sorted.head(num_outliers_to_return)

  # List of the strings of the top X outliers
  list_of_anomaly_text = top_outliers[col_name_containing_text].tolist()

  return top_outliers , list_of_anomaly_text

# --------------------------------------------



# --------------------------------------------
def preprocess_reviews(df, col_with_text):
    # Initialize NLTK's English stop words
    stop_words = set(stopwords.words("english"))
    
    # Function to remove punctuation and stop words from a text
    def clean_text(text):
        # Remove punctuation
        text = "".join([char for char in text if char not in string.punctuation])
        # Tokenize the text
        tokens = text.split()
        # Remove stop words
        tokens = [word for word in tokens if word.lower() not in stop_words]
        return " ".join(tokens)

    # Apply the cleaning function to the review column
    df[col_with_text] = df[col_with_text].apply(clean_text)
    
    return df
# --------------------------------------------

def determine_sentiment(df, col_with_sentiment_text, compound_score_threshold):
  # Create a SentimentIntensityAnalyzer object
  sia = SentimentIntensityAnalyzer()

  # Apply sentiment analysis to the 'Review' column and create a new 'Sentiment' column
  df['Sentiment_Scores'] = df[col_with_sentiment_text].apply(lambda x: sia.polarity_scores(x))

  # Extract the compound score from the sentiment scores
  df['Compound_Score'] = df['Sentiment_Scores'].apply(lambda x: x['compound'])

  # Classify the compound score into 'Positive,' 'Negative,' or 'Neutral' based on a threshold
  df['Sentiment'] = df['Compound_Score'].apply(lambda score: 'Positive' if score > compound_score_threshold else ('Negative' if score < -compound_score_threshold else 'Neutral'))

  # Drop the intermediate 'Sentiment_Scores' and 'Compound_Score' columns if needed
  df.drop(columns=['Sentiment_Scores', 'Compound_Score'], inplace=True)

  return df

# --------------------------------------------

def single_df_field_selector(
        df, 
        question_text,
        help_string,
        placeholder_text = 'Choose an option'
        ):

    """
    function to insert streamlit single selection box 
    to choose one of the columns present in the df
    """

    selected_variable = st.selectbox(
        label=question_text,
        options = df.columns,
        placeholder=placeholder_text,
        help=help_string
        )
    
    return selected_variable

# --------------------------------------------

def multi_df_field_selector(
        df, 
        question_text,
        help_string,
        placeholder_text = 'Choose option(s)'
        ):

    """
    function to insert streamlit multi selection box 
    to choose n number of the columns present in the df
    """

    selected_variables = st.multiselect(
        label=question_text,
        options = df.columns,
        placeholder=placeholder_text,
        help=help_string
        )
    
    return selected_variables

# --------------------------------------------

def radio_button(label_text, list_options, horizontal_bool, default_index, help_text=None):

    """
    Function to insert streamlit radio button. Can be horizontal (bool = True)
    or vertical (bool = False)
    """

    selection = st.radio(
        label=label_text,
        options=list_options,
        horizontal=horizontal_bool,
        index=default_index,
        help=help_text
    )

    return selection
# --------------------------------------------

def preview_df(df, expander_label, direction, num_rows):
    """
    Function to preview the top 5 rows of the loaded df
    """
    with st.expander(label=expander_label):
        if direction == 'first':
            st.dataframe(df.head(num_rows))
        elif direction == 'last':
            st.dataframe(df.tail(num_rows))

    return None

# --------------------------------------------

def multi_label_selector(
        df,
        df_col,
        question_text,
        help_string,
        placeholder_text = 'Choose option(s)'
        ):

    """
    function to insert streamlit multi selection box 
    to choose n number of labels from a given column in a given df.
    can be used to isolate labels to subsequently filter a df by for e.g.
    """

    selected_labels = st.multiselect(
        label=question_text,
        options = list(set(df[df_col])),
        placeholder=placeholder_text,
        help=help_string
        )
    
    return selected_labels

# --------------------------------------------

def single_label_selector(
        df,
        df_col,
        question_text,
        help_string,
        placeholder_text = 'Choose option'
        ):

    """
    function to insert streamlit single selection box 
    to choose a specific label from a given column in a given df.
    can be used to isolate label to subsequently filter a df by for e.g.
    """

    selected_label = st.selectbox(
        label=question_text,
        options = list(set(df[df_col])),
        placeholder=placeholder_text,
        help=help_string
        )
    
    return selected_label

# --------------------------------------------

def click_button(label_text, button_key, type_str='secondary'):
    """
    Function to inset a button. This returns a bool value. Params include:
    label text = text on the button
    button_key = unique key for session state, so can have many buttons
    type_str = default secondary. change to primary for greater emphasis in button display.
    """

    bool_data_prep_confirmed = st.button(
        label=label_text, 
        key=button_key, 
        type=type_str)
    return bool_data_prep_confirmed

# --------------------------------------------

def get_dict_subset_df_to_known_sentiment(df, sentiment_col, list_pos_labels, list_neg_labels):
    """
    Function to subset the source df into two dfs, where sentiment is known.
    One df for positive text, one for negative, based on user-provided labels.
    Return a dictionary with keys (Positive or Negative) and values 
    being the respective df for these keys.
    """
    dict_sentiment = {} #dict to contain pos_df and neg_df
    
    #subset source df to given positive labels
    pos_df = df[df[sentiment_col].isin(list_pos_labels)]
    
    #subset source df to given negative labels
    neg_df = df[df[sentiment_col].isin(list_neg_labels)]

    #update dict
    dict_sentiment['Positive'] = pos_df
    dict_sentiment['Negative'] = neg_df

    return dict_sentiment

# --------------------------------------------

def get_dict_subset_df_to_known_sentiment_and_demographics(
        df, 
        sentiment_col, 
        list_pos_labels, 
        list_neg_labels,
        list_demographic_fields,
        analysis_scenario
        ):
    
    """
    Function to subset the source df into two dfs, where sentiment is known.
    One df for positive text, one for negative, based on user-provided labels.
    Return a dictionary with keys (Positive or Negative) and values 
    being the respective df for these keys.
    """
    dict_sentiment_by_demographics = {} #dict to contain pos_df and neg_df for each demographic in scope
    
    dict_demographic_sentiment_pos = {}
    dict_demographic_sentiment_neg = {}
    
    #positive sentiment
    if analysis_scenario == 'known_sentiment_with_demographics' or analysis_scenario == 'known_sentiment_no_demographics':
        pos_filtered_df = df[df[sentiment_col].isin(list_pos_labels)]
        neg_filtered_df = df[df[sentiment_col].isin(list_neg_labels)]
        
    else:
        pos_filtered_df = df[df[sentiment_col] == 'Positive']
        neg_filtered_df = df[df[sentiment_col] == 'Negative']
    
    for demographic_col in list_demographic_fields:
        #get list of unique labels within the given demographic column
        list_unique_demo_labels = list(set(df[demographic_col]))
        
        dict_demo_col_to_label_df = {}
        for label in list_unique_demo_labels:
            #filter df further to the current label in the current demo field
            filtered_df_to_demo_label = pos_filtered_df[pos_filtered_df[demographic_col] == label]
            dict_demo_col_to_label_df[label] = filtered_df_to_demo_label
        
        dict_demographic_sentiment_pos[demographic_col] = dict_demo_col_to_label_df
    
    #negative sentiment
    for demographic_col in list_demographic_fields:
        #get list of unique labels within the given demographic column
        list_unique_demo_labels = list(set(df[demographic_col]))

        dict_demo_col_to_label_df = {}
        for label in list_unique_demo_labels:
            #filter df further to the current label in the current demo field
            filtered_df_to_demo_label = neg_filtered_df[neg_filtered_df[demographic_col] == label]
            dict_demo_col_to_label_df[label] = filtered_df_to_demo_label
                    
        dict_demographic_sentiment_neg[demographic_col] = dict_demo_col_to_label_df
 
    dict_sentiment_by_demographics['Positive'] = dict_demographic_sentiment_pos
    dict_sentiment_by_demographics['Negative'] = dict_demographic_sentiment_neg        
    return dict_sentiment_by_demographics
        
# --------------------------------------------

def get_dict_subset_df_to_unknown_sentiment_with_demographics(
        df, 
        list_demographic_fields
        ):
    
    """
    Function to subset df where sentiment is not known, into dfs for each demographic.
    """

    dict_demo_to_label_dfs = {}
    for demographic_col in list_demographic_fields:
        #get list of unique labels within the given demographic column
        list_unique_demo_labels = list(set(df[demographic_col]))
        
        dict_demo_col_to_label_df = {}
        for label in list_unique_demo_labels:
            #filter df further to the current label in the current demo field
            filtered_df_to_demo_label = df[df[demographic_col] == label]
            dict_demo_col_to_label_df[label] = filtered_df_to_demo_label
        
        dict_demo_to_label_dfs[demographic_col] = dict_demo_col_to_label_df
    
    return dict_demo_to_label_dfs
        
# --------------------------------------------

def remove_stopwords_and_punctuation(text):
    """
    function to remove stop words from a given single string
    """
    #nltk.download('stopwords')
    
    #load stopwords
    stop_words = set(stopwords.words('english'))
    #update stop_words to include common punctuation characters 
    stop_words.update(string.punctuation)
    #tokenize the words
    words = nltk.word_tokenize(text)
    #filter out stop words or punctuation
    filtered_words = [word for word in words if word.lower() not in stop_words]
    #return the joined up string containing non-stopwords or punctuation
    return ' '.join(filtered_words)

# --------------------------------------------

def remove_stopwords_from_df_col(df, col_with_stopwords_to_remove, new_col_name):
    """
    Function to remove stopwords and punctuation from a given col in a df
    and add a new col with a user-provided name (new_col_name)
    """
    df[new_col_name] = df[col_with_stopwords_to_remove].apply(remove_stopwords_and_punctuation)
    return df

# --------------------------------------------

def join_strings_with_common_and_last(words):
    if len(words) == 1:
        return words[0]
    else:
        return ', '.join(words[:-1]) + ' and ' + words[-1]

# --------------------------------------------
def check_what_languages_are_present(df, col_with_text):
    """
    Function to identify the languages present in a given col containing text
    """
    #detect languages for each row in text col. This returns the lang code. 
    # pass langcode to the pycountry to convert this to language name. 
    # Add this as a new col to df.
    df['language'] = [pycountry.languages.get(alpha_2=detect(review)).name for review in list(df[col_with_text])]
    
    # Detect languages for each row in the text column
    df['language_code'] = [detect(review) for review in df[col_with_text]]

    #get number of each language present
    lang_counts = df['language'].value_counts().sort_values(ascending = False)
    
    #use new language column to identify what languages are present
    list_unique_languages_present = lang_counts.index.tolist()

    #get number of langs present
    number_languages_in_data = len(list_unique_languages_present)

    str_languages_present = f"In descending order, the following languages are present in the dataset: {join_strings_with_common_and_last(list_unique_languages_present)}."

    return df, str_languages_present

# --------------------------------------------
def detect_language(review):
    try:
        return detect(review) if len(review) > 3 else 'Unknown'
    except LangDetectException:
        return 'Unknown'

def check_what_languages_are_present_revised(df, col_with_text):
    """
    Function to identify the languages present in a given col containing text
    """
    # Detect languages for each row in the text column
    df['language_code'] = [detect_language(review) for review in df[col_with_text]]

    # Map language codes to language names using pycountry
    df['language'] = df['language_code'].map(lambda x: pycountry.languages.get(alpha_2=x).name if x != 'Unknown' else 'Unknown')

    # Get the count of each language
    lang_counts = df['language'].value_counts().sort_values(ascending=False)

    # Use the new language column to identify what languages are present
    list_unique_languages_present = lang_counts.index.tolist()

    # Get the number of languages present
    number_languages_in_data = len(list_unique_languages_present)

    str_languages_present = f"In descending order, the following languages are present in the dataset: {', '.join(list_unique_languages_present)}."
    #st.write(df['language'].value_counts())
    return df, str_languages_present

# --------------------------------------------

def filter_df_to_label(df, col_to_filter_on, label_to_filter_on):
    """
    Function to subset a df based on a given value in a given col.
    Returns the filtered subset df with rows containing just the given value in that col.
    """
    subset_df = df[df[col_to_filter_on] == label_to_filter_on]
    return subset_df

# --------------------------------------------

def create_subset_dfs_sent_and_demo(
        df, 
        demographic_columns,
        str_df_has_sentiment_truth,
        positive_labels,
        negative_labels,
        sentiment_col,
        analysis_scenario
        ):
    
    #create a dictionary to contain the dataset(s) to analyse in next stage(s) 
    #based on user selections above
    dict_processed_dfs = {}
    #update dict with combined data set
    dict_processed_dfs['all_data'] = df

    if str_df_has_sentiment_truth == 'Yes' and len(demographic_columns) == 0:
        dict_df_selections = get_dict_subset_df_to_known_sentiment(
            df,
            sentiment_col,
            positive_labels, 
            negative_labels)
        
        analysis_method = 'known_sentiment_no_demographics'
        dict_processed_dfs[analysis_method] = dict_df_selections

    elif str_df_has_sentiment_truth == 'Yes' and len(demographic_columns) > 0:
        dict_df_selections = get_dict_subset_df_to_known_sentiment_and_demographics(
            df,
            sentiment_col,
            positive_labels, 
            negative_labels,
            demographic_columns,
            analysis_scenario
            )
        
        analysis_method = 'known_sentiment_with_demographics'
        dict_processed_dfs[analysis_method] = dict_df_selections

    elif str_df_has_sentiment_truth == 'No' and len(demographic_columns) == 0:
        analysis_method = 'unknown_sentiment_no_demographics'

    elif str_df_has_sentiment_truth == 'No' and len(demographic_columns) > 0:
        dict_df_selections = get_dict_subset_df_to_unknown_sentiment_with_demographics(df, demographic_columns)
        analysis_method = 'unknown_sentiment_with_demographics'
        dict_processed_dfs[analysis_method] = dict_df_selections

    return analysis_method, dict_processed_dfs


# --------------------------------------------

# --------------------------------------------

def preview_param_selections(
        str_df_has_sentiment_truth,
        str_df_has_service_label,
        service_col,
        list_services_present,
        positive_labels,
        negative_labels,
        review_column,
        sentiment_col,
        demographic_columns,
        filter_criteria):
    """
    Function to preview the selected parameters for the analysis being run
    """
    
    with st.expander(label=':red[Important!] Check and confirm your selected parameters before pressing confirm!'):
        st.write('The below outlines how the model will utilise the parameters:')
        st.write(f"The text that will be analysed is stored in the :red[{review_column}] column")
        if str_df_has_sentiment_truth == 'Yes':
            st.write(f"The dataset has known sentiment. The sentiment values are stored in the :red[{sentiment_col}] column.")
            if len(positive_labels) > 0:
                st.write(f"Positive words will be identied using the label(s): :red[{join_strings_with_common_and_last(positive_labels)}]")
            if len(negative_labels) > 0:
                st.write(f"Negative words will be identied using the label(s): :red[{join_strings_with_common_and_last(negative_labels)}]")
        if len(demographic_columns) > 0:
            st.write(f"Analysis will also be undertaken on the following demographics: :red[{join_strings_with_common_and_last(demographic_columns)}]")
        if str_df_has_service_label == "Yes":
            st.write(f"The :red[{service_col}] field will be used, and analysis will be undertaken on the following service: :red[{filter_criteria}]")
        else:
            st.write(f"The analysis will be undertaken on all data combined.")
    return None


# --------------------------------------------

def preview_param_selections_no_sentiment(
        str_df_has_sentiment_truth,
        str_df_has_service_label,
        service_col,
        list_services_present,
        review_column,
        sentiment_col,
        demographic_columns,
        filter_criteria):
    """
    Function to preview the selected parameters for the analysis being run
    """
    
    with st.expander(label=':red[Important!] Check and confirm your selected parameters before pressing confirm!'):
        st.write('The below outlines how the model will utilise the parameters:')
        st.write(f"The text that will be analysed is stored in the :red[{review_column}] column")

        st.write(f"The dataset does not have known sentiment. Sentiment values will be estimated from the :red[{review_column}] column.")
    
        if len(demographic_columns) > 0:
            st.write(f"Analysis will also be undertaken on the following demographics: :red[{join_strings_with_common_and_last(demographic_columns)}]")
        if str_df_has_service_label == "Yes":
            st.write(f"The :red[{service_col}] field will be used, and analysis will be undertaken on the following service: :red[{filter_criteria}]")
        else:
            st.write(f"The analysis will be undertaken on all data combined.")
    return None


# --------------------------------------------
# new functions
# --------------------------------------------

def remove_stop_words(text, stop_words):
    """
    Function to remove stop words
    """
    result = []
    for token in simple_preprocess(text):
        if token not in stop_words:# and len(token) > 3: #CHECK THIS NUMBER 3 ?
            result.append(token)
    return result

# --------------------------------------------


def lemmatize(text, nlp):
    """
    lemmatize the text, using spacy approach. can involve higher resource requiremments.
    """
    doc = nlp(" ".join(text))  # Join the list of words into a string
    return [token.lemma_ for token in doc]

# --------------------------------------------


def lemmatize_nltk(text, nlp):
    """
    Function for lemmatization using WordNetLemmatizer. Possibly lower resource requirement, limited for English language only.
    """
    doc = nlp(" ".join(text))  # Join the list of words into a string
    return [token.lemma_ for token in doc]

# --------------------------------------------


def lemmatize_remove_punctuation(text, remove_punctuation):
    lemmatizer = WordNetLemmatizer()
    if remove_punctuation == 'Yes':
        translator = str.maketrans('', '', string.punctuation)  # Removes punctuation
        text_no_punct = text.translate(translator)
        return [lemmatizer.lemmatize(word) for word in text_no_punct.split()]
    else:
        return [lemmatizer.lemmatize(word) for word in text.split()]



# --------------------------------------------
# --------------------------------------------
#<<<< Survey Analysis Page Functions >>>>
# --------------------------------------------
# --------------------------------------------

@st.cache_data(ttl=1800)
def run_lda_topic_modelling_exc_nlp_param(
    vectorizer_method,
    num_n_grams,
    remove_punctuation,
    documents,
    num_topics,
    df,
    survey_responses,
    stop_words,
):
    nlp = spacy.load("en_core_web_sm")

    #call the pre-process function - removes stop words
    processed_docs_no_stop = [remove_stop_words(doc, stop_words) for doc in documents]

    #test to check length of documents (i.e. contains some lemmatised text)
    #st.subheader(":red[Processed Documents After Removing Stop Words:]")
    #for doc in processed_docs_no_stop:
    #    st.write(doc)

    # Apply lemmatization to your text data (prevents scenarios where a word can appear single and plural in the topic)
    processed_docs_lemmatized = [lemmatize(doc, nlp) for doc in processed_docs_no_stop]

    #join the processed strings back together into a list of strings. 
    joined_processed_docs_lemmatized = [' '.join(inner_list) for inner_list in processed_docs_lemmatized]
    
    #test to check length of documents (i.e. contains some lemmatised text)
    #st.subheader(":red[Processed Documents Before Vectorization:]")
    #for doc in joined_processed_docs_lemmatized:
    #    st.write(doc)

    if vectorizer_method == 'tf-idf':
        vectorizer = TfidfVectorizer(max_df=0.85, min_df=1, stop_words='english', ngram_range=(1, num_n_grams), tokenizer=lambda x: lemmatize_remove_punctuation(x, remove_punctuation))
    else:
        vectorizer = CountVectorizer(max_df=0.85, min_df=1, stop_words='english', ngram_range=(1, num_n_grams), tokenizer=lambda x: lemmatize_remove_punctuation(x, remove_punctuation))

    # Fit the vectorizer on processed and lemmatized documents
    X = vectorizer.fit_transform(joined_processed_docs_lemmatized) 

    # Apply Latent Dirichlet Allocation (LDA) on the tf-idf matrix
    #num_topics = 5
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)

    # Get the topic distribution for each response
    topic_distributions = lda.transform(X)
    # Identify the dominant topic for each response
    dominant_topics = topic_distributions.argmax(axis=1)
    # Create a DataFrame linking responses to topics
    df_mapped_response_topic = pd.DataFrame({'Response': df[survey_responses], 'Dominant Topic': dominant_topics+1})
    #populate a dictionary with subset dataframes, with each subset representing a topic, and the df consisting of responses predominately associated with that topic
    dict_topic_to_df = {}
    for topic_num in range(num_topics):
        df_subset = df_mapped_response_topic[df_mapped_response_topic['Dominant Topic'] == topic_num+1]
        dict_topic_to_df[f"Topic {topic_num+1}"] = df_subset

    # Extract topics 
    feature_names = vectorizer.get_feature_names_out()
        
    return feature_names, lda, dict_topic_to_df, joined_processed_docs_lemmatized

# --------------------------------------------

def run_lda_topic_modelling(
    vectorizer_method,
    num_n_grams,
    remove_punctuation,
    documents,
    num_topics,
    df,
    survey_responses,
    stop_words,
    nlp
):
    
    #call the pre-process function - removes stop words
    processed_docs_no_stop = [remove_stop_words(doc, stop_words) for doc in documents]

    #test to check length of documents (i.e. contains some lemmatised text)
    #st.subheader(":red[Processed Documents After Removing Stop Words:]")
    #for doc in processed_docs_no_stop:
    #    st.write(doc)

    # Apply lemmatization to your text data (prevents scenarios where a word can appear single and plural in the topic)
    processed_docs_lemmatized = [lemmatize(doc, nlp) for doc in processed_docs_no_stop]

    #join the processed strings back together into a list of strings. 
    joined_processed_docs_lemmatized = [' '.join(inner_list) for inner_list in processed_docs_lemmatized]
    
    #test to check length of documents (i.e. contains some lemmatised text)
    #st.subheader(":red[Processed Documents Before Vectorization:]")
    #for doc in joined_processed_docs_lemmatized:
    #    st.write(doc)

    if vectorizer_method == 'tf-idf':
        vectorizer = TfidfVectorizer(max_df=0.85, min_df=1, stop_words='english', ngram_range=(1, num_n_grams), tokenizer=lambda x: lemmatize_remove_punctuation(x, remove_punctuation))
    else:
        vectorizer = CountVectorizer(max_df=0.85, min_df=1, stop_words='english', ngram_range=(1, num_n_grams), tokenizer=lambda x: lemmatize_remove_punctuation(x, remove_punctuation))

    # Fit the vectorizer on processed and lemmatized documents
    X = vectorizer.fit_transform(joined_processed_docs_lemmatized) 

    # Apply Latent Dirichlet Allocation (LDA) on the tf-idf matrix
    #num_topics = 5
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)

    # Get the topic distribution for each response
    topic_distributions = lda.transform(X)
    # Identify the dominant topic for each response
    dominant_topics = topic_distributions.argmax(axis=1)
    # Create a DataFrame linking responses to topics
    df_mapped_response_topic = pd.DataFrame({'Response': df[survey_responses], 'Dominant Topic': dominant_topics+1})
    #populate a dictionary with subset dataframes, with each subset representing a topic, and the df consisting of responses predominately associated with that topic
    dict_topic_to_df = {}
    for topic_num in range(num_topics):
        df_subset = df_mapped_response_topic[df_mapped_response_topic['Dominant Topic'] == topic_num+1]
        dict_topic_to_df[f"Topic {topic_num+1}"] = df_subset

    # Extract topics 
    feature_names = vectorizer.get_feature_names_out()
        
    return feature_names, lda, dict_topic_to_df, joined_processed_docs_lemmatized

# --------------------------------------------

def loop_over_dict_cat_vars_run_lda_topic_modelling(
    vectorizer_method,
    num_n_grams,
    remove_punctuation,
    num_topics,
    survey_responses,
    dict_cat_col_to_label_to_subset_df,
    list_cat_variables_to_analyse_by,
    dict_cat_col_unique_labels,
    stop_words,
    nlp):

    #dict to populate with the model and wordcloud for each sub-set df for cat variable / unique label
    dict_cat_col_to_label_to_model_wordcloud = {}

    for cat_var in list_cat_variables_to_analyse_by:
        
        dict_unique_label_to_model_outputs ={}
        for label_index in range(len(dict_cat_col_unique_labels[cat_var])):
            unique_label = dict_cat_col_unique_labels[cat_var][label_index]

            df = dict_cat_col_to_label_to_subset_df[cat_var][unique_label]
            documents = df[survey_responses].dropna().values.tolist()

            #call lda topic modelling function for the current cat var > unique label's df
            feature_names, lda, dict_topic_to_df, joined_processed_docs_lemmatized = run_lda_topic_modelling(
                vectorizer_method,
                num_n_grams,
                remove_punctuation,
                documents,
                num_topics,
                df,
                survey_responses,
                stop_words,
                nlp)

            dict_model_outputs = {}
            dict_model_outputs['feature_names'] = feature_names
            dict_model_outputs['lda'] = lda
            dict_model_outputs['dict_topic_to_df'] = dict_topic_to_df
            dict_model_outputs['joined_processed_docs_lemmatized'] = joined_processed_docs_lemmatized

            dict_unique_label_to_model_outputs[unique_label] = dict_model_outputs
        dict_cat_col_to_label_to_model_wordcloud[cat_var] = dict_unique_label_to_model_outputs

    return dict_cat_col_to_label_to_model_wordcloud

# --------------------------------------------

def run_all_scenarios_odds_ratio(
    list_demographics_selected,
    text_dataset,
    sentiment_col,
    analysis_scenario,
    z_critical = 1.96
    ):
    """
    Function to undertake Odds Ratio for each of a given demographic group, passed to the function.
    With 95% confidence interval (default value in function, can be adjusted if required) 
    """
    dict_results_all_unique_labels = {}
    
    for col in list_demographics_selected:
        list_unique_labels = list(set(text_dataset[col]))

        for unique_label in list_unique_labels:
            
            subset_df_exposed = text_dataset[text_dataset[col] == unique_label]
            subset_df_not_exposed = text_dataset[text_dataset[col] != unique_label]

            if analysis_scenario == 'known_sentiment_with_demographics' or analysis_scenario == 'known_sentiment_no_demographics':
                a = np.sum(subset_df_exposed[f"{sentiment_col}_binary"] == "Positive")
                b = np.sum(subset_df_exposed[f"{sentiment_col}_binary"] == "Negative")
                c = np.sum(subset_df_not_exposed[f"{sentiment_col}_binary"] == "Positive")
                d = np.sum(subset_df_not_exposed[f"{sentiment_col}_binary"] == "Negative")
            
            else:
                a = np.sum(subset_df_exposed[sentiment_col] == "Positive")
                b = np.sum(subset_df_exposed[sentiment_col] == "Negative")
                c = np.sum(subset_df_not_exposed[sentiment_col] == "Positive")
                d = np.sum(subset_df_not_exposed[sentiment_col] == "Negative")

            #dict_unique_labels[col] = list_unique_labels
        
            #list_odds_ratio = []
            #list_log_odds_ratio = []
            #list_standard_error = []
            #list_lower_ci = []
            #list_upper_ci = []
            #list_significant = []

            #a = np.sum(df_exposed[outcome_column] == good_outcome_label[0])
            #b = np.sum(df_exposed[outcome_column] == bad_outcome_label[0])
            #c = np.sum(df_not_exposed[outcome_column] == good_outcome_label[0])
            #d = np.sum(df_not_exposed[outcome_column] == bad_outcome_label[0])

            if a == 0 or b == 0 or c == 0 or d == 0:
                #this could be improved later - e.g. reporting "inf" (as divide by 0) 
                #and then running fischers exact test.
                pass
            else:
                # Calculate the odds ratio
                odds_ratio = (a * d) / (b * c)
                log_odds_ratio = math.log(odds_ratio)

                # Calculate the standard error of the log odds ratio
                standard_error = math.sqrt((1/a) + (1/b) + (1/c) + (1/d))

                # Calculate the lower and upper bounds of the confidence interval
                #z_critical = st.selectbox('Select the confidence level', options=[90, 95, 99], placeholder=95) #1.96  # 95% confidence interval
                lower_bound = math.exp(log_odds_ratio - z_critical * standard_error)
                upper_bound = math.exp(log_odds_ratio + z_critical * standard_error)

                dict_results_thisunique_label = {}
                dict_results_thisunique_label['a'] = a
                dict_results_thisunique_label['b'] = b
                dict_results_thisunique_label['c'] = c
                dict_results_thisunique_label['d'] = d
                dict_results_thisunique_label['odds_ratio'] = odds_ratio
                dict_results_thisunique_label['log_odds_ratio'] = log_odds_ratio
                dict_results_thisunique_label['standard_error'] = standard_error
                dict_results_thisunique_label['lower_bound'] = lower_bound
                dict_results_thisunique_label['upper_bound'] = upper_bound
                

                if lower_bound < 1 and upper_bound < 1:
                    sig_bool = True
                    dict_results_thisunique_label['significant'] = sig_bool
                    dict_results_thisunique_label['interpretation'] = 'lower odds in exposed group'
                elif lower_bound > 1 and upper_bound > 1:
                    sig_bool = True
                    dict_results_thisunique_label['significant'] = sig_bool
                    dict_results_thisunique_label['interpretation'] = 'higher odds in exposed group'

                elif lower_bound < 1 and upper_bound > 1:
                    sig_bool = False
                    dict_results_thisunique_label['significant'] = sig_bool
                    dict_results_thisunique_label['interpretation'] = 'no difference in the odds between exposure groups'

                else:
                    pass
                
                dict_results_all_unique_labels[f"{col}-{unique_label}"] = dict_results_thisunique_label
            
            
    return dict_results_all_unique_labels

# --------------------------------------------

def slider_input(
        label_text,
        min,
        max,
        step_int
):
    input_slider = st.slider(
        label=label_text,
        min_value=min,
        max_value=max,
        step=step_int
    )
    return input_slider

# --------------------------------------------

#TODO - call this function in the scenario where there is no known sentiment

def get_sentiment_analysis_summary(
        review_col_no_stopwords,
        df_english_reviews_pos,
        df_english_reviews_neg
):
    #undertake sentiment analysis on these separate dfs
        df_english_reviews_no_stopwords_pos_with_sentiment = determine_sentiment(df_english_reviews_pos, review_col_no_stopwords, 0.05)
        df_english_reviews_no_stopwords_neg_with_sentiment = determine_sentiment(df_english_reviews_neg, review_col_no_stopwords, 0.05)

        #TODO: Add expanders beneath each summary table, to present each df (positive / neg respectively) containing the records 
        #that the function labelled as negative, or positive, ?or neutral. Function will need adjusting for this to be done. 

        st.write('The below tables indicate how well the sentiment analysis has performed, where a ground truth is known (e.g. FFT)')
        col1, col2 = st.columns(2)
        with col1:
            st.write('Positive sentiment')
            pos_result = df_english_reviews_no_stopwords_pos_with_sentiment.groupby(f'{st.session_state["sentiment_col"]}_binary')['Sentiment'].value_counts().unstack(fill_value=0)
            st.dataframe(pos_result)
        with col2:
            st.write('Negative sentiment')
            neg_result = df_english_reviews_no_stopwords_neg_with_sentiment.groupby(f'{st.session_state["sentiment_col"]}_binary')['Sentiment'].value_counts().unstack(fill_value=0)
            st.dataframe(neg_result)

# --------------------------------------------

def run_sentiment_topic_modelling_overall(
        vectorizer_method,
        num_n_grams,
        remove_punctuation,
        num_topics,
        df,
        survey_responses,
        stop_words
):
    """
    Function to run the overall topic modelling for a combined data set. Currently called in the sentiment analysis section of the app. 
    """
    documents = df[survey_responses].dropna().values.tolist()
    
    nlp = spacy.load("en_core_web_sm")
    
    #call nested function to run lda model
    feature_names, lda, dict_topic_to_df, joined_processed_docs_lemmatized = run_lda_topic_modelling(
        vectorizer_method,
        num_n_grams,
        remove_punctuation,
        documents,
        num_topics,
        df,
        survey_responses,
        stop_words,
        nlp)
    
    return feature_names, lda, dict_topic_to_df, joined_processed_docs_lemmatized
# --------------------------------------------

def run_sentiment_topic_modelling_by_cat_var(
    vectorizer_method,
    num_n_grams,
    remove_punctuation,
    num_topics,
    dict_sentiment_demographic_label_df,
    survey_responses,
    stop_words,
    list_cat_variables,
    analysis_scenario,
    dict_sentiment_cat_var_dfs):
    
    """
    Function to run the overall topic modelling for a combined data set. Currently called in the sentiment analysis section of the app. 
    """

    nlp = spacy.load("en_core_web_sm")

    dict_results_for_cat_variables = {}
    list_sentiment_labels = ['Positive', 'Negative']
    
    for cat_var in list_cat_variables:

        dict_sentiment_results = {}
        for sentiment in list_sentiment_labels:
            
            dict_results_unique_label = {}
            #for unique_label in st.session_state['dict_processed_data'][analysis_scenario][sentiment][cat_var].keys():
            for unique_label in dict_sentiment_cat_var_dfs[analysis_scenario][sentiment][cat_var].keys():
                
                #subset_df = st.session_state['dict_processed_data'][analysis_scenario][sentiment][cat_var][unique_label]
                subset_df = dict_sentiment_cat_var_dfs[analysis_scenario][sentiment][cat_var][unique_label]
                
                #test print
                #st.write(dict_sentiment_cat_var_dfs)
                #try:
                documents = subset_df[survey_responses].dropna().values.tolist()
                
                #test print
                #st.write(f"{cat_var} - {sentiment} - {unique_label} - {len(documents)}")

                #call nested function to run lda model
                #try: 
                feature_names, lda, dict_topic_to_df, joined_processed_docs_lemmatized = run_lda_topic_modelling(
                vectorizer_method,
                num_n_grams,
                remove_punctuation,
                documents,
                num_topics,
                subset_df,
                survey_responses,
                stop_words,
                nlp)

                dict_model_outputs = {}
                dict_model_outputs['feature_names'] = feature_names
                dict_model_outputs['lda'] = lda
                dict_model_outputs['dict_topic_to_df'] = dict_topic_to_df
                dict_model_outputs['joined_processed_docs_lemmatized'] = joined_processed_docs_lemmatized
                
                #except:
                #    dict_model_outputs = {}
                #    dict_model_outputs['feature_names'] = 'not run'
                #    dict_model_outputs['lda'] = 'not run'
                #    dict_model_outputs['dict_topic_to_df'] = 'not run'
                #    dict_model_outputs['joined_processed_docs_lemmatized'] = 'not run'
                #    continue

                #except ValueError:
                #    dict_model_outputs = {}
                #    pass

                dict_results_unique_label[unique_label] = dict_model_outputs
            dict_sentiment_results[sentiment] = dict_results_unique_label
        dict_results_for_cat_variables[cat_var] = dict_sentiment_results

    return dict_results_for_cat_variables

# --------------------------------------------

@st.cache_data(ttl=1800)
def get_and_render_known_sentiment_overview(
        title,
        text_dataset,
        review_col,
        review_col_no_stopwords,
        binary_sentiment_col
):
    
    """
    Function to be called when the processed data has known sentiment. 
    Derives an overview summary and presents this on the page. No demographic breakdown at this stage.
    """

    #------------------------------------------
    #Overview of the data set
    #------------------------------------------
    st.subheader(title)
    st.write("""The below pie chart indicates the range and relative frequency of 
                languages present in the data. The word clouds show the frequency 
                of words in positive and negative English reviews, respectively.""")
    col1, col2, col3 = st.columns(3)

    with col1:
        lang_plot = create_pie_chart(text_dataset, 'language', 'Language breakdown')
        st.pyplot(fig = lang_plot)

        #split df into subset df for English reviws
        df_english_reviews = subset_df_for_specific_langage(text_dataset, 'language', 'English')
    
    df_english_reviews_pos = subset_df_for_specific_langage(df_english_reviews, binary_sentiment_col, 'Positive')
    df_english_reviews_neg = subset_df_for_specific_langage(df_english_reviews, binary_sentiment_col, 'Negative')

    with col2:
        #st.write('Positive English Reviews')
        #create_word_cloud(df_english_reviews_pos, 'Review', 'Positive English Reviews', 'cool')
        create_word_cloud(df_english_reviews_pos, review_col, 'Positive English Reviews', 'cool')
    
    with col3:
        #st.write('Negative English Reviews')
        #create_word_cloud(df_english_reviews_neg, 'Review', 'Negative English Reviews', 'hot')
        create_word_cloud(df_english_reviews_neg, review_col, 'Negative English Reviews', 'hot')
    

    #------------------------------------------
    #Identify outlier comments 
    #TODO adjust function to generate outliers to ensure any comments marked as "unpublishable" are removed from the source data so cannot be identified as an outlier
    #------------------------------------------
    st.subheader('Identify the most extreme outlier comments for each sentiment type')
        
    st.write('Any positive or negative outlier reviews are listed below for consideration. Note, these represent those reviews that diverge from the majority.')
    col1, col2 = st.columns(2)
    with col1:
        pos_anomalies, pos_list_of_anomaly_text = detect_anomalies(df_english_reviews_pos, review_col, contamination_param_threshold = 0.05, num_outliers_to_return=10)
        st.write('**:green[Positive outliers:]**')
        for pos_exception in pos_list_of_anomaly_text:
            st.write(f'"{pos_exception}"')

    with col2:
        neg_anomalies, neg_list_of_anomaly_text = detect_anomalies(df_english_reviews_neg, review_col, contamination_param_threshold = 0.05, num_outliers_to_return=10)
        st.write('**:red[Negative outliers:]**')
        for neg_exception in neg_list_of_anomaly_text:
            st.write(f'"{neg_exception}"')

# --------------------------------------------


#from sklearn.decomposition import LatentDirichletAllocation
#from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#from sklearn.model_selection import GridSearchCV
#from gensim.models import CoherenceModel
#import numpy as np
#import pandas as pd
#import spacy

def perplexity_scorer(model, X):
    # Calculate perplexity (lower is better)
    score = np.exp(-model.score(X))
    return score




#----------------------------------

def run_lda_topic_modelling_return_X(
    vectorizer_method,
    num_n_grams,
    remove_punctuation,
    documents,
    num_topics,
    df,
    survey_responses,
    stop_words,
    nlp
):
    
    #call the pre-process function - removes stop words
    processed_docs_no_stop = [remove_stop_words(doc, stop_words) for doc in documents]

    #test to check length of documents (i.e. contains some lemmatised text)
    #st.subheader(":red[Processed Documents After Removing Stop Words:]")
    #for doc in processed_docs_no_stop:
    #    st.write(doc)

    # Apply lemmatization to your text data (prevents scenarios where a word can appear single and plural in the topic)
    processed_docs_lemmatized = [lemmatize(doc, nlp) for doc in processed_docs_no_stop]

    #join the processed strings back together into a list of strings. 
    joined_processed_docs_lemmatized = [' '.join(inner_list) for inner_list in processed_docs_lemmatized]
    
    #test to check length of documents (i.e. contains some lemmatised text)
    #st.subheader(":red[Processed Documents Before Vectorization:]")
    #for doc in joined_processed_docs_lemmatized:
    #    st.write(doc)

    if vectorizer_method == 'tf-idf':
        vectorizer = TfidfVectorizer(max_df=0.85, min_df=1, stop_words='english', ngram_range=(1, num_n_grams), tokenizer=lambda x: lemmatize_remove_punctuation(x, remove_punctuation))
    else:
        vectorizer = CountVectorizer(max_df=0.85, min_df=1, stop_words='english', ngram_range=(1, num_n_grams), tokenizer=lambda x: lemmatize_remove_punctuation(x, remove_punctuation))

    # Fit the vectorizer on processed and lemmatized documents
    X = vectorizer.fit_transform(joined_processed_docs_lemmatized) 

    # Apply Latent Dirichlet Allocation (LDA) on the tf-idf matrix
    #num_topics = 5
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)

    # Get the topic distribution for each response
    topic_distributions = lda.transform(X)
    # Identify the dominant topic for each response
    dominant_topics = topic_distributions.argmax(axis=1)
    # Create a DataFrame linking responses to topics
    df_mapped_response_topic = pd.DataFrame({'Response': df[survey_responses], 'Dominant Topic': dominant_topics+1})
    #populate a dictionary with subset dataframes, with each subset representing a topic, and the df consisting of responses predominately associated with that topic
    dict_topic_to_df = {}
    for topic_num in range(num_topics):
        df_subset = df_mapped_response_topic[df_mapped_response_topic['Dominant Topic'] == topic_num+1]
        dict_topic_to_df[f"Topic {topic_num+1}"] = df_subset

    # Extract topics 
    feature_names = vectorizer.get_feature_names_out()
        
    return feature_names, lda, dict_topic_to_df, joined_processed_docs_lemmatized, X

#----------------------------------
@st.cache_data(ttl=1800)
def find_optimal_lda_params(
    df,
    survey_responses,
    stop_words,
    num_topics_range,
    vectorizer_methods,
    remove_punctuation_values,
    ngram_ranges
):  
    
    documents = df[survey_responses].dropna().values.tolist()

    nlp = spacy.load("en_core_web_sm")
    
    perplexity_results = []


    best_params = {}
    best_score = float('inf')

    for num_topics in num_topics_range:
        for vectorizer_method in vectorizer_methods:
            for remove_punctuation in remove_punctuation_values:
                for ngram_max in ngram_ranges:
                    #ngram_range = (1, ngram_max)
                    feature_names, lda, _, _, X = run_lda_topic_modelling_return_X(
                        vectorizer_method,
                        ngram_max,  # Assuming ngram_range is a tuple (min_n, max_n)
                        remove_punctuation,
                        documents,
                        num_topics,
                        df,
                        survey_responses,
                        stop_words,
                        nlp  # Assuming nlp is defined somewhere in your code
                    )

                    # Calculate perplexity as a measure of model performance
                    perplexity = lda.perplexity(X)


                    # Append the results to the list
                    perplexity_results.append({
                        'num_topics': num_topics,
                        'vectorizer_method': vectorizer_method,
                        'remove_punctuation': remove_punctuation,
                        'ngram_range': ngram_max,
                        'perplexity': perplexity
                    })

                    # Update best parameters if current perplexity is better
                    if perplexity < best_score:
                        best_score = perplexity
                        best_params = {
                            'num_topics': num_topics,
                            'vectorizer_method': vectorizer_method,
                            'remove_punctuation': remove_punctuation,
                            'ngram_range': ngram_max
                        }

    # Convert the list of results into a DataFrame
    perplexity_df = pd.DataFrame(perplexity_results)

    return best_params, perplexity_df


#-----------------------------------------------------

def translate_text(df, col_with_text, src_lang, target_lang):
  """
  Function to translate a column from a df into a target language and return as list
  """
  # Initialize the translator
  translator = Translator()

  #translate text and store in list via comprehension
  list_translated_reviews = [translator.translate(review, src=src_lang, dest=target_lang) for review in list(df[col_with_text])]

  #update df
  df[f'{col_with_text}_original'] = df[col_with_text]
  df[col_with_text] = list_translated_reviews

  #return translated text
  return list_translated_reviews, df

#-------------------------------------------------------
@st.cache_data(ttl=1800)
def translate_text_revised(df, review_col):
    """
    Function to translate a column from a df into English if the language code is not 'en'.
    """
    # Initialize the translator
    translator = Translator()
    
    # Translate text and store in list via comprehension
    list_translated_reviews = []

    for review, lang_code in zip(df[review_col], df['language_code']):
        #st.write(f"{lang_code} - {review}")
        try:
            if lang_code != 'en':
                translate_the_text = translator.translate(review, src=lang_code, dest='en')
                translated_text = translate_the_text.text
                #st.write(translated_text)
                list_translated_reviews.append(translated_text)
            else:
                # If the language is already English, keep the original text
                list_translated_reviews.append(review)
        except Exception as e:
            # Handle any exceptions during translation
            st.warning(f"Error translating: {str(e)}")
            list_translated_reviews.append(review)

    # Rename the original column to {review_col}_original
    df.rename(columns={review_col: f'{review_col}_original'}, inplace=True)

    # Add a new column with translated text
    df[f'{review_col}'] = list_translated_reviews

    return df

#-------------------------------------------------------
@st.cache_data(ttl=1800)
def translate_text_revised_argo(df, review_col):
    """
    Function to translate a column from a df into English if the language code is not 'en'.
    """
    # Load the installed packages
    installed_packages = package.get_installed_packages()

    # Get the package for English
    english_package = next((p for p in installed_packages if p.code == 'en'), None)

    if not english_package:
        st.warning("English package not found. Make sure it's installed.")
        return df

    # Translate text and store in list via comprehension
    list_translated_reviews = []

    #for review, lang_code in zip(df[review_col], df['language_code']):
    for review, lang_code in zip(df[review_col], df['language']):
        try:
            if lang_code != 'en':
                # Get the package for the source language
                source_lang_package = next((p for p in installed_packages if p.code == lang_code), None)

                if source_lang_package:
                    # Translate the text
                    translation = translate.translate(source_lang_package, english_package, review)
                    translated_text = translation.text
                    list_translated_reviews.append(translated_text)
                else:
                    st.warning(f"Package for language {lang_code} not found. Make sure it's installed.")
                    list_translated_reviews.append(review)
            else:
                # If the language is already English, keep the original text
                list_translated_reviews.append(review)
        except Exception as e:
            # Handle any exceptions during translation
            st.warning(f"Error translating: {str(e)}")
            list_translated_reviews.append(review)

    # Rename the original column to {review_col}_original
    df.rename(columns={review_col: f'{review_col}_original'}, inplace=True)

    # Add a new column with translated text
    df[f'{review_col}'] = list_translated_reviews

    return df



#-------------------------------------------------------
#def get_argos_model(source, target):
#    lang = f'{source} -> {target}'
#    source_lang = [model for model in translate.get_installed_languages() if lang in map(repr, model.translations_from)]
#    target_lang = [model for model in translate.get_installed_languages() if lang in map(repr, model.translations_to)]
    
#    return source_lang[0].get_translation(target_lang[0])



def get_argos_model(source, target):
    source_lang = [model for model in translate.get_installed_languages() if source in model.code]
    target_lang = [model for model in translate.get_installed_languages() if target in model.code]

    if source_lang and target_lang:
        return source_lang[0].get_translation(target_lang[0])
    else:
        return None  # Return None if no model is found


def create_translation_models_dict(df_series):
    """
    Function to create a dictionary of translation models based on language pairs in the DataFrame series.
    The keys are in the format f"{source_lang}-English", and values are the corresponding translation models.
    """
    translation_models_dict = {}
    
    # Get unique language pairs in the DataFrame series
    language_pairs = df_series.unique()

    for lang_pair in language_pairs:
        if lang_pair != 'en':  # Exclude English from language pairs
            source_lang, target_lang = lang_pair, 'en'
            model_key = f"{source_lang}-{target_lang}"
            translation_model = get_argos_model(source_lang, target_lang)
            translation_models_dict[model_key] = translation_model

    return translation_models_dict


#---------------------------------------------------
def translate_text(df, lang_code_col, review_col, language_col):
    list_translations = []

    translated_rows = []
    non_translated_languages = []

    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    
    to_code = "en"
    with st.spinner(text='Attempting to translate non-English text... please wait.'):
        for index, row in df.iterrows():
            from_code = row[lang_code_col]
            if from_code == 'en':
                list_translations.append(row[review_col])
            else:
                try:
                    
                    package_to_install = next(filter(lambda x: x.from_code == from_code and x.to_code == to_code, available_packages))
                    argostranslate.package.install_from_path(package_to_install.download())
                    # Translate
                    translatedText = argostranslate.translate.translate(row[review_col], from_code, to_code)
                    list_translations.append(translatedText)
                    translated_rows.append(translatedText)
                except:
                    list_translations.append(None)
                    non_translated_languages.append(row[language_col])
    
    df = df.rename(columns={review_col: f"{review_col}_original"})

    df[f'{review_col}'] = list_translations
    filtered_df = df[df[review_col].notna()]

    # Create a summary string
    num_translated = len(translated_rows)
    num_non_translated = len(non_translated_languages)
    languages_not_translated = list(set(non_translated_languages))

    summary_string = f"Translated {num_translated} rows. Could not translate {num_non_translated} rows in the following languages: {', '.join(languages_not_translated)}."

    return summary_string, filtered_df, df

#--------------------------------------------------
#FUNCTIONS FOR VARIATION BY DEMOGRAPHIC GROUP SECTION
#--------------------------------------------------
"""
def topic_modelling_by_demographic(text_dataset):
    if st.session_state['analysis_scenario'] == 'known_sentiment_with_demographics' or st.session_state['analysis_scenario'] == 'known_sentiment_no_demographics':
        sent_col = f"{st.session_state['sentiment_col']}_binary"
    else:
        sent_col = 'Sentiment'

    list_var_label_combinations = []
    for cat_var in st.session_state['demographic_columns']:
        for unique_label in list(set(st.session_state['dict_processed_data']['text_dataset_inc_na_rows'][cat_var])):
            list_var_label_combinations.append(f"{cat_var}-{unique_label}")

    for var_label_combination in list_var_label_combinations:
        try:
            with st.expander(label=var_label_combination):
                tab1, tab2 = st.tabs(['Positive', 'Negative'])
                
                cat_var = var_label_combination.split('-')[0]
                unique_label = var_label_combination.split('-')[1]
                
                with tab1: #positive #TODO replicate the change in this positive section within the tab 2 negative section
                    subset_df_current_unique_label = text_dataset[text_dataset[cat_var] == unique_label]
                    subset_df_positive = subset_df_current_unique_label[subset_df_current_unique_label[sent_col] == "Positive"]
                    #st.write(subset_df_positive.shape)
                    #st.dataframe(subset_df_positive[review_col])
                    #st.write(subset_df_positive.shape[0] > 0)
                    if subset_df_positive.shape[0] > 0:
                        #get outlier comments for this group
                        pos_anomalies, pos_list_of_anomaly_text = func.detect_anomalies(subset_df_positive, review_col, contamination_param_threshold = 0.05)
                        #st.write(pos_list_of_anomaly_text)
                        if len(pos_list_of_anomaly_text) > 0:
                            st.subheader('Positive outliers:')
                            for pos_exception in pos_list_of_anomaly_text:
                                st.write(f'"{pos_exception}"')

                        #render word cloud - positive
                        st.subheader('Word Cloud of Positive Feedback')
                        st.pyplot(func.categorical_create_word_cloud(subset_df_positive, review_col, f'Wordcloud: {review_col} - {cat_var} - {unique_label}', 'cool'))
                    else:
                        st.write('Insufficient data to run this analysis.')
                        pass

                    #render topc modelling outputs for each demographic variable in scope
                    try:
                        st.subheader('Topic modelling outputs')
                        #st.write(f"unique label: {unique_label}")
                        #st.write(f"cat_var: {cat_var}")
                        #st.write(_dict_results_for_cat_variables['Positive'][cat_var].keys())
                        dict_model_results = _dict_results_for_cat_variables['Positive'][cat_var][unique_label]
                        #st.write(dict_model_results)
                        #for topic_idx, topic in enumerate(dict_model_results['lda'].components_):
                        
                        if dict_model_results == 'Not run':
                            st.write('Topic modelling was not run.')
                        else:
                            for topic_idx, topic in enumerate(dict_model_results['lda_model'].components_):
                                top_words_idx = topic.argsort()[:-10 - 1:-1]
                                top_words = [dict_model_results['feature_names'][i] for i in top_words_idx]
                                st.write(f"**Topic {topic_idx + 1}:** {', '.join(top_words)}")
                            
                            #render df of responses
                            for topic_num in range(num_topics):
                                st.subheader(f"Topic {topic_num+1}")
                                st.dataframe(_dict_results_for_cat_variables['Positive'][cat_var][unique_label]['topic_dfs'][f"Topic {topic_num+1}"])
                    
        except KeyError:
            st.write('Insufficient data to run this analysis.')
            pass

"""
@st.cache_data(ttl=1800)
def sentiment_analysis_page_demographic_group_variation_by_service(
    _dict_results_for_cat_variables,
    text_dataset,
    review_col,
    list_demographics_selected,
    num_topics
    ):
        #first create a df for all processed data (which should ALWAYS be present)
        #text_dataset = st.session_state['dict_processed_data']['all_data']
        
        #get the column names to containing the text we want to analyse
        #review_col = st.session_state['review_column']
        review_col_no_stopwords = f"{review_col}_no_stopwords"
        #------------------------------------------
        #Odds ratio for experience, by each demographic group
        #------------------------------------------
        st.title(':green[**Variation by demographic group**]')
        st.write('This sections seeks to derive insight as to variation in reported experience by the chosen demographic groups in scope')
        st.subheader('***Odds Ratio***')
        st.write("""An Odds Ratio calculation has been used below to determine 
                 whether there is any variation in the liklihood of reporting a 
                 positive experience by each demographic group in scope. 
                 Effectively, using ethnicity as an example, this shows the likelihood 
                 of reporting a positive experience by, say, patients with a White ethnicity, 
                 compared to all other ethnicities, and repeats this for all unique labels of 
                 each categorical / demographic variables selected on the prep page.""")
        

        #extract demographics selected - hardcoded for now via index 0
        #list_demographics_selected = st.session_state['demographic_columns']

        #Repurpose the OR code from the waiting list equity app here
        dict_results_all_unique_labels = func.run_all_scenarios_odds_ratio(
            list_demographics_selected,
            text_dataset,
            st.session_state['sentiment_col'],
            st.session_state['analysis_scenario'],
            z_critical = 1.96
            )

        df_or_demographic_results = pd.DataFrame(dict_results_all_unique_labels).T
        
        #st.dataframe(df_or_demographic_results)

        if 'significant' in df_or_demographic_results.columns:
            if True in list(set(df_or_demographic_results['significant'])):
                df_subset_sig = df_or_demographic_results[df_or_demographic_results['significant'] == True]
                st.write(':green[**Significant**] Odds Ratio findings are summarised below. To view *all* results, click on the expander below.')
                st.dataframe(df_subset_sig)
            else:
                st.write('All Odds Ratios returned :red[**non-significant**] results. The results can be viewed in the expander below.')
            
            with st.expander(label='Click for all OR ratio results'):
                st.dataframe(df_or_demographic_results)
        else:
            st.write('Insufficient data, Odds Ratios not run.')
# --------------------------------------------
#Section to provide word clouds for positive and negative responses (ground truth) by demographic group
# --------------------------------------------
        
        st.subheader('***Demographic summaries***')
        st.write('The below expanders contain findings for each selected individual demographic. Where outlier comments for this group have been detected, these are included in addition to a Word cloud. Both are provided for positive and negative feedback, respectively. ')

        if st.session_state['analysis_scenario'] == 'known_sentiment_with_demographics' or st.session_state['analysis_scenario'] == 'known_sentiment_no_demographics':
            sent_col = f"{st.session_state['sentiment_col']}_binary"
        else:
            sent_col = 'Sentiment'

        list_var_label_combinations = []
        for cat_var in list_demographics_selected:
            for unique_label in list(set(text_dataset[cat_var])):
                list_var_label_combinations.append(f"{cat_var}-{unique_label}")
        
        #st.write(list_var_label_combinations)

        for var_label_combination in list_var_label_combinations:
            try:
                with st.expander(label=var_label_combination):
                    tab1, tab2 = st.tabs(['Positive', 'Negative'])
                    
                    cat_var = var_label_combination.split('-')[0]
                    unique_label = var_label_combination.split('-')[1]
                    
                    with tab1: #positive #TODO replicate the change in this positive section within the tab 2 negative section
                        subset_df_current_unique_label = text_dataset[text_dataset[cat_var] == unique_label]
                        subset_df_positive = subset_df_current_unique_label[subset_df_current_unique_label[sent_col] == "Positive"]
                        #st.write(subset_df_positive.shape)
                        #st.dataframe(subset_df_positive[review_col])
                        #st.write(subset_df_positive.shape[0] > 0)
                        if subset_df_positive.shape[0] > 0:
                            #get outlier comments for this group
                            pos_anomalies, pos_list_of_anomaly_text = func.detect_anomalies(subset_df_positive, review_col, contamination_param_threshold = 0.05)
                            #st.write(pos_list_of_anomaly_text)
                            if len(pos_list_of_anomaly_text) > 0:
                                st.subheader('Positive outliers:')
                                for pos_exception in pos_list_of_anomaly_text:
                                    st.write(f'"{pos_exception}"')

                            #render word cloud - positive
                            st.subheader('Word Cloud of Positive Feedback')
                            st.pyplot(func.categorical_create_word_cloud(subset_df_positive, review_col, f'Wordcloud: {review_col} - {cat_var} - {unique_label}', 'cool'))
                        else:
                            st.write('Insufficient data to run this analysis.')
                            pass

                        #render topc modelling outputs for each demographic variable in scope
                        try:
                            st.subheader('Topic modelling outputs')
                            #st.write(f"unique label: {unique_label}")
                            #st.write(f"cat_var: {cat_var}")
                            #st.write(_dict_results_for_cat_variables['Positive'][cat_var].keys())
                            dict_model_results = _dict_results_for_cat_variables['Positive'][cat_var][unique_label]
                            #st.write(dict_model_results)
                            #for topic_idx, topic in enumerate(dict_model_results['lda'].components_):
                            
                            if dict_model_results == 'Not run':
                                st.write('Topic modelling was not run.')
                            else:
                                for topic_idx, topic in enumerate(dict_model_results['lda_model'].components_):
                                    top_words_idx = topic.argsort()[:-10 - 1:-1]
                                    top_words = [dict_model_results['feature_names'][i] for i in top_words_idx]
                                    st.write(f"**Topic {topic_idx + 1}:** {', '.join(top_words)}")
                                
                                #render df of responses
                                for topic_num in range(num_topics):
                                    st.subheader(f"Topic {topic_num+1}")
                                    st.dataframe(_dict_results_for_cat_variables['Positive'][cat_var][unique_label]['topic_dfs'][f"Topic {topic_num+1}"])
                        
                        except KeyError:
                            st.write('Insufficient data to run this analysis.')
                            pass

                    with tab2: #negative
                        subset_df_negative = subset_df_current_unique_label[subset_df_current_unique_label[sent_col] == "Negative"]
                        
                        #--
                        #st.write(f"got here for: {var_label_combination} - {subset_df_negative.shape}")
                        #st.dataframe(subset_df_negative)
                        #--
                        if subset_df_negative.shape[0] > 0:
                            #get outlier comments for this group
                            neg_anomalies, neg_list_of_anomaly_text = func.detect_anomalies(subset_df_negative, f"{review_col}", contamination_param_threshold = 0.05)
                            if len(neg_list_of_anomaly_text) > 0:
                                st.subheader('Negative outliers:')
                                for neg_exception in neg_list_of_anomaly_text:
                                    st.write(f'"{neg_exception}"')
                            
                            #render word cloud - negative
                            st.subheader('Word Cloud of Negative Feedback')
                            st.pyplot(func.categorical_create_word_cloud(subset_df_negative, review_col, f'Wordcloud: {review_col} - {cat_var} - {unique_label}', 'hot'))
                        
                        else:
                            st.write('Insufficient data to run this analysis.')
                            pass
                        
                        #render topc modelling outputs for each demographic variable in scope
                        try:
                            st.subheader('Topic modelling outputs')
                            dict_model_results = _dict_results_for_cat_variables['Negative'][cat_var][unique_label]
                            
                            if dict_model_results == 'Not run':
                                st.write('Topic modelling was not run.')
                            else:
                                for topic_idx, topic in enumerate(dict_model_results['lda_model'].components_):
                                    top_words_idx = topic.argsort()[:-10 - 1:-1]
                                    top_words = [dict_model_results['feature_names'][i] for i in top_words_idx]
                                    st.write(f"**Topic {topic_idx + 1}:** {', '.join(top_words)}")
                        
                            #render df of responses
                                for topic_num in range(num_topics):
                                    st.subheader(f"Topic {topic_num+1}")
                                    st.dataframe(_dict_results_for_cat_variables['Negative'][cat_var][unique_label]['topic_dfs'][f"Topic {topic_num+1}"])
                            
                            
                        except KeyError:
                            st.write('Insufficient data to run this analysis.')
                            pass
            except:
                continue

#--------------------------------------------------
#--------------------------------------------------
#--------------------------------------------------
#--------------------------------------------------
#--------------------------------------------------
#--------------------------------------------------
#--------------------------------------------------
#--------------------------------------------------
#--------------------------------------------------