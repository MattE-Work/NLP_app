import streamlit as st
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import spacy
from spacy import displacy
from spacy.lang.en.stop_words import STOP_WORDS

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

import altair as alt

#clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#manual cosine similarity calculation
from sklearn.metrics.pairwise import cosine_similarity

#PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Sentiment Analysis
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

#nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_web_md")

#topic modelling
#text classification
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

#topic model finding optimal params
from sklearn.model_selection import train_test_split

#correlation analysis
from scipy.stats import spearmanr

#word cloud production
from wordcloud import WordCloud

#---------------------------------------

@st.cache_data(ttl=3600)
def remove_line_breaks(text_with_line_breaks):
    """
    Function to take in the text as an argument, and remove line breaks from this text.
    Taken from: https://youtu.be/ytAyCO-n8tY?si=CO7MDBEb0sm5z2I5&t=369
    """
    text_without_line_breaks = text_with_line_breaks.replace("\n\n", " ").replace("\n", " ")
    return text_without_line_breaks

#---------------------------------------
'''
@st.cache_data(ttl=3600)
def preprocess_reviews_whole_df(df):
    """
    Function to lemmatize, remove stop words, and punctuation for all text columns in a DataFrame.
    """
    # Initialize spaCy's English language model
    #nlp = spacy.load("en_core_web_sm")
    nlp = spacy.load("en_core_web_md")
    
    def clean_text_spacy(text):
        """Cleans a single text string using spaCy's lemmatization."""
        # Remove trailing whitespace characters
        text = text.strip()

        doc = nlp(text)

        # Remove stop words and lemmatize
        stop_words = spacy.lang.en.stop_words.STOP_WORDS
        clean_tokens = [token.lemma_.lower() for token in doc if token.lemma_.lower() not in stop_words]
        
        # Remove punctuation from clean tokens
        clean_tokens = [token for token in clean_tokens if token not in string.punctuation]

        return " ".join(clean_tokens)

    # Apply the cleaning function to all text columns
    #cleaned_df = df.copy()  # Create a copy of the original DataFrame
    dict_col_to_cleaned_series = {}
    for column in df.select_dtypes(include='object'):  # Iterate over text columns
        dict_col_to_cleaned_series[f"{column}_cleaned"] = df[column].apply(clean_text_spacy)
    cleaned_df = pd.DataFrame.from_dict(dict_col_to_cleaned_series)
    return cleaned_df
'''
#---------------------------------------

def clean_text_spacy_user_defined(text, remove_punctuation=False, lemmatize=False, remove_stopwords=False):
    """Cleans a single text string based on specified cleaning options."""
    doc = nlp(text.strip())
    cleaned_tokens = []

    for token in doc:
        if remove_stopwords and token.is_stop:
            continue
        if remove_punctuation and token.is_punct:
            continue
        if lemmatize:
            cleaned_tokens.append(token.lemma_.lower())
        else:
            cleaned_tokens.append(token.text.lower())

    return " ".join(cleaned_tokens)

#---------------------------------------

@st.cache_data(ttl=3600)
def preprocess_reviews_whole_df(df, remove_punctuation=True, lemmatize=True, remove_stopwords=True):
    """
    Function to preprocess all text columns in a DataFrame based on specified cleaning options.
    """
    dict_col_to_cleaned_series = {}
    for column in df.select_dtypes(include='object'):  # Iterate over text columns
        # Apply the cleaning function with specified options
        cleaned_series = df[column].apply(lambda text: clean_text_spacy_user_defined(text, 
                                                                         remove_punctuation=remove_punctuation, 
                                                                         lemmatize=lemmatize, 
                                                                         remove_stopwords=remove_stopwords))
        dict_col_to_cleaned_series[f"{column}_cleaned"] = cleaned_series

    cleaned_df = pd.DataFrame.from_dict(dict_col_to_cleaned_series)
    return cleaned_df


#---------------------------------------

@st.cache_data(ttl=3600)
def compare_each_question_to_each_strat_objective(strategy_aspects, staff_reviews, _nlp, threshold):
    """
    Calculate similarity scores between staff reviews and strategy aspects using manual cosine similarity calculation.
    """
    similarity_scores_dict = {}
    threshold_scores_dict = {}

    # Pre-compute aspect vectors outside the loop
    aspect_vectors = {aspect_name: nlp(aspect_text).vector for aspect_name, aspect_text in strategy_aspects.items()}

    for question in staff_reviews.columns:
        question_similarity_df = pd.DataFrame(index=staff_reviews.index)
        question_similarity_df[question] = staff_reviews[question]
        question_threshold_df = question_similarity_df.copy(deep=True)

        for aspect_name in strategy_aspects.keys():
            similarity_scores = []
            aspect_vector = aspect_vectors[aspect_name]  # Pre-computed vector
            aspect_vector = aspect_vector.reshape(1, -1)  # Reshape for cosine_similarity function
            
            for review_text in staff_reviews[question]:
                doc_review = nlp(review_text)
                review_vector = doc_review.vector.reshape(1, -1)  # Reshape for cosine_similarity function
                # Compute cosine similarity
                similarity_score = cosine_similarity(review_vector, aspect_vector)[0][0]
                similarity_scores.append(similarity_score)

            question_similarity_df[f"{aspect_name}-similarity"] = similarity_scores
            threshold_scores = [score >= threshold for score in similarity_scores]
            question_threshold_df[f"{aspect_name}-threshold"] = threshold_scores

        similarity_scores_dict[question] = question_similarity_df
        threshold_scores_dict[question] = question_threshold_df

    return similarity_scores_dict, threshold_scores_dict

#---------------------------------------
'''
@st.cache_data(ttl=3600)
def compare_each_question_to_each_strat_objective(strategy_aspects, staff_reviews, _nlp, threshold):
    """
    Calculate similarity scores between staff reviews and strategy aspects.

    Args:
    - strategy_aspects: A dictionary where keys are strategy aspect names and values are cleaned concatenated text for the strategy objectives for that aspect.
    - staff_reviews: DataFrame containing cleaned staff reviews.

    Returns:
    - similarity_scores_dict: A dictionary where keys are staff survey questions (column headers) and values are DataFrames containing similarity scores for each response to each strategy aspect.
    """
    similarity_scores_dict = {}
    threshold_scores_dict = {}

    # Loop over each question in the DataFrame
    for question in staff_reviews.columns:
        # Initialize DataFrame to store similarity scores for this question
        question_similarity_df = pd.DataFrame(index=staff_reviews.index)
        # Add original review text to DataFrame
        question_similarity_df[question] = staff_reviews[question]
        
        question_threshold_df = question_similarity_df.copy(deep=True)

        # Loop over each strategy aspect
        for aspect_name, aspect_text in strategy_aspects.items():
            # Calculate similarity scores between each staff review and the strategy aspect
            similarity_scores = []
            for review_text in staff_reviews[question]:
                doc_review = nlp(review_text)
                doc_aspect = nlp(aspect_text)
                similarity_score = doc_review.similarity(doc_aspect)
                similarity_scores.append(similarity_score)

            # Add similarity scores for this aspect to the DataFrame
            question_similarity_df[f"{aspect_name}_similarity"] = similarity_scores
            # Check if similarity score is greater than or equal to threshold
            threshold_scores = [score >= threshold for score in similarity_scores]
            question_threshold_df[f"{aspect_name}_threshold"] = threshold_scores

        # Add DataFrame for this question to the dictionary
        similarity_scores_dict[question] = question_similarity_df
        threshold_scores_dict[question] = question_threshold_df

    return similarity_scores_dict, threshold_scores_dict
'''

#---------------------------------------

@st.cache_data(ttl=3600)
def calculate_aggregate_scores(similarity_scores_dict, threshold=None):
    """
    Calculate aggregate scores for similarity matches.

    Args:
    - similarity_scores_dict: A dictionary where keys are staff survey questions (column headers) and values are DataFrames containing similarity scores for each response to each strategy aspect.
    - threshold: Float between 0.1 and 0.99 to determine the threshold for similarity scores. If provided, the proportion of positive matches will be calculated based on this threshold.

    Returns:
    - sum_scores_dict: A dictionary where keys are staff survey questions and values are dictionaries containing the sum of similarity scores for each strategy aspect.
    - proportion_matches_dict: A dictionary where keys are staff survey questions and values are dictionaries containing the proportion of positive matches for each strategy aspect.
    """
    sum_scores_dict = {}
    proportion_matches_dict = {}

    # Loop over each question in the similarity_scores_dict
    for question, similarity_df in similarity_scores_dict.items():
        # Filtering out rows where the response might be empty or NaN before processing
        non_empty_similarity_df = similarity_df.dropna(subset=[question])

        sum_scores = {}
        proportion_matches = {}

        # Calculate sum of scores for each strategy aspect
        sum_scores = similarity_df.drop(columns=[question]).sum().to_dict()

        # Calculate proportion of positive matches for each strategy aspect if threshold is provided
        if threshold is not None:
            threshold_matches_df = similarity_df.drop(columns=[question]).apply(lambda x: x >= threshold)
            proportion_matches = threshold_matches_df.mean().to_dict()

        sum_scores_dict[question] = sum_scores
        proportion_matches_dict[question] = proportion_matches

    return sum_scores_dict, proportion_matches_dict

#---------------------------------------
'''
@st.cache_data(ttl=3600)
def categorize_alignment_based_on_distribution(similarity_df):
    """
    Categorize alignment into 'High', 'Moderate', or 'Low' based on the distribution of similarity scores.

    Args:
    - similarity_df: DataFrame with the first column containing survey responses and subsequent columns containing similarity scores for each strategy aspect.

    Returns:
    - DataFrame with additional columns for each strategy aspect indicating the alignment category.
    """
    # Create a copy of the DataFrame to avoid modifying the original data
    updated_df = similarity_df.copy()

    # Iterate through each strategy aspect column to calculate percentiles and categorize alignment
    for aspect in similarity_df.columns[1:]:  # Skip the first column with survey responses
        high_threshold = similarity_df[aspect].quantile(0.75)
        low_threshold = similarity_df[aspect].quantile(0.25)
        
        # Function to categorize each score
        def categorize(score):
            if score > high_threshold:
                return 'High Alignment'
            elif score > low_threshold:
                return 'Moderate Alignment'
            else:
                return 'Low Alignment'
        
        # Apply categorization function to each score for the current aspect
        alignment_category = similarity_df[aspect].apply(categorize)
        
        # Add the categorization as a new column
        updated_df[f"{aspect}-Alignment"] = alignment_category
    
    return updated_df
'''
#---------------------------------------

@st.cache_data(ttl=3600)
def categorize_alignment_based_on_distribution(similarity_df, text_column_label):
    """
    Categorizes alignment into 'High', 'Moderate', or 'Low' based on the distribution of similarity scores.
    
    Args:
    - similarity_df: DataFrame with one column containing survey responses and subsequent columns containing similarity scores for each strategy aspect.
    - text_column_label: The label of the column containing the source review text (ensure this matches exactly).
    
    Returns:
    - A DataFrame with the source text and additional columns for each strategy aspect indicating the alignment category.
    """

    # Create a copy of the DataFrame to avoid modifying the original data   
    updated_df = similarity_df.copy()

    # Iterate through each strategy aspect column to calculate percentiles and categorize alignment
    for aspect in similarity_df.columns[1:]:  # Skip the first column with survey responses
        high_threshold = similarity_df[aspect].quantile(0.75)
        low_threshold = similarity_df[aspect].quantile(0.25)

        # Apply categorization function to each score for the current aspect using lambda to pass extra arguments
        alignment_category = similarity_df[aspect].apply(lambda score: categorize(score, low_threshold, high_threshold))
        
        # Add the categorization as a new column
        new_col_name = aspect.split('-')[0]
        updated_df[f"{new_col_name}-Alignment"] = alignment_category

    # Select only the specified text column and the "-Alignment" columns for the final DataFrame
    alignment_columns = [col for col in updated_df.columns if "-Alignment" in col]
    # Ensure to adjust this line if the text column label should not be modified
    final_df = updated_df[[f"{text_column_label}_cleaned"] + alignment_columns]  # Removed "_cleaned" suffix adjustment for generality

    return final_df

#---------------------------------------

# Function to categorize each score
def categorize(score, low_threshold, high_threshold):
    """
    function to classify the similarity scores to one of 3 labels, based on their distribution.
    Called in the categorize_alignment_based_on_distribution function
    """
    if score > high_threshold:
        return 'High Alignment'
    elif score > low_threshold:
        return 'Moderate Alignment'
    else:
        return 'Low Alignment'

#---------------------------------------

def plot_alignment_distribution(df, title="Alignment Distribution"):
    """
    Plots an Altair stacked bar chart showing the distribution of alignment categories 
    across each strategy aspect in the DataFrame.

    Args:
    - df: DataFrame with the source text and additional columns for each strategy aspect 
          indicating the alignment category, as output by the categorize_alignment_based_on_distribution function.
    - title: A string that will be used as the chart title.
    """
    # Melt the DataFrame to long format for easier plotting with Altair
    df_long = df.melt(id_vars=[df.columns[0]], var_name='Strategy Aspect', value_name='Alignment')

    # Simplified chart for debugging
    chart = alt.Chart(df_long).mark_bar().encode(
        x='Strategy Aspect:N',
        y='count():Q',
        color='Alignment:N'
    ).properties(
        title=title,  # Set the chart title
    )
    st.altair_chart(chart, use_container_width=True)

#---------------------------------------

def plot_similarity_metrics(similarity_metrics_dict, metric_name):
    """
    Plot a bar chart to visualize similarity metrics.

    Args:
    - similarity_metrics_dict: A dictionary where keys are staff survey questions and values are dictionaries containing the similarity metrics for each strategy aspect.
    - metric_name: Name of the similarity metric (e.g., "Sum of Scores" or "Proportion of Positive Matches").

    Returns:
    - None
    """
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(similarity_metrics_dict).stack().reset_index()
    df.columns = ['Strategy Aspect', 'Question', metric_name]

    # Plot the bar chart with horizontal bars
    #fig, ax = plt.subplots(figsize=(10, 6))
    fig, ax = plt.subplots(figsize=(8, 8)) #adjusted to match shape of radar plot
    sns.barplot(data=df, x=metric_name, y='Strategy Aspect', orient='h', hue='Question', ax=ax)

    # Add labels and title
    ax.set_xlabel(metric_name)
    ax.set_ylabel("Strategy Aspect")
    ax.set_title(f"{metric_name} by Strategy Aspect")

    # Show the plot in Streamlit
    st.pyplot(fig)

#---------------------------------------

def plot_radar_chart(proportion_matches_dict, survey_responses_col_name):
    """
    Plot a radar chart to visualize proportions above a threshold.

    Args:
    - proportion_matches_dict (dict): A dictionary where the outer key is the column name containing the staff reviews,
                                      and the inner keys are labels/categories with values as proportions above the threshold.
    - survey_responses_col_name (str): The column name containing the staff reviews.

    Returns:
    - fig (matplotlib.figure.Figure): The matplotlib figure object.
    """
    # Extract inner dictionary and labels
    proportion_matches_inner_dict = proportion_matches_dict[f"{survey_responses_col_name}_cleaned"]
    labels = list(proportion_matches_inner_dict.keys())
    
    # Extract proportions
    proportions = list(proportion_matches_inner_dict.values())

    # Number of categories
    num_categories = len(labels)

    # Create angles for radar chart
    angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()

    # The plot is circular, so we need to "complete the loop" and append the start
    proportions += proportions[:1]
    angles += angles[:1]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, proportions, color='skyblue', alpha=0.4)
    ax.plot(angles, proportions, color='blue', linewidth=2)
    ax.set_yticklabels([])  # Hide radial labels
    ax.set_theta_offset(np.pi / 2)  # Start the plot at the top
    ax.set_theta_direction(-1)  # Reverse the direction of theta
    ax.set_xticks(angles[:-1])  # Set the ticks to be at the positions of each label
    ax.set_xticklabels(labels)  # Set the labels
    ax.set_title(f'Proportions Above Threshold for {survey_responses_col_name}', size=20, pad=20)

    # Add scale text annotations
    radial_ticks = np.linspace(0.2, 1.0, 5)
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([f'{tick:.1f}' for tick in radial_ticks], fontsize=10, color='gray')

    # Add scale text annotations
    for angle, proportion in zip(angles, proportions[:-1]):
        ax.text(angle, proportion, f'{proportion:.2f}', ha='center', va='bottom')

    # Show the plot in Streamlit
    st.pyplot(fig)

    #return fig

#---------------------------------------

# Function to plot clustered data
def plot_clustered_data(df, column_for_hover, centroids):

    # Create a DataFrame for centroids
    centroids_df = pd.DataFrame(centroids, columns=[0, 1])
    centroids_df['Cluster_Centroids'] = range(len(centroids_df))
    #centroids_df[f"{column_for_hover}_cleaned"] = "Centroid"  # Label centroids
    
    # Plot clustered data
    fig = px.scatter(df, x=0, y=1, color="Cluster_Labels", hover_data=[f"{column_for_hover}_cleaned"])
    
    # Update layout to remove overlapping keys
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(0,0,0,0)"  # Transparent background
    ))

    # Define colors for centroids
    centroid_colors = px.colors.qualitative.Light24[:len(centroids)]

    # Add centroids to the plot
    for i, centroid in centroids_df.iterrows():
        fig.add_trace(go.Scatter(x=[centroid[0]], y=[centroid[1]], mode='markers', marker=dict(symbol='x', color=centroid_colors[i], size=10), name=f'Centroid {i+1}'))
    
    # Update layout
    fig.update_layout(title="Clustered Data Visualization", xaxis_title="Principal Component 1", yaxis_title="Principal Component 2")
    return fig

#---------------------------------------

@st.cache_data(ttl=3600)
def preprocess_reviews(df, col_with_text):
  """
  Function to lemmatize, remove stop words and punctuation for a col in a df (e.g. staff reviews)
  """
  # Initialize NLTK's English stop words and lemmatizer
  stop_words = set(stopwords.words("english"))
  lemmatizer = WordNetLemmatizer()

  def clean_text_spacy(text):
    """Cleans a single text string using spaCy's lemmatization."""
    # Remove trailing whitespace characters
    text = text.strip()

    doc = nlp(text)

    # Remove stop words and lemmatize
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    clean_tokens = [token.lemma_.lower() for token in doc if token.lemma_.lower() not in stop_words]
    
    # Remove punctuation from clean tokens
    clean_tokens = [token for token in clean_tokens if token not in string.punctuation]

    return " ".join(clean_tokens)

  # Apply the cleaning function to the review column
  df[f"{col_with_text}_cleaned"] = df[col_with_text].apply(clean_text_spacy)

  return df

#---------------------------------------

def render_topic_modelling_outputs(dict_topic_to_df):
    """
    convenience function to render source data sligned to the dominant topic
    """
    for key in dict_topic_to_df.keys():
        st.subheader(key)
        st.dataframe(dict_topic_to_df[key])

#---------------------------------------

def render_topic_terms(lda, feature_names):
    """
    convenience function to render terms by topics from topic modelling
    """
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-10 - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        st.write(f"**Topic {topic_idx + 1}:** {', '.join(top_words)}")

#---------------------------------------

#@st.cache_data(ttl=3600)
def run_lda_topic_modelling(
    vectorizer_method,
    num_n_grams,
    remove_punctuation,
    #documents,
    num_topics,
    df,
    survey_responses,
    #stop_words,
    nlp = nlp
):
    
    #call the pre-process function - removes stop words
    #processed_docs_no_stop = [remove_stop_words(doc, stop_words) for doc in documents]

    #test to check length of documents (i.e. contains some lemmatised text)
    #st.subheader(":red[Processed Documents After Removing Stop Words:]")
    #for doc in processed_docs_no_stop:
    #    st.write(doc)

    # Apply lemmatization to your text data (prevents scenarios where a word can appear single and plural in the topic)
    #processed_docs_lemmatized = [lemmatize(doc, nlp) for doc in processed_docs_no_stop]

    #join the processed strings back together into a list of strings. 
    #joined_processed_docs_lemmatized = [' '.join(inner_list) for inner_list in processed_docs_lemmatized]
    
    joined_processed_docs_lemmatized = list(df[f'{survey_responses}_cleaned'])

    #test to check length of documents (i.e. contains some lemmatised text)
    #st.subheader(":red[Processed Documents Before Vectorization:]")
    #for doc in joined_processed_docs_lemmatized:
    #    st.write(doc)

    if vectorizer_method == 'tf-idf':
        vectorizer = TfidfVectorizer(max_df=0.85, min_df=1, stop_words='english', ngram_range=(1, num_n_grams)) #, tokenizer=lambda x: lemmatize_remove_punctuation(x, remove_punctuation))
    else:
        vectorizer = CountVectorizer(max_df=0.85, min_df=1, stop_words='english', ngram_range=(1, num_n_grams)) #, tokenizer=lambda x: lemmatize_remove_punctuation(x, remove_punctuation))

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
    df_mapped_response_topic = pd.DataFrame({'Response': df[f'{survey_responses}_cleaned'], 'Dominant Topic': dominant_topics+1})
    #populate a dictionary with subset dataframes, with each subset representing a topic, and the df consisting of responses predominately associated with that topic
    dict_topic_to_df = {}
    for topic_num in range(num_topics):
        df_subset = df_mapped_response_topic[df_mapped_response_topic['Dominant Topic'] == topic_num+1]
        dict_topic_to_df[f"Topic {topic_num+1}"] = df_subset

    # Extract topics 
    feature_names = vectorizer.get_feature_names_out()
        
    return feature_names, lda, dict_topic_to_df, joined_processed_docs_lemmatized

#---------------------------------------
@st.cache_data(ttl=3600)
def concatenate_text_by_column(df):
    """
    Concatenates text from all rows of each column in the DataFrame.
    
    Args:
    - df: pandas DataFrame containing text columns
    
    Returns:
    - concatenated_text_dict: dictionary with column labels as keys and concatenated text strings as values
    """
    concatenated_text_dict = {}
    
    # Loop over the columns of the DataFrame
    for column in df.columns:
        # Concatenate text from all rows of the column
        concatenated_text = ' '.join(df[column].astype(str))
        # Store the concatenated text in the dictionary
        concatenated_text_dict[column] = concatenated_text
    
    return concatenated_text_dict


#---------------------------------------

@st.cache_data(ttl=3600)
def clean_text_spacy(text):
    """Cleans a single text string using spaCy's lemmatization.
    Removes stop words and punctuation.
    intended use case on strategy objectives"""
    # Remove trailing whitespace characters
    text = text.strip()

    doc = nlp(text)

    # Remove stop words and lemmatize
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    clean_tokens = [token.lemma_.lower() for token in doc if token.lemma_.lower() not in stop_words]
    
    # Remove punctuation from clean tokens
    clean_tokens = [token for token in clean_tokens if token not in string.punctuation]

    return " ".join(clean_tokens)

#---------------------------------------

@st.cache_data(ttl=3600)
def compare_two_texts(survey_response_series, strategy_objective_text, strat_objective_reference):
  
  cleaned_strat_objective_text = clean_text_spacy(strategy_objective_text)
  survey_docs = [nlp(response) for response in survey_response_series]
  
  list_responses = []
  list_strat_objective_reference = []
  list_similarity_scores = []

  for response_doc in survey_docs:
    objective_doc = nlp(cleaned_strat_objective_text)  # Process objective as a Doc object
    similarity = response_doc.similarity(objective_doc)

    #print(f"Response: {response_doc.text[:30]} - Objective: Aspect1 - Similarity: {similarity:.2f}")

    #update lists
    list_responses.append(str(response_doc))
    list_strat_objective_reference.append(strat_objective_reference)
    list_similarity_scores.append(similarity)

  dict_results = {'responses' : list_responses,
                  'strat_obj_reference': list_strat_objective_reference,
                  'similarity_score': list_similarity_scores}
  
  df_results = pd.DataFrame.from_dict(dict_results)
  
  return df_results



#---------------------------------------
#<<< Sentiment analysis functions start >>>
#---------------------------------------


def render_clean_text_user_inputs():
    col1, col2, col3 = st.columns(3)
    with col1:
        sent_analysis_remove_punc = st.selectbox(label='Would you like to remove punctuation?', options=['Yes', 'No'], help='Some punctuation may change the tone of the text - it may be useful to attempt with and without punctuation present.')
        if sent_analysis_remove_punc == 'Yes':
            sent_analysis_remove_punc = True
        else:
            sent_analysis_remove_punc = False
    with col2:
        sent_analysis_lemmatize = st.selectbox(label='Would you like to lemmatize the text?', options=['Yes', 'No'], help='This determines whether to reduce a word to its root form. It can help standardise the text to an extent and may aid comparison, but in doing so, it loses any nuanced or abstract terms.')
        if sent_analysis_lemmatize == 'Yes':
            sent_analysis_lemmatize = True
        else:
            sent_analysis_lemmatize = False
    with col3:
        sent_analysis_remove_stopwords = st.selectbox(label='Would you like to remove stopwords?', options=['Yes', 'No'], help='This removes words such as "not", "the" etc. Careful as this may change the context.')
        if sent_analysis_remove_stopwords == 'Yes':
            sent_analysis_remove_stopwords = True
        else:
            sent_analysis_remove_stopwords = False

    return sent_analysis_remove_punc, sent_analysis_lemmatize, sent_analysis_remove_stopwords


#---------------------------------------

def clean_text(text, remove_punctuation=False, lemmatize=False, remove_stopwords=False):
    """
    Cleans the input text based on specified options.
    
    Args:
    - text (str): The input text to be cleaned.
    - remove_punctuation (bool): Whether to remove punctuation from the text.
    - lemmatize (bool): Whether to lemmatize words in the text.
    - remove_stopwords (bool): Whether to remove stopwords from the text.
    
    Returns:
    - str: The cleaned text.
    """
    # Process the text using spaCy
    doc = nlp(text)
    
    # Initialize an empty list to hold processed words
    processed_words = []
    
    for token in doc:
        # Initialize a flag to indicate whether the token should be included
        include_token = True
        
        # Remove stopwords
        if remove_stopwords and token.text.lower() in STOP_WORDS:
            include_token = False
        
        # Remove punctuation
        if remove_punctuation and token.text in string.punctuation:
            include_token = False
            
        # Lemmatize
        if lemmatize:
            processed_word = token.lemma_
        else:
            processed_word = token.text
        
        # If the token is to be included, add it to the list
        if include_token:
            processed_words.append(processed_word)
    
    # Join the processed words back into a single string
    cleaned_text = ' '.join(processed_words)
    
    return cleaned_text


#---------------------------------------

def user_selection_of_summary_metric():
    """
    Helper function to render select box for user selection of summary metric
    """
    col1, col2 = st.columns(2)
    with col1:
        aggregate_metric = st.selectbox(
            label='What aggregate metric do you want to use for sentiment analysis?', 
            options=['mean', 'median'], 
            help='If the distribution of sentiment scores is roughly normal, use mean. Else, if the distribution is skewed, use the median.')
    
    return aggregate_metric
#---------------------------------------

def analyze_sentiment_vader(text, neutral_tolerance=0.05):
    """
    Perform sentiment analysis on the provided text using VADER, with a specified tolerance for neutral sentiment.
    
    Args:
    - text (str): The text to analyze.
    - neutral_tolerance (float): The tolerance range around 0 to consider the sentiment as Neutral. 
                                  Sentiments with a compound score within [-neutral_tolerance, +neutral_tolerance] 
                                  will be classified as Neutral.
    
    Returns:
    - dict: A dictionary containing the sentiment ("Positive", "Negative", or "Neutral") and the compound score.
    """
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)
    compound_score = scores['compound']
    
    # Determine sentiment based on the compound score and neutral_tolerance
    if -neutral_tolerance < compound_score < neutral_tolerance:
        sentiment = "Neutral"
    elif compound_score >= neutral_tolerance:
        sentiment = "Positive"
    else:
        sentiment = "Negative"
    
    return {"sentiment": sentiment, "compound": compound_score}

#---------------------------------------

@st.cache_data(ttl=3600)
def perform_sentiment_analysis(df, text_column, neutral_tolerance=0.1):
    """
    Applies VADER sentiment analysis on a DataFrame text column, ensuring the text column remains the first column.
    
    Args:
    - df (pd.DataFrame): The DataFrame containing the text data.
    - text_column (str): The name of the column containing the text to analyze.
    - neutral_tolerance (float): Tolerance for classifying sentiments as Neutral.
    
    Returns:
    - pd.DataFrame: A DataFrame with the original text column first, followed by sentiment analysis results.
    """
    # Perform sentiment analysis and store results in a list
    analysis_results = df[text_column].apply(lambda x: analyze_sentiment_vader(x, neutral_tolerance))
    
    # Convert the list of dictionaries to a DataFrame
    results_df = pd.DataFrame(analysis_results.tolist())
    
    # Insert the original text column at the beginning of the results DataFrame
    results_df.insert(0, text_column, df[text_column])
    
    #rename compound to Sentiment_score
    results_df.rename(columns={'compound': 'Sentiment_score'}, inplace=True)

    return results_df

#---------------------------------------

@st.cache_data(ttl=3600)
def plot_sentiment_histogram(df, compound_column='Sentiment_score'):
    """
    Plots a histogram of sentiment analysis compound scores using Altair,
    coloring the bars based on sentiment score ranges.

    Args:
    - df (pd.DataFrame): DataFrame containing sentiment analysis results, including compound scores.
    - compound_column (str): Name of the column in df containing the compound sentiment scores.

    Returns:
    - alt.Chart: An Altair chart object ready to be displayed.
    """
    # Define a new column in the DataFrame for sentiment score categories
    df['Sentiment Category'] = pd.cut(
        df[compound_column],
        bins=[-1, -0.05, 0.05, 1],
        labels=['Negative', 'Neutral', 'Positive']
    )
    
    # Create the histogram
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X(f"{compound_column}:Q", bin=alt.Bin(maxbins=50), title='Sentiment_score'),
        y=alt.Y('count()', title='Count'),
        color=alt.Color('Sentiment Category:N', scale=alt.Scale(domain=['Negative', 'Neutral', 'Positive'],
                                                                 range=['red', 'gold', 'green'])),
        tooltip=[alt.Tooltip(f"{compound_column}:Q", title='Sentiment_score'), alt.Tooltip('count()', title='Count')]
    ).properties(
        title='Histogram of overall Sentiment Scores'
    )
    
    return chart

#---------------------------------------

def concatenate_columns(source_df, target_df, target_columns):
    """
    Concatenates specified columns from the target DataFrame to the source DataFrame.
    
    Args:
    - source_df (pd.DataFrame): The DataFrame to which the columns will be added.
    - target_df (pd.DataFrame): The DataFrame from which the columns will be taken.
    - target_columns (list of str): The list of column names from the target DataFrame to concatenate to the source DataFrame.
    
    Returns:
    - pd.DataFrame: A new DataFrame with the specified columns from the target DataFrame added to the source DataFrame.
    """
    # Ensure that the source DataFrame and target DataFrame have the same number of rows
    if len(source_df) != len(target_df):
        raise ValueError("Source and target DataFrames must have the same number of rows")
    
    # Select the specified columns from the target DataFrame
    columns_to_add = target_df[target_columns]
    
    # Concatenate the selected columns to the source DataFrame
    concatenated_df = pd.concat([source_df, columns_to_add], axis=1)
    
    return concatenated_df

#---------------------------------------

@st.cache_data(ttl=3600)
def sentiment_analysis_by_strategy_aspect(df, aspect_columns, sentiment_column):
    """
    Analyzes sentiment scores by strategy aspect and alignment classification.
    
    Args:
    - df (pd.DataFrame): The DataFrame containing sentiment scores and alignment classifications.
    - aspect_columns (list of str): The list of column names corresponding to strategy aspect alignments.
    - sentiment_column (str): The name of the column containing sentiment scores.
    
    Returns:
    - pd.DataFrame: A DataFrame with average sentiment scores for each alignment classification within each strategy aspect.
    """
    # Initialize an empty list to hold the grouped DataFrames
    grouped_dfs = []
    
    # Iterate through each strategy aspect column
    for aspect in aspect_columns:
        # Group by the strategy aspect column and calculate the mean sentiment score for each alignment classification
        grouped = df.groupby(aspect)[sentiment_column].mean().reset_index()
        grouped.rename(columns={sentiment_column: f"{aspect}_Avg_Sentiment", aspect: "Alignment"}, inplace=True)
        grouped['Strategy_Aspect'] = aspect  # Add a column to indicate the strategy aspect
        
        # Append the grouped DataFrame to the list
        grouped_dfs.append(grouped)
    
    # Concatenate all grouped DataFrames in the list
    results_df = pd.concat(grouped_dfs, axis=0, ignore_index=True)
    
    return results_df

#---------------------------------------
'''
@st.cache_data(ttl=3600)
def sentiment_analysis_by_strategy_aspect_compressed(df, aspect_columns, sentiment_column):
    """
    Analyzes and compresses sentiment scores by strategy aspect and alignment classification into a compact DataFrame.
    
    Args:
    - df (pd.DataFrame): The DataFrame containing sentiment scores and alignment classifications.
    - aspect_columns (list of str): The list of column names corresponding to strategy aspect alignments.
    - sentiment_column (str): The name of the column containing sentiment scores.
    
    Returns:
    - pd.DataFrame: A compressed DataFrame with 'Alignment' in the left column and strategy aspects as other columns,
                    where each cell contains the average sentiment score for the combination.
    """
    # Initialize an empty DataFrame to hold the compressed results
    compressed_df = pd.DataFrame(columns=['Alignment'] + aspect_columns)
    
    # Define the alignment categories
    alignment_categories = ['Low Alignment', 'Moderate Alignment', 'High Alignment']
    
    # Populate the 'Alignment' column with the alignment categories
    compressed_df['Alignment'] = alignment_categories
    
    # Iterate through each strategy aspect column to calculate the mean sentiment score for each alignment classification
    for aspect in aspect_columns:
        # Initialize a temporary dictionary to store average sentiment scores for this aspect
        aspect_avg_scores = {}
        
        for alignment in alignment_categories:
            # Calculate the mean sentiment score for the current alignment classification within the current aspect
            mean_score = df[df[aspect] == alignment][sentiment_column].mean()
            aspect_avg_scores[alignment] = mean_score
        
        # Add the calculated average scores to the compressed DataFrame
        compressed_df[aspect] = compressed_df['Alignment'].map(aspect_avg_scores)
    
    return compressed_df
'''
#---------------------------------------

@st.cache_data(ttl=3600)
def sentiment_analysis_by_strategy_aspect_compressed(df, aspect_columns, sentiment_column, metric="mean"):
    """
    Analyzes and compresses sentiment scores by strategy aspect and alignment classification into a compact DataFrame,
    using either the mean or median sentiment score based on the 'metric' argument.
    
    Args:
    - df (pd.DataFrame): The DataFrame containing sentiment scores and alignment classifications.
    - aspect_columns (list of str): The list of column names corresponding to strategy aspect alignments.
    - sentiment_column (str): The name of the column containing sentiment scores.
    - metric (str): The summary metric to use for aggregating sentiment scores, either "mean" or "median".
    
    Returns:
    - pd.DataFrame: A compressed DataFrame with 'Alignment' in the left column and strategy aspects as other columns,
                    where each cell contains the aggregated sentiment score for the combination.
    """
    # Initialize an empty DataFrame to hold the compressed results
    compressed_df = pd.DataFrame(columns=['Alignment'] + aspect_columns)
    
    # Define the alignment categories
    alignment_categories = ['Low Alignment', 'Moderate Alignment', 'High Alignment']
    
    # Populate the 'Alignment' column with the alignment categories
    compressed_df['Alignment'] = alignment_categories
    
    # Select the aggregation function based on the 'metric' argument
    if metric == "median":
        agg_func = pd.Series.median
    elif metric == 'mean':  
        agg_func = pd.Series.mean
    else:
        st.write('Aggregate score not recognised. Re-select in parameters and try again.')
    
    # Iterate through each strategy aspect column to calculate the aggregated sentiment score for each alignment classification
    for aspect in aspect_columns:
        aspect_scores = {}
        for alignment in alignment_categories:
            filtered_scores = df[df[aspect] == alignment][sentiment_column]
            aggregated_score = agg_func(filtered_scores)
            aspect_scores[alignment] = aggregated_score
        
        # Add the calculated aggregated scores to the compressed DataFrame
        compressed_df[aspect] = compressed_df['Alignment'].map(aspect_scores)
    
    return compressed_df

#---------------------------------------

@st.cache_data(ttl=3600)
def visualize_sentiment_heatmap_altair(df_sentiment_analysis, metric):
    """
    Visualizes the sentiment analysis results as a heatmap using Altair.
    
    Args:
    - df_sentiment_analysis (pd.DataFrame): DataFrame with 'Alignment' as the first column,
                                            strategy aspects as subsequent columns, and average sentiment scores.
    """
    # Melt the DataFrame to long format suitable for Altair
    df_long = df_sentiment_analysis.melt(id_vars=['Alignment'], var_name='Strategy Aspect', value_name=f'{metric} sentiment score')
    
    # Define the desired order for the 'Alignment' categories
    alignment_order = ['High Alignment', 'Moderate Alignment', 'Low Alignment']

    # Create the heatmap
    heatmap = alt.Chart(df_long).mark_rect().encode(
        x='Strategy Aspect:O',  # O denotes an ordinal data type
        y=alt.Y('Alignment:O', sort=alignment_order),
        color=alt.Color(f'{metric} sentiment score:Q', scale=alt.Scale(scheme='yellowgreenblue')),
        tooltip=['Strategy Aspect', 'Alignment', f'{metric} sentiment score']
    ).properties(
        title='Sentiment Analysis by Comparison Text component and Alignment'
    )
    
    return heatmap

#---------------------------------------

def filter_responses_for_analysis(df, aspect_column, alignment_value, sentiment_score_column, sentiment_threshold = 0.05):
    """
    Filters the DataFrame for responses associated with a specific alignment and sentiment score.
    
    Args:
    - df (pd.DataFrame): The original DataFrame containing survey responses and their analysis results.
    - aspect_column (str): Column name indicating the strategy aspect alignment classification.
    - alignment_value (str): The alignment classification to filter by (e.g., 'Low Alignment').
    - sentiment_score_column (str): Column name containing the sentiment scores.
    - sentiment_threshold (float): Threshold for filtering sentiment scores (e.g., scores below this value for negative sentiment).
    
    Returns:
    - pd.DataFrame: Filtered DataFrame with responses meeting the specified criteria.
    """
    # Filter the DataFrame based on the specified alignment and sentiment criteria
    filtered_df = df[(df[aspect_column] == alignment_value) & (df[sentiment_score_column] <= sentiment_threshold)]
    
    return filtered_df

#---------------------------------------
#TODO NOT TESTED THIS FUNCTION YET
def filter_responses_for_analysis_central_tendency(df, aspect_column, alignment_value, sentiment_score_column, central_tendency="mean", tolerance=0.1):
    """
    Filters the DataFrame for responses associated with a specific alignment and within a range around the central sentiment score (mean or median).
    
    Args:
    - df (pd.DataFrame): The original DataFrame containing survey responses and their analysis results.
    - aspect_column (str): Column name indicating the strategy aspect alignment classification.
    - alignment_value (str): The alignment classification to filter by (e.g., 'Low Alignment').
    - sentiment_score_column (str): Column name containing the sentiment scores.
    - central_tendency (str): "mean" or "median" to specify which central tendency measure to use for filtering.
    - tolerance (float): Specifies the range around the central tendency measure to include responses. For example, a tolerance of 0.1 means including responses within 0.1 units of the mean/median score.
    
    Returns:
    - pd.DataFrame: Filtered DataFrame with responses meeting the specified criteria.
    """
    # Subset the DataFrame based on the specified alignment
    aligned_df = df[df[aspect_column] == alignment_value]
    
    # Calculate the central tendency measure for the subset
    if central_tendency == "median":
        central_value = aligned_df[sentiment_score_column].median()
    else:  # Default to mean if anything other than "median" is provided
        central_value = aligned_df[sentiment_score_column].mean()
    
    # Define the upper and lower bounds based on the tolerance
    lower_bound = central_value - tolerance
    upper_bound = central_value + tolerance
    
    # Filter the subset based on the calculated bounds
    filtered_df = aligned_df[(aligned_df[sentiment_score_column] >= lower_bound) & (aligned_df[sentiment_score_column] <= upper_bound)]
    
    return filtered_df

#---------------------------------------

def render_density_plot(df, sentiment_column_name="compound"):
    """
    Renders a density plot of sentiment scores using Altair.
    
    Args:
    - df (pd.DataFrame): The DataFrame containing sentiment scores.
    - sentiment_column_name (str): The column name in df that contains the sentiment scores. Defaults to "compound".
    """
    # Create the density plot
    density_plot = alt.Chart(df).transform_density(
        sentiment_column_name,
        as_=[sentiment_column_name, 'density'],
    ).mark_area(opacity=0.5).encode(
        x=alt.X(f"{sentiment_column_name}:Q", title="Sentiment Score"),
        y=alt.Y("density:Q", title="Density"),
        tooltip=[alt.Tooltip(f"{sentiment_column_name}:Q", title="Sentiment Score"),
                 alt.Tooltip("density:Q", title="Density")]
    ).properties(
        title="Density Plot of Sentiment Scores"
    )
    
    # Display the plot
    st.altair_chart(density_plot, use_container_width=True)

#---------------------------------------
# Correlation Analysis functions
#---------------------------------------



#---------------------------------------
#
#---------------------------------------


#---------------------------------------
#<<< kmeans clustering functions start >>>
#---------------------------------------

def run_kmeans(data, k):
    """
    Function to perform K-means clustering on the given data.
    
    Parameters:
    - data: The input data for clustering (numpy array or pandas DataFrame).
    - k: The number of clusters to identify.
    
    Returns:
    - labels: The cluster labels assigned to each data point.
    - centroids: The centroids of the clusters.
    """
    # Initialize KMeans object
    kmeans = KMeans(n_clusters=k, random_state=42)
    
    # Fit KMeans model to the data
    kmeans.fit(data)
    
    # Get cluster labels and centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    return labels, centroids, kmeans

#---------------------------------------

def get_k_means_performance_metrics(kmeans_object, source_df_passed_to_k_means, cluster_labels):
    """
    function to produce the performance metrics of kmeans. 
    requires the source df that was passed to kmeans, i.e. without the cluster labels column appended
    requires the cluster_labels series returned from the kmeans object. 
    requires the kmeans object itself.
    """
    # Obtain inertia
    inertia = kmeans_object.inertia_

    # Obtain silhouette score
    silhouette = silhouette_score(source_df_passed_to_k_means, cluster_labels)

    return inertia, silhouette

#---------------------------------------

def find_optimal_k(data, max_k=10):
    """
    Function to identify the optimal value of K for K-means clustering using the silhouette score.
    
    Parameters:
    - data: The input data for clustering (numpy array or pandas DataFrame).
    - max_k: The maximum value of K to consider (default: 10).
    
    Returns:
    - optimal_k: The optimal value of K based on the silhouette score.
    - silhouette_scores: Silhouette scores for each value of K.
    """
    silhouette_scores = []
    
    # Calculate silhouette score for each value of K
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        silhouette_scores.append(silhouette_score(data, labels))
    
    # Find optimal K based on silhouette score
    optimal_k = np.argmax(silhouette_scores) + 2  # Add 2 because silhouette score starts from K=2
    
    return optimal_k, silhouette_scores

#---------------------------------------

def plot_silhouette_scores(silhouette_scores):
    """
    Function to plot silhouette scores for different values of K.
    
    Parameters:
    - silhouette_scores: Silhouette scores for each value of K.
    
    Returns:
    - fig: Matplotlib figure object.
    """
    k_values = range(2, len(silhouette_scores) + 2)
    fig, ax = plt.subplots()
    ax.plot(k_values, silhouette_scores, marker='o')
    ax.set_xlabel('Number of Clusters (K)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score vs. Number of Clusters')
    return fig


#---------------------------------------

def append_cluster_labels(df, cluster_labels):
    """
    Function to append cluster labels to the original DataFrame.
    
    Parameters:
    - df: Original DataFrame containing review text and similarity scores.
    - cluster_labels: Series containing cluster labels.
    
    Returns:
    - updated_df: DataFrame with cluster labels appended as a new column.
    """
    # Make a copy of the original DataFrame to avoid modifying the original
    updated_df = df.copy()
    
    # Append cluster labels as a new column
    updated_df['Cluster_Labels'] = cluster_labels
    
    return updated_df


#---------------------------------------

def append_df_to_df(left_df, right_df):
    """
    Function to append a DataFrame to another DataFrame.
    
    Parameters:
    - df1: Original DataFrame.
    - df2: DataFrame to be appended to df1.
    
    Returns:
    - updated_df: DataFrame with df2 appended as new columns.
    """
    # Make a copy of the original DataFrame to avoid modifying the original
    #updated_df = left_df.copy()
    
    # Append df2 to df1
    #updated_df = pd.concat([updated_df, right_df], axis=1)
    updated_df = pd.concat([left_df, right_df], axis=1)
    
    return updated_df

#---------------------------------------

def run_correlation_analysis(df, score_columns, method='spearman'):
    """
    Runs correlation analysis on specified score columns within a DataFrame.
    
    Parameters:
    - df: DataFrame containing the columns to be correlated.
    - score_columns: List of column names to include in the correlation analysis.
    - method: 'pearson' for Pearson's correlation, 'spearman' for Spearman's rank correlation.
    
    Returns:
    - correlation_matrix: DataFrame representing the correlation matrix of the specified columns.
    """
    correlation_matrix = df[score_columns].corr(method=method)
    return correlation_matrix

#---------------------------------------

def visualize_correlation_matrix_altair(correlation_matrix, title="Correlation Matrix"):
    """
    Visualizes a correlation matrix using Altair, suitable for display in a Streamlit app.
    
    Args:
    - correlation_matrix (pd.DataFrame): The correlation matrix to visualize.
    - title (str): Title of the plot.
    """
    # Reset index to transform the DataFrame into a long format
    correlation_long = correlation_matrix.reset_index().melt(id_vars='index')
    correlation_long.columns = ['Variable 1', 'Variable 2', 'Correlation']
    
    # Create the heatmap
    heatmap = alt.Chart(correlation_long).mark_rect().encode(
        x=alt.X('Variable 1:O', title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('Variable 2:O', title=None),
        color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='redblue', domain=(-1, 1))),
        tooltip=[
            alt.Tooltip('Variable 1:N', title='Variable 1'),
            alt.Tooltip('Variable 2:N', title='Variable 2'),
            alt.Tooltip('Correlation:Q', title='Correlation')
        ]
    ).properties(
        title=title,
        width=300,  # Adjust width as needed
        height=300   # Adjust height as needed
    )
    
    return heatmap


#---------------------------------------

def visualize_combined_correlation_heatmap(dict_correlation_matrices, sentiment_column_name, title="Combined Correlation Heatmap"):
    """
    Visualizes combined correlation data from multiple matrices into a single heatmap using Altair,
    suitable for display in a Streamlit app.
    
    Args:
    - dict_correlation_matrices (dict): A dictionary where keys are strategy aspect names and values are the correlation matrices.
    - title (str): Title of the plot.
    """
    # Initialize an empty DataFrame for combined data
    combined_data = pd.DataFrame()

    # Loop through the dictionary to extract and transform correlation data
    for aspect, matrix in dict_correlation_matrices.items():
        # Extract 'sentiment_score' correlation with the current aspect
        # Assuming the sentiment score's correlations are stored in a row/column named 'sentiment_score'
        # Adjust the extraction logic based on your actual matrix structure
        correlation_data = matrix.loc[matrix.index == sentiment_column_name, [aspect]].reset_index()
        correlation_data.columns = ['Variable', 'Correlation']
        correlation_data['Strategy Aspect'] = aspect  # Add the strategy aspect as a column

        # Append to the combined DataFrame
        combined_data = pd.concat([combined_data, correlation_data], ignore_index=True)
    
    # Rename 'Variable' column to a more descriptive name if desired
    combined_data.rename(columns={'Variable': 'Metric'}, inplace=True)

    # Create the heatmap
    heatmap = alt.Chart(combined_data).mark_rect().encode(
        x=alt.X('Strategy Aspect:O', title=None),
        y=alt.Y('Metric:O', title=None),
        color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='redblue', domain=(-1, 1))),
        tooltip=[
            alt.Tooltip('Strategy Aspect:N', title='Strategy Aspect'),
            alt.Tooltip('Metric:N', title='Metric'),
            alt.Tooltip('Correlation:Q', title='Correlation')
        ]
    ).properties(
        title=title,
        #width=600,  # Adjust width based on the number of aspects
        height=250   # Adjust height as needed
    )
    
    return heatmap


#---------------------------------------

def run_correlation_analysis_with_significance_and_filter(df, aspect_columns, sentiment_score_column, method='spearman', p_value_threshold=0.05):
    """
    Runs correlation analysis with significance testing, returns a DataFrame of results,
    and filters results based on a p-value threshold.
    
    Parameters:
    - df: DataFrame containing the columns to be correlated.
    - aspect_columns: List of column names for strategy aspects.
    - sentiment_score_column: Name of the column containing sentiment scores.
    - method: Correlation method ('spearman' by default).
    - p_value_threshold: Threshold for filtering based on p-value (default 0.05).
    
    Returns:
    - all_results_df: DataFrame containing all correlation results.
    - filtered_results_df: DataFrame containing only results where p-value < p_value_threshold.
    """
    results = []

    sig_results_present = False

    for aspect_column in aspect_columns:
        # Calculate Spearman correlation and p-value
        corr_coefficient, p_value = spearmanr(df[sentiment_score_column], df[aspect_column])
        
        #update bool flag for sig results
        if p_value < p_value_threshold:
            sig_results_present = True

        # Append results including aspect name, correlation, and p-value to the list
        results.append({
            'Strategy Aspect': aspect_column,
            'Correlation': corr_coefficient,
            'P-Value': p_value
        })
    
    # Convert results list to DataFrame
    all_results_df = pd.DataFrame(results)
    
    # Filter DataFrame based on p-value threshold
    filtered_results_df = all_results_df[all_results_df['P-Value'] < p_value_threshold]
    
    return all_results_df, filtered_results_df, sig_results_present

#---------------------------------------
#<<< kmeans clustering functions end >>>
#---------------------------------------

#---------------------------------------
#<<< PCA functions start >>>
#---------------------------------------

def perform_pca(data, n_components=2):
    """
    Function to perform Principal Component Analysis (PCA) on the given data.
    
    Parameters:
    - data: The input data for PCA (numpy array or pandas DataFrame).
    - n_components: The number of principal components to retain (default: 2).
    
    Returns:
    - pca_data: The transformed data after PCA.
    - pca: The PCA object fitted to the data.
    - explanation: Explanation of the findings from PCA.
    """
    # Standardize the data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(standardized_data)
    
    # Explanation of the findings
    explanation = f"**Explained Variance Ratio**:\
        \nPrincipal Component 1 accounts for :red[***{round((pca.explained_variance_ratio_[0]*100),2)}%***] of the variance\
        \nPrincipal Component 2 accounts for :red[***{round((pca.explained_variance_ratio_[1]*100),2)}%***] of the variance\
        \nCombined the principal components capture about :red[***{sum(pca.explained_variance_ratio_) * 100:.2f}%***] of the information in the original 4-dimensional dataset\
        nRetaining a high percentage of the total variance (typically 80% or higher) in the reduced-dimensional representation is often considered desirable, as it indicates \
        that the principal components capture the majority of the information present in the original dataset."
    
    return pca_data, pca, explanation

#---------------------------------------
#<<< PCA functions end >>>
#---------------------------------------

#---------------------------------------
#<<< Topic Modelling Starts >>>
#---------------------------------------

def apply_user_filters(
    df,
    sentiment_column,
    sentiment_score_range=None, 
    comparison_text_aspect_alignment=None, 
    selected_alignment_labels=None, 
    selected_sentiment_classes=None, 
    selected_cluster_labels=None, 
    comparison_text_aspect_sim_score=None, 
    similarity_score_range=None, 
    dict_concat_strat_objectives=None):
    
    # Filter by Sentiment Score
    if sentiment_score_range:
        df = df[df[sentiment_column].between(*sentiment_score_range)]
        
    # Filter by Alignment Labels
    if selected_alignment_labels and comparison_text_aspect_alignment:
        alignment_columns = [f"{aspect}-Alignment" for aspect in comparison_text_aspect_alignment]
        alignment_filter = df[alignment_columns].apply(lambda x: x.isin(selected_alignment_labels)).any(axis=1)
        df = df[alignment_filter]
    
    # Filter by Sentiment Classification
    if selected_sentiment_classes:
        df = df[df['sentiment_classification'].isin(selected_sentiment_classes)]
        
    # Filter by Cluster Label
    if selected_cluster_labels is not None:
        # Ensure selected cluster labels are integers
        selected_cluster_labels_int = [int(label) for label in selected_cluster_labels]
        df = df[df['Cluster_Labels'].isin(selected_cluster_labels_int)]
    
    # Filter by Similarity Scores
    if similarity_score_range and comparison_text_aspect_sim_score:
        for aspect in comparison_text_aspect_sim_score:
            column_name = f"{aspect}-similarity"
            if column_name in df.columns:  # Check if column exists to avoid KeyError
                df = df[df[column_name].between(*similarity_score_range)]
    
    return df





#---------------------------------------

# Sample function for LDA fitting and scoring

def fit_lda_and_score(data, num_topics, max_words, vectorizer_method):
    """
    Used to find 'optimal' params without exhaustive brute force of all possible combinations
    """
    vectorizer = (TfidfVectorizer if vectorizer_method == 'tf-idf' else CountVectorizer)(max_df=0.95, min_df=2, max_features=max_words)
    dtm = vectorizer.fit_transform(data)
    
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(dtm)
    
    # Use perplexity as the score; lower is generally better
    score = lda.perplexity(dtm)
    
    return score



#---------------------------------------

def random_search_lda_params(df, text_column, n_iter=10):
    """
    Performs a random search to find the optimal LDA model parameters.

    Args:
    - df (pd.DataFrame): The DataFrame containing the text data.
    - text_column (str): The name of the column containing the text data.
    - n_iter (int): The number of iterations for the random search.

    Returns:
    - best_params (dict): The best-performing parameters found.
    - best_score (float): The perplexity score of the best-performing LDA model.
    """
    best_score = np.inf
    best_params = None
    unique_words = count_unique_words(df, text_column)  # Number of unique words

    for _ in range(n_iter):
        num_topics = np.random.randint(1, 11)
        ngram_range = (1, np.random.randint(1, 11))
        max_features = np.random.randint(1, unique_words)  # max_features between 1 and the total number of unique words
        vectorizer_method = np.random.choice(['tf-idf', 'count'])
        
        vectorizer = (TfidfVectorizer if vectorizer_method == 'tf-idf' else CountVectorizer)(
            ngram_range=ngram_range,
            max_features=max_features,
            stop_words='english'
        )

        # Split data to train and test set
        train_text, test_text = train_test_split(df[text_column], test_size=0.1, random_state=42)
        
        # Fit-transform the text data
        dtm_train = vectorizer.fit_transform(train_text)
        dtm_test = vectorizer.transform(test_text)
        
        # Initialize and fit LDA
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(dtm_train)
        
        # Calculate perplexity on the test set
        perplexity = lda.perplexity(dtm_test)
        
        # Update best parameters if perplexity improves
        if perplexity < best_score:
            best_score = perplexity
            best_params = {
                'num_topics': num_topics,
                'ngram_range': ngram_range,
                'vectorizer_method': vectorizer_method,
                'max_features': max_features
            }

    return best_params, best_score

#---------------------------------------

def count_unique_words(df, text_column):
    """
    Counts the number of unique words across all text in a specified column of the DataFrame.

    Args:
    - df (pd.DataFrame): The DataFrame containing the text data.
    - text_column (str): The name of the column containing the text data.

    Returns:
    - int: The count of unique words.
    """
    vectorizer = CountVectorizer()
    dtm = vectorizer.fit_transform(df[text_column])
    return len(vectorizer.get_feature_names_out())

#---------------------------------------

def perform_topic_modeling_and_return_results(df, text_column, text_column_original, num_topics, ngram_range, max_features, vectorizer_method):
    """
    Performs topic modeling and returns structured data about the topics, document-topic distributions,
    includes the cleaned and original source text, sentiment score, and a dominant topic label for each document, and term weights.
    
    Args:
    - df (pd.DataFrame): DataFrame containing the text data.
    - text_column (str): Name of the column containing the cleaned text data.
    - text_column_original (str): Name of the column containing the original text data.
    - num_topics (int): Number of topics to generate.
    - ngram_range (tuple): N-gram range for vectorization.
    - max_features (int): Maximum number of features.
    - vectorizer_method (str): Specifies the vectorization method ('tf-idf' or 'count').
    
    Returns:
    - topics_df (pd.DataFrame): DataFrame with topics and their associated terms.
    - doc_topic_df (pd.DataFrame): DataFrame with the topic distribution for each document, including cleaned and original source text, sentiment score, and dominant topic.
    - term_topic_weights_df (pd.DataFrame): DataFrame with term weights across topics.
    - perplexity (float): The perplexity score of the trained LDA model.
    """
    # Initialize vectorizer based on the method chosen
    if vectorizer_method == 'tf-idf':
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english', ngram_range=(1, ngram_range), max_features=max_features)
    else:
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', ngram_range=(1, ngram_range), max_features=max_features)
    
    # Vectorization
    dtm = vectorizer.fit_transform(df[text_column])
    
    # LDA Topic Modeling
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
    lda.fit(dtm)
    
    # Calculate perplexity
    perplexity = lda.perplexity(dtm)
    
    # Prepare topics_df based on the top terms for each topic
    feature_names = vectorizer.get_feature_names_out()
    topics = {
        f"Topic {i+1}": [feature_names[index] for index in topic.argsort()[:-11:-1]]
        for i, topic in enumerate(lda.components_)
    }
    topics_df = pd.DataFrame(topics)
    
    # Prepare doc_topic_df including cleaned source text, original source text, and sentiment score
    doc_topic_distributions = lda.transform(dtm)
    doc_topic_df = pd.DataFrame(doc_topic_distributions, columns=[f"Topic {i+1}" for i in range(num_topics)])
    doc_topic_df.insert(0, text_column, df[text_column].values)  # Insert cleaned text column at position 0
    doc_topic_df.insert(1, text_column_original, df[text_column_original].values)  # Insert original text column at position 1
    doc_topic_df.insert(2, 'Sentiment_score', df['Sentiment_score'].values)  # Insert sentiment score column at position 2
    doc_topic_df.insert(3, 'sentiment', df['sentiment'].values)  # Insert sentiment label column at position 2

    # Calculate the dominant topic for each document
    dominant_topic = doc_topic_df.iloc[:, 4:].idxmax(axis=1)  # Adjust index range to skip source text columns and sentiment score
    doc_topic_df['Dominant Topic'] = dominant_topic.apply(lambda x: "Topic " + x.split()[1])  # Ensure labels are "Topic X"
    
    # Prepare term_topic_weights_df including term weights across topics
    term_topic_weights = []
    for topic_idx, topic in enumerate(lda.components_):
        for term_idx, weight in enumerate(topic):
            term_topic_weights.append({
                'Term': feature_names[term_idx],
                'Topic': f"Topic {topic_idx + 1}",
                'Weight': weight
            })
    term_topic_weights_df = pd.DataFrame(term_topic_weights)
    
    return topics_df, doc_topic_df, term_topic_weights_df, perplexity

#---------------------------------------

def generate_topic_wordclouds_by_dominant_topic(df, text_column, wordcloud_colour_scheme='viridis', dominant_topic_column='Dominant Topic'):
    """
    Generates word clouds for each dominant topic from the cleaned source text,
    and returns a dictionary of matplotlib figures.

    Args:
    - df (pd.DataFrame): DataFrame containing the text data and dominant topic for each document.
    - text_column (str): Name of the column containing the cleaned source text.
    - dominant_topic_column (str): Name of the column containing the dominant topic for each document.
    - wordcloud_colour_scheme (str): Color scheme for the word clouds.

    Returns:
    - dict: A dictionary where keys are topics ("Topic 1", "Topic 2", etc.) and values are matplotlib figure objects.
    """
    wordclouds = {}
    topics = df[dominant_topic_column].unique()
    
    for topic in sorted(topics):
        text_data = df[df[dominant_topic_column] == topic][text_column].str.cat(sep=' ')
        wordcloud = WordCloud(background_color='white', colormap=wordcloud_colour_scheme, contour_color='steelblue', contour_width=2).generate(text_data)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(f'{topic}', size=20)
        ax.axis('off')
        plt.close(fig)  # Close plt to prevent it from displaying immediately

        wordclouds[topic] = fig
    
    return wordclouds

#---------------------------------------

# Example conversion to long format 
def convert_to_long_format(topics_df):
    long_df = topics_df.reset_index().melt(id_vars='index', var_name='Topic', value_name='Weight')
    long_df.rename(columns={'index': 'Term'}, inplace=True)
    return long_df

#---------------------------------------

def create_interactive_heatmap_indivudal_topic(df, sort_terms=True):
    """
    Creates and displays an interactive heatmap using Altair.

    Args:
    - df (pd.DataFrame): The DataFrame to visualize.
    - sort_terms (bool, optional): Determines if terms should be sorted alphabetically. 
                                   Set to False to use the DataFrame's order. Defaults to True.
    """
    # Define how to sort the y-axis based on the sort_terms argument
    if sort_terms:
        sort_arg = 'y'
    else:
        sort_arg = alt.Y('Term:N', sort=None)  # Use DataFrame's order

    # Define the heatmap
    heatmap = alt.Chart(df).mark_rect().encode(
        x='Topic:N',
        y=sort_arg,  # Apply the determined sort argument
        color='Weight:Q',
        tooltip=['Term:N', 'Topic:N', 'Weight:Q']
    ).properties(
        title='Topic-Term Heatmap',
        width=600,
        height=max(300, 25*len(df))  # Adjust height based on number of terms
    ).interactive()

    # Display the heatmap in Streamlit
    st.altair_chart(heatmap, use_container_width=True)

#---------------------------------------

def create_interactive_heatmap(df):
    # Define the heatmap
    heatmap = alt.Chart(df).mark_rect().encode(
        x='Topic:N',
        y='Term:N',
        color='Weight:Q',
        tooltip=['Term:N', 'Topic:N', 'Weight:Q']
    ).properties(
        width=600,
        height=max(300, 25*len(df))  # Adjust height based on number of terms
    ).interactive()

    # Display the heatmap in Streamlit
    st.altair_chart(heatmap, use_container_width=True)

#---------------------------------------

def visualize_topic_distributions(term_topic_weights_df):
    """
    Visualizes the distribution of term weights for each topic in a single Altair chart.
    
    Args:
    - term_topic_weights_df (pd.DataFrame): DataFrame with term weights across topics.
    """
    chart = alt.Chart(term_topic_weights_df).transform_density(
        density='Weight',
        groupby=['Topic'],
        as_=['Weight', 'Density'],
        extent=[0, 1],  # Adjust this extent based on the range of your term weights
        counts=True
    ).mark_area(opacity=0.5).encode(
        alt.X('Weight:Q', title='Term Weight'),
        alt.Y('Density:Q', title='Density'),
        alt.Color('Topic:N', title='Topic'),
        tooltip=['Topic:N', 'Weight:Q', 'Density:Q']
    ).properties(
        title='Distribution of Term Weights by Topic'
    ).facet(
        facet='Topic:N',
        columns=2  # Adjust the number of columns based on how you want to layout your topics
    ).resolve_scale(
        y='independent'  # Allows each facet to have its own y-axis scale
    )

    # Display the chart in Streamlit
    st.altair_chart(chart, use_container_width=True)

#---------------------------------------

def visualize_term_weight_distributions_overlaid(df):
    """
    Visualizes the distribution of term weights for all topics overlaid in a single Altair plot.
    
    Args:
    - df (pd.DataFrame): DataFrame with term weights across topics. It must have 'Term', 'Topic', and 'Weight' columns.
    """
    # Create a base chart
    base = alt.Chart(df).transform_density(
        density='Weight',
        bandwidth=0.05,
        groupby=['Topic'],
        as_=['Weight', 'Density']
    )
    
    # Layer the density plots for each topic
    density_plot = base.mark_area(opacity=0.5).encode(
        alt.X('Weight:Q', title='Term Weight'),
        alt.Y('Density:Q', title='Density'),
        alt.Color('Topic:N', legend=alt.Legend(title="Topics")),
        tooltip=['Topic:N', 'Weight:Q', 'Density:Q']
    ).properties(title='Distribution of Term Weights by Topic')
    
    # Display the plot in Streamlit
    st.altair_chart(density_plot, use_container_width=True)

#---------------------------------------

def visualize_topic_probabilities(doc_topic_df, topic_label):
    """
    Visualizes topic probabilities for a specified topic in an interactive Altair chart.
    
    Args:
    - doc_topic_df (pd.DataFrame): DataFrame containing topic probabilities for each document.
    - topic_label (str): Label of the topic column to visualize.
    """
    chart = alt.Chart(doc_topic_df).mark_point().encode(
        x='index:Q',
        y=alt.Y(f'{topic_label}:Q', title='Topic Probability'),
        tooltip=['index', topic_label]
    ).interactive().properties(
        title=f'Distribution of Probabilities for {topic_label}',
        #width=600,
        height=400
    )
    
    st.altair_chart(chart, use_container_width=True)

#---------------------------------------

def detect_topic_outliers(doc_topic_df, topic_label, num_outliers=10):
    """
    Detects outliers in topic probabilities, both least associated and most aligned documents.
    
    Args:
    - doc_topic_df (pd.DataFrame): DataFrame containing topic probabilities for each document.
    - topic_label (str): The topic column to analyze for outliers.
    - num_outliers (int): Maximum number of outliers to return for both least and most associated documents.
    
    Returns:
    - (least_associated, most_aligned): Tuple of DataFrames with least associated and most aligned outliers.
    """
    # Calculate lower and upper quantiles
    lower_quantile = doc_topic_df[topic_label].quantile(0.25)
    upper_quantile = doc_topic_df[topic_label].quantile(0.75)
    
    # Filter documents based on quantiles
    least_associated = doc_topic_df[doc_topic_df[topic_label] < lower_quantile].sort_values(by=topic_label).head(num_outliers)
    most_aligned = doc_topic_df[doc_topic_df[topic_label] > upper_quantile].sort_values(by=topic_label, ascending=False).head(num_outliers)
    
    return least_associated, most_aligned

#---------------------------------------
#create test comparison text
#---------------------------------------


@st.cache_data(ttl=3600)
#note the objectives below are made up, for illustrative purposes when no file uploaded and dummy data is required
#these were produced using an LLM and are not reflective of actual objectives for a given organisation
def dummy_strat_objectives():
    maximize_technology = [
    "Implement AI-driven solutions to optimize business processes.",
    "Upgrade infrastructure to support cloud computing.",
    "Develop a mobile app to enhance customer engagement.",
    "Invest in IoT devices for real-time data analytics.",
    "Utilize blockchain technology to enhance security and transparency.",
    "Implement machine learning algorithms for predictive analytics.",
    ]

    enhance_sustainability = [
        "Reduce carbon footprint by adopting renewable energy sources.",
        "Implement waste reduction strategies across all operations.",
        "Promote recycling initiatives within the organization.",
        "Optimize supply chain to minimize environmental impact.",
        "Invest in eco-friendly packaging solutions.",
        "Partner with green suppliers to support sustainable practices.",
    ]

    strengthen_market_position = [
        "Expand market reach through strategic partnerships.",
        "Launch targeted marketing campaigns to reach new demographics.",
        "Enhance brand visibility through social media engagement.",
        "Conduct market research to identify emerging trends and opportunities.",
        "Diversify product offerings to cater to evolving customer needs.",
        "Establish industry leadership by showcasing expertise through thought leadership content.",
    ]

    improve_customer_experience = [
        "Personalize customer interactions through data-driven insights.",
        "Streamline the checkout process for a seamless purchasing experience.",
        "Implement a customer feedback system to gather insights and improve services.",
        "Offer 24/7 customer support through multiple channels.",
        "Enhance website usability for easier navigation.",
        "Develop loyalty programs to reward repeat customers.",
    ]


    strategy_data = {
        "aspect1": maximize_technology,
        "aspect2": enhance_sustainability,
        "aspect3": strengthen_market_position,
        "aspect4": improve_customer_experience
    }

    strategy_data_concat = {}
    for key in strategy_data.keys():
        combined_string = ""
        for string_list in strategy_data[key]:
            combined_string += string_list + " "
        strategy_data_concat[key] = combined_string
    return strategy_data, strategy_data_concat

#-------------------------------------
#the below staff survey responses are entirely made up, produced by gemini LLM for 
#the purpose of building and testing this apps functionality. They are not real.
@st.cache_data(ttl=3600)
def produce_staff_survey_dummy_data():
    # Create a pandas Series from the list of staff survey responses
    staff_responses_doing_well = [
        "Providing clear communication channels to ensure everyone is informed and engaged.",
        "Offering regular training opportunities tailored to individual needs and career aspirations.",
        "Encouraging a positive work environment where collaboration and support are valued.",
        "Recognizing employee achievements through various appreciation programs and rewards.",
        "Maintaining a focus on diversity and inclusion, ensuring everyone feels respected and valued.",
        "Supporting work-life balance initiatives such as flexible schedules and remote work options.",
        "Providing competitive compensation packages that reflect employees' contributions and market standards.",
        "Offering flexibility in work schedules to accommodate personal commitments and preferences.",
        "Promoting teamwork and collaboration through cross-functional projects and team-building activities.",
        "Encouraging innovation and creativity by fostering an environment where new ideas are welcomed and explored.",
        "Providing adequate resources for tasks, ensuring employees have the tools they need to succeed.",
        "Maintaining a clean and organized workspace to enhance productivity and well-being.",
        "Offering opportunities for career advancement through mentorship programs and professional development initiatives.",
        "Providing timely feedback and performance evaluations to help employees grow and improve.",
        "Offering comprehensive benefits packages that support employees' health, wellness, and financial security.",
        "Maintaining high standards of professionalism in all interactions and operations.",
        "Encouraging open communication among staff at all levels to foster transparency and trust.",
        "Providing opportunities for skill development through training workshops and educational resources.",
        "Offering mentorship programs to help employees navigate their career paths and reach their goals.",
        "Maintaining a strong commitment to customer satisfaction by delivering quality products and services.",
        "Implementing efficient processes and procedures to streamline workflows and maximize productivity.",
        "Ensuring a safe work environment through regular inspections and safety protocols.",
        "Promoting a healthy work-life balance by offering flexible work arrangements and wellness programs.",
        "Encouraging autonomy and independence by empowering employees to take ownership of their work.",
        "Recognizing and rewarding hard work and dedication through performance bonuses and awards.",
        "Providing opportunities for cross-training to enhance employees' skills and versatility.",
        "Demonstrating strong leadership from management through clear direction and support.",
        "Supporting employee well-being initiatives such as mental health resources and wellness activities.",
        "Providing clear goals and expectations to guide employees' efforts and priorities.",
        "Offering opportunities for professional growth through advanced training and certifications.",
        "Maintaining transparency in decision-making processes to foster trust and accountability.",
        "Supporting a culture of continuous learning through ongoing education and development opportunities.",
        "Providing access to necessary tools and technologies to facilitate efficient work processes.",
        "Encouraging a culture of feedback and improvement by soliciting input from employees.",
        "Demonstrating a commitment to sustainability practices through eco-friendly initiatives and policies.",
        "Fostering a sense of community within the organization through team-building events and social activities.",
        "Offering opportunities for personal development through self-assessment and goal-setting.",
        "Providing flexibility in work arrangements to accommodate individual needs and preferences.",
        "Demonstrating commitment to corporate social responsibility through community involvement and philanthropy.",
        "Encouraging work-life integration by offering support for personal and professional responsibilities.",
        "Maintaining a strong company culture based on shared values and mutual respect.",
        "Offering opportunities for leadership development through mentorship and training programs.",
        "Providing a supportive work environment for all employees regardless of background or identity.",
        "Demonstrating a commitment to employee wellness through health initiatives and benefits.",
        "Encouraging a healthy work-life balance by promoting time management and prioritization skills.",
        "Providing opportunities for cross-departmental collaboration to foster innovation and synergy.",
        "Maintaining a positive attitude towards change and adaptation to new circumstances.",
        "Demonstrating respect for work-life boundaries and personal time outside of work.",
        "Offering opportunities for professional networking through industry events and conferences.",
        "Providing opportunities for involvement in decision-making processes to empower employees.",
        "Maintaining open lines of communication between departments to facilitate collaboration.",
        "Supporting employee-driven initiatives that align with company goals and values.",
        "Demonstrating a commitment to employee recognition through formal and informal channels.",
        "Offering opportunities for flexible work arrangements to accommodate diverse needs.",
        "Providing access to employee assistance programs for support during challenging times.",
        "Demonstrating empathy and understanding towards employees' personal and professional challenges.",
        "Encouraging a culture of inclusivity and diversity through training and awareness programs.",
        "Providing opportunities for team-building activities to strengthen relationships and morale.",
        "Maintaining a supportive atmosphere for professional development and growth.",
        "Demonstrating flexibility in adapting to employees' individual needs and circumstances.",
        "Offering opportunities for volunteering and community engagement to give back.",
        "Providing access to leadership training programs to nurture future leaders within the organization.",
        "Demonstrating a commitment to work-life balance initiatives through policy and practice.",
        "Supporting employee wellness programs through incentives and resources.",
        "Offering opportunities for mentorship and coaching to support career development.",
        "Providing access to mental health resources and support services.",
        "Demonstrating a commitment to employee satisfaction through regular feedback and surveys.",
        "Offering opportunities for continuous improvement through training and development.",
        "Maintaining a positive and constructive work environment through effective conflict resolution.",
        "Demonstrating transparency in decision-making processes to build trust and accountability.",
        "Supporting a culture of work-life harmony by promoting balance and flexibility.",
        "Offering opportunities for cross-functional collaboration to leverage diverse expertise.",
        "Providing access to learning and development resources to support career growth.",
        "Demonstrating flexibility in accommodating employee needs and preferences.",
        "Encouraging a culture of trust and respect through open communication and transparency.",
        "Supporting work-from-home initiatives to promote flexibility and work-life balance.",
        "Providing opportunities for professional certifications to enhance skills and credentials.",
        "Offering access to employee resource groups for networking and support.",
        "Demonstrating a commitment to employee empowerment through delegation and autonomy.",
        "Supporting a culture of accountability by setting clear expectations and goals.",
        "Providing opportunities for career coaching and guidance.",
        "Demonstrating a commitment to work-life integration through flexible scheduling options.",
        "Offering opportunities for personal growth and development through mentorship and training.",
        "Providing access to flexible spending accounts to support employee financial wellness.",
        "Demonstrating a commitment to employee well-being through wellness programs and initiatives.",
        "Supporting a culture of recognition and appreciation through formal and informal channels.",
        "Providing access to professional development workshops and seminars.",
        "Demonstrating flexibility in accommodating family-related needs and responsibilities.",
        "Offering opportunities for leadership skill development through training and mentorship.",
        "Providing access to wellness programs and resources to support physical and mental health.",
        "Demonstrating a commitment to work-life flexibility through policy and practice.",
        "Supporting a culture of continuous feedback and improvement through regular evaluations.",
        "Offering opportunities for sabbaticals and extended leave to support work-life balance.",
        "Providing access to educational reimbursement programs to support ongoing learning.",
        "Demonstrating a commitment to diversity training and awareness.",
        "Supporting a culture of work-life balance through policies and initiatives.",
        "Offering opportunities for remote work to accommodate diverse needs and preferences.",
        "Providing access to employee recognition programs to celebrate achievements.",
        "Demonstrating flexibility in scheduling and time-off policies to accommodate individual needs.",
        "Supporting a culture of work-life harmony by promoting balance and well-being."
    ]

    # Create a pandas Series of staff responses to the question "what could we do better?"

    staff_improvement_responses = [
        "Improving communication regarding changes in policies or procedures.",
        "Providing more comprehensive training programs to address skill gaps.",
        "Creating a more inclusive work environment that values diverse perspectives.",
        "Implementing clearer expectations and goals for performance evaluation.",
        "Offering more flexibility in work schedules to accommodate personal needs.",
        "Enhancing recognition programs to acknowledge employee contributions more frequently.",
        "Increasing transparency in decision-making processes.",
        "Investing in better resources and tools to streamline workflows.",
        "Improving work-life balance initiatives to reduce stress and burnout.",
        "Fostering a stronger sense of teamwork and collaboration among departments.",
        "Providing more opportunities for professional development and advancement.",
        "Addressing issues related to workplace culture and morale.",
        "Enhancing performance feedback mechanisms to facilitate growth.",
        "Offering more competitive compensation and benefits packages.",
        "Providing better support for remote work arrangements.",
        "Improving communication regarding changes in policies or procedures.",
        "Providing more comprehensive training programs to address skill gaps.",
        "Creating a more inclusive work environment that values diverse perspectives.",
        "Implementing clearer expectations and goals for performance evaluation.",
        "Offering more flexibility in work schedules to accommodate personal needs.",
        "Enhancing recognition programs to acknowledge employee contributions more frequently.",
        "Increasing transparency in decision-making processes.",
        "Investing in better resources and tools to streamline workflows.",
        "Improving work-life balance initiatives to reduce stress and burnout.",
        "Fostering a stronger sense of teamwork and collaboration among departments.",
        "Providing more opportunities for professional development and advancement.",
        "Addressing issues related to workplace culture and morale.",
        "Enhancing performance feedback mechanisms to facilitate growth.",
        "Offering more competitive compensation and benefits packages.",
        "Providing better support for remote work arrangements.",
        "Addressing concerns related to workload and workload distribution.",
        "Improving the onboarding process for new employees.",
        "Enhancing communication between management and staff.",
        "Implementing more effective conflict resolution strategies.",
        "Investing in technology upgrades to improve efficiency.",
        "Offering more opportunities for employee involvement in decision-making.",
        "Addressing issues related to workplace diversity and inclusion.",
        "Improving the effectiveness of meetings and reducing unnecessary meetings.",
        "Providing better access to resources for professional development.",
        "Addressing concerns related to work-life balance and stress management.",
        "Enhancing employee recognition programs to be more meaningful.",
        "Improving feedback mechanisms for performance evaluations.",
        "Offering more support for work-from-home arrangements.",
        "Providing better access to mental health resources and support.",
        "Addressing concerns related to workload and work-life balance.",
        "Enhancing communication channels between different departments.",
        "Improving the efficiency of administrative processes.",
        "Offering more opportunities for cross-training and skill development.",
        "Addressing issues related to workplace diversity and inclusion.",
        "Investing in better equipment and technology for employees.",
        "Improving communication about changes in company strategy.",
        "Offering more opportunities for flexible work arrangements.",
        "Addressing concerns related to career advancement and growth opportunities.",
        "Enhancing feedback mechanisms for employee suggestions and ideas.",
        "Improving communication about company goals and objectives.",
        "Offering more opportunities for team-building and social events.",
        "Addressing concerns related to work-life balance and burnout.",
        "Enhancing the performance evaluation process to be more objective.",
        "Improving communication between different levels of management.",
        "Offering more opportunities for learning and development.",
        "Addressing concerns related to workload and time management.",
        "Enhancing communication about changes in company policies.",
        "Improving the accessibility of company resources for all employees.",
        "Offering more opportunities for career counseling and guidance.",
        "Addressing concerns related to employee morale and motivation.",
        "Enhancing communication about company initiatives and projects.",
        "Improving the efficiency of team meetings and communication.",
        "Offering more opportunities for cross-departmental collaboration.",
        "Addressing concerns related to workplace stress and mental health.",
        "Enhancing communication about changes in work processes.",
        "Improving support for remote work arrangements.",
        "Offering more opportunities for recognition and appreciation.",
        "Addressing concerns related to workload distribution.",
        "Enhancing communication about company performance and goals.",
        "Improving transparency in decision-making processes.",
        "Offering more opportunities for professional networking.",
        "Addressing concerns related to communication between teams.",
        "Enhancing communication about company structure.",
        "Improving support for work-life balance initiatives.",
        "Offering more opportunities for leadership development.",
        "Addressing concerns related to employee engagement and motivation.",
        "Enhancing communication about company policies and procedures.",
        "Improving access to information and resources for all employees.",
        "Offering more opportunities for feedback and input from employees.",
        "Addressing concerns related to workload and time management.",
        "Enhancing communication about changes in work assignments.",
        "Improving support for work-from-home arrangements.",
        "Offering more opportunities for employee recognition and rewards.",
        "Addressing concerns related to career growth and advancement.",
        "Enhancing communication about company performance and objectives.",
        "Improving support for remote work technology and infrastructure.",
        "Offering more opportunities for skill development and training.",
        "Addressing concerns related to workplace culture and morale.",
        "Enhancing communication about changes in company strategy.",
        "Improving support for employee mental health and well-being.",
        "Offering more opportunities for career counseling and guidance.",
        "Addressing concerns related to employee workload and stress levels.",
        "Enhancing communication about company values and mission.",
        "Improving support for employee work-life balance.",
        "Offering more opportunities for professional networking and development."
    ]

    #convert to pd series
    staff_doing_well = pd.Series(staff_responses_doing_well)
    staff_improvement_series = pd.Series(staff_improvement_responses)

    dict_made_up_data = {'doing_well': staff_doing_well, 'could do better': staff_improvement_series}
    df_dummy_data = pd.DataFrame(dict_made_up_data)

    return df_dummy_data

#------------------------------------------------------------
@st.cache_data(ttl=3600)
def create_df_strat_objs():
    dict_strat_objectives = dummy_strat_objectives()
    return dict_strat_objectives

#------------------------------------------------------------

def text_comparison_example_page(nlp):
    st.header('Testing Alignment with Strategy')

    #nlp = spacy.load("en_core_web_sm")

    apples = nlp("I like apples")
    oranges = nlp("I like oranges")
    apples_oranges = apples.similarity(oranges)
    oranges_apples = oranges.similarity(apples)

    st.subheader('Example of how this works:')
    with st.expander(label='Click for overview of method'):
        st.write('Imagine we have two texts, one that reads "I like apples" and one that reads "I like oranges" (as below):')
        col1, col2 = st.columns(2)
        with col1:
            st.write(apples)
        with col2:
            st.write(oranges)
        st.write('We can use a python library to check how similar these texts are. This is done using \
            the SpaCy "en_core_web_sm" model, each word has 300 dimensions. \
            This means that each word is represented as a vector in a 300-dimensional space, \
            where each dimension captures certain semantic and syntactic features of the word. \
            These vectors are learned during the training of the model and can be used to \
            compute similarities between words, among other tasks, through techniques such as cosine similarity.')

        st.write('The library can then provide a numeric value for how similar the texts are, expressed from 0 to 1')

        st.write(f'In the case of the above two sentences, the score is :red[**{round(apples_oranges,4)}**]. The similarity scores \
            can be derived in both directions though these may not match :red[***exactly***] so it may be better to apply a threshold to check whether the similarity score \
            is greater than this threshold (in which case consider them similar) or not (in which case consider them not similar)')

        st.write(f"Applying a threshold would give us a True/False result, e.g. by applying a 0.8 threshold, the above \
            comparison would return :red[**{apples_oranges > 0.8}**]")


    st.subheader('Overview of dummy data set')
    with st.expander(label='Click for overview of df'):
        # Specify the number of visible rows
        num_rows = 10

        df = produce_staff_survey_dummy_data()
        st.dataframe(df, height=30 * num_rows) 

    st.subheader('Overview of strategy components used')
    #produce combined string for all objectives in each strat aspect
    dict_strat_objectives_not_concatenated, dict_strat_objectives = dummy_strat_objectives()

    with st.expander(label='Click for overview of the strategy components used so far'):
        for key in dict_strat_objectives.keys():
            string_to_write = ""
            string_to_write+=f":red[***{key}***]"
            string_to_write+=": "
            string_to_write+=dict_strat_objectives[key]

            st.write(string_to_write)

    #---------------------------------------------------

    st.subheader("Let's look at one component of the strategy...")
    with st.expander(label='Click for explanation of prep applied to the text'):
        st.write(':red[**Step 1:**] Clean the strategy source text, removing stop words, punctuation, and lemmatizing the words the remain (e.g. changing words like :red[***"providing"***] to :red[***"provide"***])')

        st.write(f"**original text**: {dict_strat_objectives['aspect2']}")

        cleaned_aspect1 = clean_text_spacy(dict_strat_objectives['aspect2'])
        st.write(f"**cleaned text**: {cleaned_aspect1}")

    #----------------------------------------------------

    st.subheader("So, let's look at 1 of the 2 questions in the dummy data (***'What did we do well?'***)")
    with st.expander(label='Visualising the output of the method'):
        survey_responses = preprocess_reviews(df, 'doing_well')['doing_well_cleaned']
        survey_docs = [nlp(response) for response in survey_responses]

        aspect_to_use = st.selectbox(label='Select the strategy aspect to use', options=['aspect1','aspect2','aspect3','aspect4'])

        test_df_results = compare_two_texts(survey_responses, dict_strat_objectives[aspect_to_use], aspect_to_use)

        threshold = st.slider(label='Set the threshold to use', min_value=0.01, max_value=0.99, value=0.8, step=0.01)

        # Check if each value in the DataFrame is greater than or equal to the threshold
        greater_than_threshold = test_df_results['similarity_score'] >= threshold

        # Add a new boolean column based on the result of the check
        test_df_results['Above_Threshold?'] = greater_than_threshold

        #count the number of similar reviews i.e those above threshold
        similar_reviews = test_df_results[test_df_results['Above_Threshold?'] == True]
        not_similar_reviews = test_df_results[test_df_results['Above_Threshold?'] == False]

        #feedback to user
        st.write(f"There were :red[**{similar_reviews.shape[0]}**] similar reviews out of the :red[**{test_df_results.shape[0]}**] reviews present in the data set using a :red[**{threshold}**] threshold.")

        st.dataframe(test_df_results, height=30 * num_rows) 
    #----------------------------------------------------------
    st.subheader("Potential next steps")
    with st.expander(label='Click for overview of next steps / considerations'):
        st.write("We could use the output from the above to subset the source data (all staff survey reviews) \
            into those the model estimates to be similar to the strategic aim/area (based on our threshold value), \
                and those that are not similar.")

        st.write(f"Subsequent qualitative analysis methods can be applied to these two data sets to attempt to identify themes/latent topics etc.")

        st.write("this prototype has been set to only look at 1 aspect of the strategy. Once approach confirmed \
            the method could be updated to loop over all 4 aspects of the strategy")

        #subset the source df


        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f':green[**Similar reviews**] (n={similar_reviews.shape[0]})')
            st.dataframe(similar_reviews, height=30 * num_rows) 
        with col2: 
            st.subheader(f':red[**Dis-similar reviews** (n={not_similar_reviews.shape[0]})]')
            st.dataframe(not_similar_reviews, height=30 * num_rows) 

#-----------------------------------------

def about(nlp):
    
    st.title('About text comparison')
    st.write("This resource has been worked up to facilitate comparing one piece of \
    text to another piece of text. The use-case this has been built around is \
    comparing the text from survey responses to the text of strategic objectives in an \
    organisation's strategy document, to determine how similar these two are, \
    and use this similarity score as a proxy metric for the extent the nature \
    of the strategy is known to survey respondents.")

    st.write('You can use the sub-menu on the left 👈🏻 to access either the demonstration page \
        or the tool functionality itself which you can use on your own data / use case.')

    st.title('Methods used:')
    st.write('The below provides a summary of the methods available in the resource.')
    
    st.subheader('Named Entity Recognition (NER)')
    with st.expander('Click for explanation:'):
        st.write("Named Entity Recognition (NER) is a natural language processing (NLP) \
            task that involves identifying and classifying named entities in text into \
            predefined categories such as person names, organizations, locations, dates, \
            quantities, and more. The goal of NER is to locate and classify entities \
            mentioned in unstructured text, enabling machines to understand the meaning \
            and context of the text.")

        test_sentence = "Apple is looking at buying U.K. startup for $1 billion"

        st.write(f"For example, imagine we have the text ***:red['{test_sentence}']***. A SpaCy model will \
            look at each entity (word) in the sentence, and attempt to identify the nature \
            of that entity. The output using the medium model is below:")
        
        doc = nlp(test_sentence)

        df_ner = create_summary_df_of_ner_outputs(doc)
        
        st.dataframe(df_ner)
        
        st.write("Now you try, use the box below to enter a sentence and see what the SpaCy medium model can detect.")
        user_text = st.text_input(label='Enter a sentence to run through the NER model')
        if len(user_text) != 0:
            doc = nlp(user_text)
            df_ner_user = create_summary_df_of_ner_outputs(doc)
            st.dataframe(df_ner_user)

#---------------------------------------------------

def create_summary_df_of_ner_outputs(doc):
    """
    Function to loop over every entity in the document (piece of text as an nlp object),
    identify any Named Entities present, and summarise these, including definitions of the NE's,
    in a df. 
    Returns the summary dataframe. 
    """
    #call the spacy_ner_dict function to create a dictionary of the labels to definitions
    dict_named_entity_labels = spacy_ner_dict()

    list_text = []
    list_label = []
    list_definition = []
    for ent in doc.ents:
        list_text.append(ent.text)
        list_label.append(ent.label_)
        list_definition.append(dict_named_entity_labels[ent.label_])
    df_ner = pd.DataFrame(data={'text': list_text, 'NER_label': list_label, 'Definition': list_definition})
    return df_ner

#---------------------------------------------------
@st.cache_data(ttl=3600)
def spacy_ner_dict():
    named_entity_labels = {
    'PERSON': 'People, including fictional.',
    'NORP': 'Nationalities or religious or political groups.',
    'FAC': 'Buildings, airports, highways, bridges, etc.',
    'ORG': 'Companies, agencies, institutions, etc.',
    'GPE': 'Countries, cities, states.',
    'LOC': 'Non-GPE locations, mountain ranges, bodies of water.',
    'PRODUCT': 'Objects, vehicles, foods, etc. (not services).',
    'EVENT': 'Named hurricanes, battles, wars, sports events, etc.',
    'WORK_OF_ART': 'Titles of books, songs, etc.',
    'LAW': 'Named documents made into laws.',
    'LANGUAGE': 'Any named language.',
    'DATE': 'Absolute or relative dates or periods.',
    'TIME': 'Times smaller than a day.',
    'PERCENT': 'Percentage, including "%".',
    'MONEY': 'Monetary values, including unit.',
    'QUANTITY': 'Measurements, as of weight or distance.',
    'ORDINAL': '"first", "second", etc.',
    'CARDINAL': 'Numerals that do not fall under another type.'
    }
    return named_entity_labels

#---------------------------------------------------