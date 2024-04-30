

#---------------------------------------
#IMPORT LIBRARIES!
#---------------------------------------
import string
from nltk.stem import WordNetLemmatizer
import streamlit as st
import spacy
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt

#PCA analysis libraries
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#kmeans clustering libraries
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from functions import strat_func


#---------------------------------------
#IMPORT FUNCTIONS!
#---------------------------------------
from functions import strat_func
from functions import survey_func

#---------------------------------------
#CODE STARTS!
#---------------------------------------

#nlp = spacy.load("en_core_web_sm") #30mb -> least accurate
nlp = spacy.load("en_core_web_md") #120mb -> ?allegedly 94% accuracy - large model (~870mb!) would be more accurate, but slower = tradeoff

#app starts - set up page layout
st.set_page_config(initial_sidebar_state='collapsed', layout='wide')

# Specify the number of visible rows for dataframes
num_rows = 10
height_int = 30

#test print - confirmed working
#st.dataframe(df_dummy_review_data.head())

col1, col2 = st.columns(2)
with col1:
    st.title(':orange[Survey Analysis DEV]')
#with col2:
    #debug mode toggle
    #debug_mode = st.radio(label='Turn on debug mode?', options=[True, False], horizontal=True, index=1)

st.write('This page facilitates the use of LDA topic modelling for rapid qualitative analysis of textual survey data. A high-level explanation of the LDA algorithm is available in the expander below.')

#------------------------------------------
#explanation of topic modelling
#------------------------------------------

with st.expander(label='Click for topic modelling overview'):
    st.write('This tool uses the LDA algorithm to discover latent topics in a collection of text documents. LDA assumes that documents are mixtures of topics, and topics are mixtures of words. It seeks to uncover the underlying semantic structure in the text. ')
    st.write('The output of LDA includes identified topics, each represented by a distribution of words. Each document is associated with a distribution of topics. Outputs are to be interpreted along with context/domain knowledge.')
    st.write('Stopwords (the, and, a etc.) have been removed. Punctuation removal is optional for topic modelling here, but is included in the word cloud.')

st.subheader('Set text comparison parameters')
with st.expander('Click to select file and question to analyse'):
    file_path = st.file_uploader(label='Select the survey results file to use')
    if file_path != None:
        df_survey_responses = pd.read_csv(file_path)
    else:
        #st.stop() #once finalised replace the line below with this st.stop() line. This will ensure data selection is the first stage.
        df_survey_responses = strat_func.produce_staff_survey_dummy_data()
        st.write(':red[**NOTE:** Demonstration data being used as no file was selected above ☝🏻]')
            
    survey_responses_col_name = st.selectbox(label='Select the column you want to analyse', options=df_survey_responses.columns)

    #drop nas
    df_survey_responses = df_survey_responses.dropna(subset=[survey_responses_col_name])
    df_survey_responses.reset_index(inplace=True)
    #ensure the responses are interpreted as string
    df_survey_responses[survey_responses_col_name] = df_survey_responses[survey_responses_col_name].astype(str)
    
    #for ease, create a subset df that has isolated just the response column we are interested in
    # with missing values removed
    df_survey_responses = pd.DataFrame(df_survey_responses[survey_responses_col_name])

    #user defines text pre-processing / cleaning parameters
    sent_analysis_remove_punc, sent_analysis_lemmatize, sent_analysis_remove_stopwords = strat_func.render_clean_text_user_inputs()

cleaned_df = strat_func.preprocess_reviews_whole_df(df_survey_responses, sent_analysis_remove_punc, sent_analysis_lemmatize, sent_analysis_remove_stopwords)

#cleaned_df = cleaned_df.dropna(subset=survey_responses_col_name)

#----------------------------------------------
#Sentiment analysis section
#----------------------------------------------
sentiment_column = 'Sentiment_score'  

st.subheader('Sentiment analysis')
with st.expander(label='Click for sentiment analysis outputs'):
    st.write('Sentiment analysis is used to classify the response as positive or \
    negative based on the words used in the survey response.')

    #run sentiment analysis functions and get overall sentiment analysis df
    df_sentiment_results = strat_func.perform_sentiment_analysis(cleaned_df, f"{survey_responses_col_name}_cleaned")

    #render visualisation of sentiment
    col1, col2 = st.columns(2)
    with col1:
        #plot the sentiment score distribution as a histogram
        sentiment_histogram = strat_func.plot_sentiment_histogram(df_sentiment_results)
        st.altair_chart(sentiment_histogram, use_container_width=True)

    with col2:
        #render sentiment score density plot - visualises sentiment scores as a continuous distribution
        strat_func.render_density_plot(df_sentiment_results, sentiment_column_name=sentiment_column)

#lazy approach to avoid rewriting code used elsewhere in the text comparison section
combined_df = df_sentiment_results.copy(deep=True)

#----------------------------------------
#<<< Topic Modelling section starts >>>
#----------------------------------------
#update the first df in list below to include the original source text 
#to make interpreting outliers and themes easier
combined_df.insert(1, f"{survey_responses_col_name}_original", df_survey_responses[survey_responses_col_name])

st.subheader("Topic Modelling")
st.write('Now we have clustered the data, we can undertake topic modelling to \
    determine whether there are any latent topics or themes within the text for \
        each cluster.')

with st.expander(label='Set up topic modelling parameters'):
    #set up tabs to save screen space for topic modelling set up
    topic_model_prep_tabs = [
        'Topic model parameters',
        'Filter criteria',
        'Preview filtered dataframe'
    ]

    tab1, tab2, tab3 = st.tabs(topic_model_prep_tabs)


    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            num_topics = st.slider(label='Number of topics to use', min_value=1, max_value=10, value=3, step=1)
        with col2:
            num_n_grams = st.slider(label='Number of n-grams to use:', min_value=1, max_value=10, value=2, step=1, help='This is the value used for the max number of n-grams when running the topic analysis code. Where a sentence reads "Project planning is key to delivery", an n-gram value of 2 would capture the phrases "Project planning", "planning is", "is key" etc. and attempt to identify phrases associated together.')
        with col3:
            upper_limit_max_words = strat_func.count_unique_words(combined_df, f"{survey_responses_col_name}_cleaned")
            max_words = st.slider(label='Max words (de-noise)', min_value=1, max_value=upper_limit_max_words+1, value=upper_limit_max_words+1, step=1, help='This pecifies the maximum number of most frequent words to be considered when vectorizing text. For example, set to 1000, after counting the frequency of all words in the corpus, the vectorizer will only keep the top 1000 most frequent words and disregard the rest. This can be helpful for reducing the dimensionality of the feature space and can also act as a form of noise reduction, especially when dealing with very large text data.')
        
        col1, col2 = st.columns(2)
        with col1:
            wordcloud_colour_scheme = st.selectbox(
                label='Select the colour scheme for the word cloud', 
                options=['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
                            'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b',
                            'tab20c'],
                            index=8
                )
            
        with col2:
            vectorizer_method = st.radio(label='Select vectorizer method', options=['tf-idf', 'count'], horizontal=True)
    
    with tab2:
        # Example of capturing general filtering criteria
        criteria_options = [
            'Sentiment Score']
    

        selected_criteria = st.multiselect('Select criteria for filtering:', criteria_options)

        sentiment_score_range = None
        col1, col2 = st.columns(2)
        # Dynamically generate additional options based on selected criteria
        if 'Sentiment Score' in selected_criteria:
            with col1:
                sentiment_score_range = st.slider(
                    "Select a range of sentiment scores:",
                    min_value=float(combined_df[sentiment_column].min()),
                    max_value=float(combined_df[sentiment_column].max()),
                    value=(float(combined_df[sentiment_column].min()), float(combined_df[sentiment_column].max()))
                )
    
    #preview filtered df
    with tab3:
        #apply filtering
        df_filtered = survey_func.apply_sentiment_filters(combined_df, sentiment_column, sentiment_score_range)
        st.write(survey_responses_col_name)
        #drop na rows
        df_filtered = df_filtered.dropna(subset = f"{survey_responses_col_name}_cleaned")
        #render preview
        st.write(f"The filtered dataframe has :red[**{df_filtered.shape[0]}**] rows.")
        st.write(df_filtered)

search_local_optima_params = st.selectbox(label='Search for local optimal parameters?', options=['Yes', 'No'], index=1)
if search_local_optima_params == 'Yes':
    best_params, best_score = strat_func.random_search_lda_params(df_filtered,  f"{survey_responses_col_name}_cleaned")
    st.write(f'The following were identified as locally optimal parameters with a score of :red[**{round(best_score, 3)}**]:')
    st.write(f"**Number of topics:** {best_params['num_topics']}")
    st.write(f"**Number of n-grams:** {best_params['ngram_range'][1]}")
    st.write(f"**Vectorizer method:** {best_params['vectorizer_method']}")
    st.write(f"**Max words:** {best_params['max_features']}")


#run topic modelling
with st.expander(label='Click to view topic modelling outputs'):
    topics_df, doc_topic_df, term_topic_weights_df, perplexity = strat_func.perform_topic_modeling_and_return_results(
        df_filtered, 
        f"{survey_responses_col_name}_cleaned",
        f"{survey_responses_col_name}_original",
        num_topics, 
        num_n_grams,
        max_words, 
        vectorizer_method
        )

    #convert topic terms output to dictionary
    # Initialize an empty dictionary to store topics and their terms
    dict_topics_terms = {}

    # Iterate through each column in topics_df to construct the comma-separated string of terms
    for column in topics_df.columns:
        # Join the terms in the column into a single string, separated by commas
        # We use .dropna() to ensure we don't include any NaN values in our string
        terms_string = ', '.join(topics_df[column].dropna())
        dict_topics_terms[column] = terms_string
    
    #produce word clouds
    dict_wordclouds = strat_func.generate_topic_wordclouds_by_dominant_topic(doc_topic_df, f"{survey_responses_col_name}_cleaned", wordcloud_colour_scheme)

    # Create tabs for each word cloud
    tab_titles = [f"Topic {i+1}" for i in range(len(list(topics_df.columns)))]
    tabs = st.tabs(tab_titles)

    #set up tabs for each topic in scope and populate with topic specific outputs
    for i, title in enumerate(tab_titles):
        with tabs[i]:
            st.subheader('Topic-terms identified:')
            dict_topics_terms[tab_titles[i]]

            st.subheader(f'Responses aligned to {tab_titles[i]}')
            temp_filtered_dom_topic = doc_topic_df[doc_topic_df['Dominant Topic'] == tab_titles[i]]
            st.dataframe(temp_filtered_dom_topic)
            
            if temp_filtered_dom_topic.shape[0] > 0:

                topic_prob_text = "term weights indicate how strongly a term \
                    is associated with a topic, serving as a measure of \
                        relevance of that term to the topic's composition"

                st.subheader('Visualising topic probabilities', help=topic_prob_text)
                strat_func.visualize_topic_probabilities(temp_filtered_dom_topic.reset_index(), tab_titles[i])
                
                st.subheader('Overall sentiment for this topic')
                chart_sentiment_topic = strat_func.plot_sentiment_histogram(temp_filtered_dom_topic)
                st.altair_chart(chart_sentiment_topic, use_container_width=True)

                st.subheader('Outliers')
                #identify the outliers
                least_associated, most_aligned = strat_func.detect_topic_outliers(temp_filtered_dom_topic, tab_titles[i], num_outliers=10)
                
                #display outliers
                col1, col2 = st.columns(2)
                with col1:
                    st.write('**Least associated:**')
                    st.dataframe(least_associated)
                    chart_least = strat_func.plot_sentiment_histogram(least_associated)
                    st.altair_chart(chart_least, use_container_width=True)

                with col2:
                    st.write('**Most associated:**')
                    st.dataframe(most_aligned)

                    chart_most = strat_func.plot_sentiment_histogram(most_aligned)
                    st.altair_chart(chart_most, use_container_width=True)
                
                st.subheader('Word cloud')
                st.pyplot(dict_wordclouds[title])
            
                term_weight_help_text = "Each cell in the heatmap corresponds to \
                    a term-topic pair, with the cell's color intensity representing \
                    the weight or significance of the term in that topic. A darker or \
                    more intense color typically means the term is more important or \
                    prevalent in characterizing the topic."

                st.subheader('Term-weight heatmap', help=term_weight_help_text)
                df_temp_topic_term_weight = term_topic_weights_df[term_topic_weights_df['Topic'] == tab_titles[i]]
                df_temp_topic_term_weight_sorted = df_temp_topic_term_weight.sort_values(by='Weight', ascending=False)
                
                #render the sorted term-weight heatmap
                strat_func.create_interactive_heatmap_indivudal_topic(df_temp_topic_term_weight_sorted, False)

#long_format_df = strat_func.convert_to_long_format(topics_df)
#st.write(long_format_df)
with st.expander(label = 'Combined outputs'):
    
    st.write(f"The model has a score of: :red[**{round(perplexity,3)}**]")
    #render term-weight distributions
    term_weight_distribution_help = "how concentrated the term weights \
        are around certain values, indicating how focused (coherent) \
        each topic is, and how distinct (separated) the topics \
        are from each other based on their term weight distributions."

    st.subheader('Term-weight distribution', help=term_weight_distribution_help)
    strat_func.visualize_term_weight_distributions_overlaid(term_topic_weights_df)

    combined_help_text = 'The combined heatmap allows side-by-side comparison of \
    topics, allowing users to quickly see which terms are important across \
    multiple topics and how topics differ in term composition. \
    \nSome terms may be significant across multiple topics (appearing as consistently \
    colored across different columns), indicating common themes or language. \
    Conversely, terms that are highly weighted in one topic but not in others \
    can signal terms unique to that topic, helping to distinguish it.'
    
    st.subheader('Summary of topic terms')
    st.write(topics_df)

    st.subheader('Combined term-weight heat map', help=combined_help_text)
    strat_func.create_interactive_heatmap(term_topic_weights_df)
