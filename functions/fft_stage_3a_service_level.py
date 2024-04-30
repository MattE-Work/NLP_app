
import streamlit as st

import pandas as pd
from transformers import pipeline

#topic modelling
from nltk.corpus import stopwords

#import functions and test data
import functions.functions as func
import data.test_data as data

#import other pages
import functions.fft_stage_2_sentiment_analysis as sentiment_analysis

import spacy


@st.cache_data(ttl=1800)
def sentiment_analysis_page_overview():
    st.title(':green[Service Level Qualitative Analysis]')
    intro_text = """This page provides various qualitative analyses of the text you 
             have prepared in step 1. 
             This page is structured into two main sections:
             \n1. The results / findings for the chosen service overall, followed by 
             \n2. The results by catagorical feature selected in the prep stage."""
    st.write(intro_text)
    if st.session_state['analysis_scenario'] == 'unknown_sentiment_with_demographics' or st.session_state['analysis_scenario'] == 'unknown_sentiment_no_demographics':
        st.write( """ No known sentiment was provided in the data prep stage, as a 
                          result, sentiment analysis has been run to **estimate** the sentiment based 
                          on the text provided. Where a positive/negative breakdown is provided
                          on this page, this is based on the output of the sentiment analysis. """)
    st.write(f"**Scenario:** {' '.join(st.session_state['analysis_scenario'].split('_'))}")


@st.cache_data(ttl=1800)
def service_all_data_overview(
    service):
    #check the analysis scenario that was run and render the page accordingly
    st.title(f':orange[{service}]')
    if st.session_state['analysis_scenario'] == 'known_sentiment_with_demographics':
        #produce lists containing keys and nested keys for each scenario
        list_outer_keys_known_sentiment_with_demographics = ["dict_processed_data" ,"tabs" ,"cat_var_tabs" ,"analysis_scenario"]
        list_nested_keys_dict_processed_data = ['all_data', 'known_sentiment_with_demographics']
        list_nested_keys_known_sentiment_with_demographics = ['Positive', 'Negative']

        #populate dict of the above lists for scenario: known_sentiment_with_demographics
        dict_keys_known_sentiment_with_demographics = {
            'list_outer_keys_known_sentiment_with_demographics': list_outer_keys_known_sentiment_with_demographics,
            'list_nested_keys_dict_processed_data': list_nested_keys_dict_processed_data,
            'list_nested_keys_known_sentiment_with_demographics': list_nested_keys_known_sentiment_with_demographics
        }

        return dict_keys_known_sentiment_with_demographics
        #------------
        #check session state contents from processing page - only show when debug mode is True
        #if debug_mode == 'Yes':
        #    st.write('After data prep, the session state CURRENTLY includes:')
        #    st.write(list(st.session_state.keys())) 
        #    st.write('nested keys in dict_processed_data')
        #    st.write(list(st.session_state['dict_processed_data'].keys()))
            #st.dataframe(st.session_state['dict_processed_data']['all_data'])
        #    st.write('nested keys within dict_processed_data > known_sentiment_with_demographics')
        #    st.write(list(st.session_state['dict_processed_data']['known_sentiment_with_demographics'].keys()))
        #------------
    
    elif st.session_state['analysis_scenario'] == 'known_sentiment_no_demographics':
        list_outer_keys_known_sentiment_no_demographics = ["dict_processed_data" , "analysis_scenario"]
        list_nested_keys_dict_processed_data = ['all_data', 'known_sentiment_no_demographics']
        list_nested_keys_known_sentiment_no_demographics = ['Positive', 'Negative']
        #populate dict of the above lists for scenario: known_sentiment_with_demographics
        dict_keys_known_sentiment_no_demographics = {
            'list_outer_keys_known_sentiment_no_demographics': list_outer_keys_known_sentiment_no_demographics,
            'list_nested_keys_dict_processed_data': list_nested_keys_dict_processed_data,
            'list_nested_keys_known_sentiment_no_demographics': list_nested_keys_known_sentiment_no_demographics
        }
        return dict_keys_known_sentiment_no_demographics

    elif st.session_state['analysis_scenario'] == 'unknown_sentiment_no_demographics':
        list_outer_keys_no_sentiment_no_demographics = ["dict_processed_data" , "analysis_scenario"]
        list_nested_keys_dict_processed_data = ['all_data', 'unknown_sentiment_no_demographics']
        list_nested_keys_no_sentiment_no_demographics = ['Positive', 'Negative']
        #populate dict of the above lists for scenario: known_sentiment_with_demographics
        dict_keys_no_sentiment_no_demographics = {
            'list_outer_keys_no_sentiment_no_demographics': list_outer_keys_no_sentiment_no_demographics,
            'list_nested_keys_dict_processed_data': list_nested_keys_dict_processed_data,
            'list_nested_keys_no_sentiment_no_demographics': list_nested_keys_no_sentiment_no_demographics
        }
        return dict_keys_no_sentiment_no_demographics

    elif st.session_state['analysis_scenario'] == 'unknown_sentiment_with_demographics':
        list_outer_keys_no_sentiment_no_demographics = ["dict_processed_data" , "analysis_scenario"]
        list_nested_keys_dict_processed_data = ['all_data', 'unknown_sentiment_no_demographics']
        list_nested_keys_no_sentiment_no_demographics = ['Positive', 'Negative']
        #populate dict of the above lists for scenario: known_sentiment_with_demographics
        dict_keys_no_sentiment_no_demographics = {
            'list_outer_keys_no_sentiment_no_demographics': list_outer_keys_no_sentiment_no_demographics,
            'list_nested_keys_dict_processed_data': list_nested_keys_dict_processed_data,
            'list_nested_keys_no_sentiment_no_demographics': list_nested_keys_no_sentiment_no_demographics
        }
        return dict_keys_no_sentiment_no_demographics

#---------------------------------

@st.cache_data(ttl=1800)
def service_df_overview(
    text_dataset,
    review_col,
    analysis_scenario
    ):
    #first create a df for all processed data (which should ALWAYS be present)
    #text_dataset = st.session_state['dict_processed_data']['all_data']
    
    #get the column names to containing the text we want to analyse
    review_col_no_stopwords = f"{review_col}_no_stopwords"
    
    if analysis_scenario == 'known_sentiment_no_demographics' or analysis_scenario == 'known_sentiment_with_demographics':
        binary_sentiment_col = f'{st.session_state["sentiment_col"]}_binary'
    else:
        binary_sentiment_col = 'Sentiment' #the col name to use if sentiment analysis run, in the absence of known sentiment
    #------------------------------------------
    #Overview of the data set
    #------------------------------------------
    if analysis_scenario == 'known_sentiment_no_demographics' or analysis_scenario == 'known_sentiment_with_demographics':
        func.get_and_render_known_sentiment_overview('Data overview', text_dataset, review_col, review_col_no_stopwords, binary_sentiment_col)


#---------------------------------




def run_combined_analysis(
        service_subset_df, # Pass the service subset dataframe as an argument
        survey_responses_column, 
        demographic_columns,
       #service_col,
        vectorizer_method, 
        num_n_grams, 
        remove_punctuation, 
        num_topics, 
        stop_words):  
    
    # Check if the service subset dataframe is empty
    if service_subset_df.empty:
        st.write(f"Skipping analysis for service as the subset dataframe is empty.")
        return None  # Skip analysis for this service
    
    dict_service_results = {}

    nlp = spacy.load("en_core_web_sm")

    # Extract relevant documents from the service subset dataframe
    service_subset_documents = service_subset_df[survey_responses_column].tolist()

    # Apply the LDA function to the service subset
    service_feature_names, service_lda, service_dict_topic_to_df, service_joined_processed_docs_lemmatized = func.run_lda_topic_modelling(
        vectorizer_method=vectorizer_method,
        num_n_grams=num_n_grams,
        remove_punctuation=remove_punctuation,
        documents=service_subset_documents,
        num_topics=num_topics,
        df=service_subset_df,
        survey_responses=survey_responses_column,
        stop_words=stop_words,
        nlp=nlp
    )

    # Store the results for this service subset
    dict_temp = {}
    dict_temp['feature_names'] = service_feature_names
    dict_temp['lda_model'] = service_lda
    dict_temp['topic_dfs'] = service_dict_topic_to_df
    dict_temp['processed_docs'] = service_joined_processed_docs_lemmatized
    dict_service_results['all_combined_service_data'] = dict_temp

    # Now, run demographic analysis for each selected demographic column within the service subset
    dict_demographic_topic_modelling_results = {}
    
    temp_demographic_dict = {}
    for demographic_column in demographic_columns:
        # Identify unique values in the demographic column
        unique_values = service_subset_df[demographic_column].unique()
        

        temp_unique_value_dict = {}
        for value in unique_values:
            # Create a subset dataframe based on the demographic value
            subset_df = service_subset_df[service_subset_df[demographic_column] == value]

            # Extract relevant documents from the subset dataframe
            subset_documents = subset_df[survey_responses_column].tolist()

            #only run topic modelling if the subset documents has more than 1 row (consistently fails when just 1 record)
            if subset_df.shape[0] > 1:
                # Apply the LDA function to the subset
                feature_names, lda, dict_topic_to_df, joined_processed_docs_lemmatized = func.run_lda_topic_modelling(
                    vectorizer_method=vectorizer_method,
                    num_n_grams=num_n_grams,
                    remove_punctuation=remove_punctuation,
                    documents=subset_documents,
                    num_topics=num_topics,
                    df=subset_df,
                    survey_responses=survey_responses_column,
                    stop_words=stop_words,
                    nlp=nlp
                )

                # Store the results for this demographic subset within the service subset
                
                temp_unique_value_dict[value] = {
                    'feature_names': feature_names,
                    'lda_model': lda,
                    'topic_dfs': dict_topic_to_df,
                    'processed_docs': joined_processed_docs_lemmatized
                }
            else:
                temp_unique_value_dict[value] = 'Not run'
            temp_demographic_dict[demographic_column] = temp_unique_value_dict
    return temp_demographic_dict
