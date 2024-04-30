import streamlit as st

import pandas as pd
from transformers import pipeline

#topic modelling
from nltk.corpus import stopwords

#import functions and test data
import functions.functions as func
import data.test_data as data

def run_and_render_overall_results():
        
    #--------------------------------------
    #extract required variables from session state to later run overall topic modelling
    survey_responses = st.session_state['review_column']
    stop_words = set(stopwords.words("english"))
    df = st.session_state['dict_processed_data']['all_data']
    
    #st.dataframe(df)

    st.title(':green[Combined Qualitative Analysis]')
    
    import functions.fft_stage_2_sentiment_analysis as sentiment_analysis
    
    #test prints to check keys in session state - delete in final
    #for key in st.session_state.keys():
    #    st.write(key)
    #st.write('')
    #st.write(st.session_state['dict_processed_data']['known_sentiment_no_demographics'].keys())
        
    #STEP 1 - RUN OVERVIEW FUNCTION
    #st.write(st.session_state['analysis_scenario'])
    dict_keys_sentiment_demographic_choices = sentiment_analysis.sentiment_analysis_page_overview() 

    #STEP 2- RUN TOPIC MODELLING FUNCTION
    #requires some parameters to be set up OUTSIDE of the function
    st.subheader('Topic Modelling')
    

    find_optimal_params = func.radio_button('Find optimum params (can take time!)?', list_options=['Yes', 'No'], horizontal_bool=True, default_index=1)
    
    if find_optimal_params == 'Yes':
    
        #--------------------------------------
        #Run hyper param search to ID optimum parameters to then 
        #inform user in param selection stage
        with st.spinner('Please wait...'):
            best_params, perplexity_df = func.find_optimal_lda_params(
                df, #data
                survey_responses,
                stop_words,
                list(range(1,11,1)), #num topics range
                ['tf-idf', 'count'], #vectorizer methods
                ["Yes", "No"], #remove punctuation values
                list(range(1,11,1)), #ngram ranges
            )

        st.write("The optimum parameter values identified are as follows:")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write(f"**no. of topics:** {best_params['num_topics']}")
        with col2:
            st.write(f"**Vectorizer:** {best_params['vectorizer_method']}")
        with col3:
            st.write(f"**remove punctuation?** {best_params['remove_punctuation']}")
        with col4:
            st.write(f"**max words:** {best_params['ngram_range']}")
        
        with st.expander(label='Click for ordered parameters (best to worst based on "perplexity" score)'):
            st.dataframe(perplexity_df.sort_values(by='perplexity', ascending=True))
            
        #--------------------------------------
    
    with st.expander(label='Click to change topic modelling parameters'):
        col1, col2 = st.columns(2)
        with col1:
            num_topics = st.slider(label='Number of topics to use', min_value=1, max_value=10, value=3, step=1)
        with col2:
            num_n_grams = st.slider(label='Max words to include in the topic analysis', min_value=1, max_value=10, value=2, step=1, help='This is the value used for the max number of n-grams when running the topic analysis code. Where a sentence reads "Project planning is key to delivery", an n-gram value of 2 would capture the phrases "Project planning", "planning is", "is key" etc. and attempt to identify phrases associated together.')
        col1, col2, col3 = st.columns(3)
        with col1:
            #determine whether or not to remove punctuation
            remove_punctuation = st.selectbox(label='Remove punctuation in the analysis?', options=['Yes','No'], index=0)

        with col2:
            wordcloud_colour_scheme = st.selectbox(
                label='Select the colour scheme for the word cloud', 
                options=['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
                            'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b',
                            'tab20c'],
                            index=8
                )
            
        with col3:
            vectorizer_method = st.radio(label='Select vectorizer method', options=['tf-idf', 'count'], horizontal=True)

    
    
    if st.session_state['analysis_scenario'] == 'known_sentiment_no_demographics' or st.session_state['analysis_scenario'] == 'known_sentiment_with_demographics':
        subset_positive_df = df[df[f"{st.session_state['sentiment_col']}_binary"] == 'Positive']
        subset_negative_df = df[df[f"{st.session_state['sentiment_col']}_binary"] == 'Negative']

    else: #no known sentiment, so refer to the column name created by sentiment analysis function
        subset_positive_df = df[df['Sentiment'] == 'Positive']
        subset_negative_df = df[df['Sentiment'] == 'Negative']

    list_dfs = [subset_positive_df, subset_negative_df]

    tab1, tab2 = st.tabs(['Positive', 'Negative'])

    for index_num in range(len(list_dfs)):
        feature_names, lda, dict_topic_to_df, joined_processed_docs_lemmatized = func.run_sentiment_topic_modelling_overall(
            vectorizer_method,
            num_n_grams,
            remove_punctuation,
            num_topics,
            list_dfs[index_num],
            survey_responses,
            stop_words
            )
        
        if index_num == 0:
            with tab1:
                for topic_idx, topic in enumerate(lda.components_):
                            top_words_idx = topic.argsort()[:-10 - 1:-1]
                            top_words = [feature_names[i] for i in top_words_idx]
                            st.write(f"**Topic {topic_idx + 1}:** {', '.join(top_words)}")
                with st.expander(label='Click to see associated responses'):
                    for topic_num in range(num_topics):
                        st.subheader(f"Topic {topic_num+1}")
                        st.dataframe(dict_topic_to_df[f"Topic {topic_num+1}"])

        else:
            with tab2:
                for topic_idx, topic in enumerate(lda.components_):
                        top_words_idx = topic.argsort()[:-10 - 1:-1]
                        top_words = [feature_names[i] for i in top_words_idx]
                        st.write(f"**Topic {topic_idx + 1}:** {', '.join(top_words)}")

                with st.expander(label='Click to see associated responses'):
                    for topic_num in range(num_topics):
                        st.subheader(f"Topic {topic_num+1}")
                        st.dataframe(dict_topic_to_df[f"Topic {topic_num+1}"])

    #for key in st.session_state.keys():
    #    st.write(key)

    #st.write(st.session_state['demographic_columns'])
    #st.write(st.session_state['dict_processed_data']['known_sentiment_with_demographics'])

    if st.session_state['analysis_scenario'] == 'known_sentiment_with_demographics':
        session_state_dict = st.session_state['dict_processed_data']['known_sentiment_with_demographics']
    elif st.session_state['analysis_scenario'] == 'unknown_sentiment_with_demographics':
        session_state_dict = st.session_state['dict_processed_data']['unknown_sentiment_with_demographics']
    else:
        pass
    st.write(session_state_dict)
    #st.write(st.session_state['analysis_scenario'])
    #for key in st.session_state.keys():
    #    st.write(key)

    #try: 
    dict_results_for_cat_variables = func.run_sentiment_topic_modelling_by_cat_var(
        vectorizer_method,
        num_n_grams,
        remove_punctuation,
        num_topics,
        session_state_dict,
        survey_responses,
        stop_words,
        st.session_state['demographic_columns'],
        st.session_state['analysis_scenario'],
        st.session_state['dict_processed_data']
        )

    #TODO: incorporate output from above demographic level topic modelling, contained in
    # variable called dict_results_for_cat_variables into the summary expander for each unique label
    # in function below. may need dict_results_for_cat_variables as an input argument. not sure. 

    st.session_state['dict_results_for_cat_variables'] = dict_results_for_cat_variables
    
    #except:
    #    pass

    #STEP 3 - RUN DEMOGRAPHIC BREAKDOWN
    if st.session_state['analysis_scenario'] == 'known_sentiment_with_demographics' or st.session_state['analysis_scenario'] ==  'unknown_sentiment_with_demographics':
        text_dataset = st.session_state['dict_processed_data']['all_data']

    #    sentiment_analysis.sentiment_analysis_page_demographic_group_variation(
    #        st.session_state['dict_results_for_cat_variables'],
    #        text_dataset,
    #        st.session_state['review_column'],
    #        st.session_state['demographic_columns'],
    #        num_topics)
    
    #STEP 3 - RUN DEMOGRAPHIC BREAKDOWN - ALTERNATIVE PRESENTATION
    #if st.session_state['analysis_scenario'] == 'known_sentiment_with_demographics' or st.session_state['analysis_scenario'] ==  'unknown_sentiment_with_demographics':
        
        #for key in st.session_state['dict_service_to_dict_processed_dfs']['Service A'].keys():
        #    st.write(key)

        sentiment_analysis.sentiment_analysis_page_demographic_group_variation_by_service(
            #dict_results_for_cat_variables_by_service[service],
            st.session_state['dict_results_for_cat_variables'],
            text_dataset,
            st.session_state['review_column'],
            st.session_state['demographic_columns'],
            num_topics)
    
    #except:# NameError:
    #    pass