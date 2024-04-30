import streamlit as st

import pandas as pd
from transformers import pipeline

#topic modelling
from nltk.corpus import stopwords

#import functions and test data
import functions.functions as func
import data.test_data as data


#app starts - set up page layout
st.set_page_config(initial_sidebar_state='collapsed', layout='wide')

menu = [
    'About this section', 
    'Step 1: Data Prep', 
    #'Step 2: Combined Analysis (optional)',
    #'Step 3: Service level analysis (optional)'
    #'Step 2: Analysis outputs',
    'Step 2: Analysis outputs']

choice = st.sidebar.selectbox('Menu', menu, index=0)

if choice == 'About this section':
    st.title(':green[Customer feedback]')
    #st.title(':green[Friends and Family]')
    st.write('''This section provides functionality to analyse :red[**labelled**] dataset 
             using techniques such as sentiment analysis, keyword extraction etc.''')
    
    st.subheader('How to use the tools on this page of the app:')
    st.write("""This section consists of 2 pages. The only core or 
             mandatory requirement is to run through the data prep section first (step 1), and 
             then review the outputs in the other section (step 2), which will be derived as applicable to your use-case. 
             \nYou can navigate to the 2 subpages using the 'Menu' drop 
             down to the left 👈🏻""")


elif choice == 'Step 1: Data Prep':
    import functions.fft_stage_1_data_prep as data_prep
    data_prep.data_prep()

elif choice == 'Step 2: Combined Analysis (optional)':
    if "analysis_scenario" not in st.session_state.keys():
        st.write("""⛔Oh no!⛔
                 \nLooks like you've not run the mandatory data prep page first. 
                 \nPlease:
                 \n1. Use the menu drop down to the left 👈🏻
                 \n2. Navigate to Step 1 and complete the set-up process 📋
                 \n3. Then come back here when you've done that 👍🏻""")
    else:
        
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
        
        #st.write(st.session_state['analysis_scenario'])
        #for key in st.session_state.keys():
        #    st.write(key)

        #test section
        st.title('session state demographic_columns:')
        st.write(st.session_state['demographic_columns'])
        
        st.title('session state analysis_scenario:')
        st.write(st.session_state['analysis_scenario'])
        
        st.title('session state dict_processed_data:')
        st.write(st.session_state['dict_processed_data'])


        try: 
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
        
        except:# NameError:
            pass


elif choice == 'Step 2: Analysis outputs':
    if 'str_df_has_service_label' not in st.session_state.keys():
        st.write("""⛔Oh no!⛔
                 \nLooks like you've either not run the mandatory data prep page yet. 
                 \nPlease:
                 \n1. Use the menu drop down to the left 👈🏻
                 \n2. Navigate to Step 1 and complete the set-up process 📋
                 \n3. Then come back here when you've done that 👍🏻""")
        
    else:
        import functions.fft_stage_3a_service_level as service_sentiment_analysis
        import functions.fft_stage_2_sentiment_analysis as sentiment_analysis
        import functions.fft_stage_3_overall_embed_service_page as render_overall_results
        
        #provide overview to page
        service_sentiment_analysis.sentiment_analysis_page_overview()
       
        #-------------------------------------------------
        dict_service_results = {}
        dict_service_sent_to_demo_results = {}

        topic_model_help_text = """This tool uses the LDA algorithm to discover 
            latent topics in a collection of text documents. LDA assumes that documents are mixtures of topics, 
            and topics are mixtures of words. It seeks to uncover the underlying semantic structure in the text. 
            \nThe output of LDA includes identified topics, each represented by a distribution of words. 
            Each document is associated with a distribution of topics. Outputs are to be 
            interpreted along with context/domain knowledge.
            \nStopwords (the, and, a etc.) have been removed. Punctuation removal is 
            optional for topic modelling here, but is included in the word cloud."""

        df = st.session_state['dict_processed_data']['all_data']

        survey_responses = st.session_state['review_column']
        stop_words = set(stopwords.words("english"))
        #-------------------------------------------------
        #set up topic modelling params expander section            
        st.subheader(':green[Set up Topic Modelling parameters]', help=topic_model_help_text)
        find_optimal_params = func.radio_button('Run all combinations to identify optimum parameters?', list_options=['Yes', 'No'], horizontal_bool=True, default_index=1, help_text='This will add run-time, but will run all combinations of all topic modelling parameters on the survey question in scope, and provide the optimum parameters. Note that this is undertaken on the overall data set (i.e. it is not undertaken on any demographic breakdown if demgoraphic variables have been entered).')
        
        if find_optimal_params == 'Yes':
            #stop_words = set(stopwords.words('english')) #22/1 commented out

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
        #-------------------------------------------------

        #STEP 1 - RUN OVERVIEW FUNCTIONS
        dict_keys_sentiment_demographic_choices = service_sentiment_analysis.service_all_data_overview(st.session_state['analysis_scope'])
        
        service_sentiment_analysis.service_df_overview(
            df,
            st.session_state['review_column'],
            st.session_state['analysis_scenario'])

        #STEP 2 - RUN TOPIC MODELLING 
        try:
            #survey_responses = st.session_state['review_column']
            #stop_words = set(stopwords.words("english"))
            
            if st.session_state['analysis_scenario'] == 'known_sentiment_no_demographics' or st.session_state['analysis_scenario'] == 'known_sentiment_with_demographics':
                subset_positive_df = df[df[f"{st.session_state['sentiment_col']}_binary"] == 'Positive']
                subset_negative_df = df[df[f"{st.session_state['sentiment_col']}_binary"] == 'Negative']

            else: #no known sentiment, so refer to the column name created by sentiment analysis function
                subset_positive_df = df[df['Sentiment'] == 'Positive']
                subset_negative_df = df[df['Sentiment'] == 'Negative']

            list_dfs = [subset_positive_df, subset_negative_df]

            tab1, tab2 = st.tabs(['Positive', 'Negative'])
            list_sentiment = ['Positive', 'Negative']
           
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
        except ValueError:
            st.write('Insufficient data to run analysis')

        #----------------------------
        #testing section list_sentiment
        #st.title(st.session_state['analysis_scenario'])

        
        dict_sent_to_demo_results = {}
        for index_num in range(len(list_dfs)):
            results_by_demographic = service_sentiment_analysis.run_combined_analysis(
            list_dfs[index_num], 
            st.session_state['review_column'], 
            st.session_state['demographic_columns'], 
            #st.session_state['service_col'],
            vectorizer_method, 
            num_n_grams, 
            remove_punctuation, 
            num_topics, 
            stop_words)

            dict_sent_to_demo_results[list_sentiment[index_num]] = results_by_demographic
        #dict_service_sent_to_demo_results[service] = dict_sent_to_demo_results
        dict_service_sent_to_demo_results[st.session_state['analysis_scope']] = dict_sent_to_demo_results

#
# TESTING SECTION!
#
        #st.title(f'{st.session_state["analysis_scope"]}')
        #st.write(dict_sent_to_demo_results)
        #st.write(
        #    dict_service_sent_to_demo_results[st.session_state["analysis_scope"]]
        #    ['Positive']
        #    ['Question 11: Are you male or female?']
        #    ['Male']
        #    ['feature_names'])
        #st.write(
        #    dict_service_sent_to_demo_results[st.session_state["analysis_scope"]]
        #    ['Positive']
        #    ['Question 11: Are you male or female?']
        #    ['Male'].keys())
        
        # Access results for a specific demographic column and unique value within a service subset
        # For example, to access results for 'Gender' = 'F' within 'Service' = 'ServiceA':
        #results_for_gender_F_service_A = results['Service']['ServiceA']['demographic_analysis']['Gender']['F']

        #----------------------------

        dict_results_for_cat_variables_by_service = {}
        if st.session_state['analysis_scenario'] == 'known_sentiment_with_demographics':
            session_state_dict = st.session_state['dict_processed_data']['known_sentiment_with_demographics']
        elif st.session_state['analysis_scenario'] == 'unknown_sentiment_with_demographics':
            session_state_dict = st.session_state['dict_processed_data']['unknown_sentiment_with_demographics']
        else:
            pass

        #-------------------------------------------------
        
        #STEP 3 - RUN DEMOGRAPHIC BREAKDOWN
        if st.session_state['analysis_scenario'] == 'known_sentiment_with_demographics' or st.session_state['analysis_scenario'] ==  'unknown_sentiment_with_demographics':
            st.title(':orange[**Variation by demographic group**]')

            #ODDS RATIO SECTION
            st.write('This sections seeks to derive insight as to variation in reported experience by the chosen demographic groups in scope')
            st.subheader('***Odds Ratio***')
            st.write("""An Odds Ratio calculation has been used below to determine 
                    whether there is any variation in the liklihood of reporting a 
                    positive experience by each demographic group in scope. 
                    Effectively, using ethnicity as an example, this shows the likelihood 
                    of reporting a positive experience by, say, patients with a White ethnicity, 
                    compared to all other ethnicities, and repeats this for all unique labels of 
                    each categorical / demographic variables selected on the prep page.""")
            

            #Repurpose the OR code from the waiting list equity app here
            dict_results_all_unique_labels = func.run_all_scenarios_odds_ratio(
                st.session_state['demographic_columns'],
                st.session_state['dict_processed_data']['text_dataset_inc_na_rows'],
                st.session_state['sentiment_col'],
                st.session_state['analysis_scenario'],
                z_critical = 1.96
                )

            df_or_demographic_results = pd.DataFrame(dict_results_all_unique_labels).T
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

            #TOPIC MODELLING SECTION
            st.subheader('***Demographic summaries***')
            st.write('The below expanders contain findings for each selected individual demographic. Where outlier comments for this group have been detected, these are included in addition to a Word cloud. Both are provided for positive and negative feedback, respectively. ')

            _df = st.session_state['dict_processed_data']['text_dataset_inc_na_rows']

            if st.session_state['analysis_scenario'] == 'known_sentiment_with_demographics' or st.session_state['analysis_scenario'] == 'known_sentiment_no_demographics':
                sentiment_col = f"{st.session_state['sentiment_col']}_binary"
                
            else:
                sentiment_col = f"{st.session_state['sentiment_col']}"
            #subset df to just positive reviews
            df_positive = _df[_df[sentiment_col] == 'Positive']

            #subset df to just negative reviews
            df_negative = _df[_df[sentiment_col] == 'Negative']

            #detect anomalies
            #pos_anomalies, pos_list_of_anomaly_text = func.detect_anomalies(df_positive, st.session_state['review_column'], contamination_param_threshold = 0.05)
            
            #create a dictionary of catagorical variable/column (key) and list of unique labels in that column (values)
            list_cat_var = st.session_state['demographic_columns']
            dict_cat_var_unique_labels = {}
            for cat_var in list_cat_var:
                dict_cat_var_unique_labels[cat_var] = list(set(_df[cat_var]))

            dict_unique_label_to_car_var = {}
            for key in dict_cat_var_unique_labels.keys():
                for unique_label in dict_cat_var_unique_labels[key]:
                    dict_unique_label_to_car_var[unique_label] = key

            #create dictionary of the subset df for each cat var / unique label combination
            dict_cat_var_unique_label_dfs = {}
            dict_cat_var_unique_label_dfs_positive = {}
            dict_cat_var_unique_label_dfs_negative = {}
            list_unique_labels = []
            for cat_var in dict_cat_var_unique_labels.keys():
                for unique_label in dict_cat_var_unique_labels[cat_var]:
                    if type(unique_label) == str:
                        list_unique_labels.append(unique_label)
                        temp_df = func.subset_df_for_specific_langage(_df, cat_var, unique_label)
                        dict_cat_var_unique_label_dfs[unique_label] = temp_df

                        temp_df_drop_na = temp_df.dropna(subset=[st.session_state['review_column']])
                        dict_cat_var_unique_label_dfs_positive[unique_label] = func.subset_df_for_specific_langage(temp_df_drop_na, sentiment_col, "Positive")
                        dict_cat_var_unique_label_dfs_negative[unique_label] = func.subset_df_for_specific_langage(temp_df_drop_na, sentiment_col, "Negative")
            
            #st.write(dict_cat_var_unique_label_dfs_positive['Male'][f"{st.session_state['sentiment_col']}_binary"].isna().sum())
            #st.write(dict_cat_var_unique_label_dfs_positive['Female'][f"{st.session_state['sentiment_col']}_binary"].isna().sum())

            #run anomaly detection for each unique label in each demographic catagorical field selected
            dict_unique_label_pos_anomalies = {}
            dict_unique_label_neg_anomalies = {}
            dict_unique_label_pos_word_cloud = {}
            dict_unique_label_neg_word_cloud = {}

            #dict_cat_var_unique_label_dfs_positive['Female'][dict_cat_var_unique_label_dfs_positive['Female']]

            for unique_label in list_unique_labels:
                #st.write(dict_cat_var_unique_label_dfs_positive[unique_label].shape)
                #attempt to ID pos anomalies
                if dict_cat_var_unique_label_dfs_positive[unique_label].shape[0] > 0:
                    pos_anomalies, pos_list_of_anomaly_text = func.detect_anomalies(dict_cat_var_unique_label_dfs_positive[unique_label], st.session_state['review_column'], contamination_param_threshold = 0.05)
                    dict_unique_label_pos_anomalies[unique_label] = [pos_anomalies, pos_list_of_anomaly_text]
                    #produce word cloud
                    dict_unique_label_pos_word_cloud[unique_label] = func.categorical_create_word_cloud(dict_cat_var_unique_label_dfs_positive[unique_label], st.session_state['review_column'], f'Wordcloud: {st.session_state["review_column"]} - {dict_unique_label_to_car_var[unique_label]} - {unique_label}', 'cool')

                else:
                    dict_unique_label_pos_anomalies[unique_label] = 'Insufficient data'
                
                
                #attempt to ID neg anomalies
                if dict_cat_var_unique_label_dfs_negative[unique_label].shape[0] > 0:
                    neg_anomalies, neg_list_of_anomaly_text = func.detect_anomalies(dict_cat_var_unique_label_dfs_negative[unique_label], st.session_state['review_column'], contamination_param_threshold = 0.05)
                    dict_unique_label_neg_anomalies[unique_label] = [neg_anomalies, neg_list_of_anomaly_text]
                    #produce word cloud
                    dict_unique_label_neg_word_cloud[unique_label] = func.categorical_create_word_cloud(dict_cat_var_unique_label_dfs_negative[unique_label], st.session_state['review_column'], f'Wordcloud: {st.session_state["review_column"]} - {dict_unique_label_to_car_var[unique_label]} - {unique_label}', 'hot')
                else:
                    dict_unique_label_neg_anomalies[unique_label] = 'Insufficient data'
            
            
            #run topic modelling and store outputs into dict for later rendering to user 
            dict_demographic_topic_modelling_outputs_pos = {}
            dict_unique_label_topc_model_ran_bool_pos = {}
            dict_demographic_topic_modelling_outputs_neg = {}
            dict_unique_label_topc_model_ran_bool_neg = {}

            #topic model positive dfs
            for unique_label in list_unique_labels:
                if dict_cat_var_unique_label_dfs_positive[unique_label].empty:
                    dict_unique_label_topc_model_ran_bool_pos[unique_label] = False
                else:
                    dict_unique_label_topc_model_ran_bool_pos[unique_label] = True
                    try:
                        subset_df = dict_cat_var_unique_label_dfs_positive[unique_label]
                        # Extract relevant documents from the subset dataframe
                        subset_documents = subset_df[st.session_state['review_column']].tolist()
                        # Apply the LDA function to the subset
                        feature_names, lda, dict_topic_to_df, joined_processed_docs_lemmatized = func.run_lda_topic_modelling_exc_nlp_param(
                        vectorizer_method=vectorizer_method,
                        num_n_grams=num_n_grams,
                        remove_punctuation=remove_punctuation,
                        documents=subset_documents,
                        num_topics=num_topics,
                        df=subset_df,
                        survey_responses=st.session_state['review_column'],
                        stop_words=stop_words,
                        ) 
                        
                        dict_model_outputs = {}
                        dict_model_outputs['feature_names'] = feature_names
                        dict_model_outputs['lda'] = lda
                        dict_model_outputs['dict_topic_to_df'] = dict_topic_to_df
                        dict_model_outputs['joined_processed_docs_lemmatized'] = joined_processed_docs_lemmatized

                        dict_demographic_topic_modelling_outputs_pos[unique_label] = dict_model_outputs
                    except:
                        dict_demographic_topic_modelling_outputs_pos[unique_label] = 'Topic modelling did not run.'
            #topic model negative dfs
            for unique_label in list_unique_labels:
                if dict_cat_var_unique_label_dfs_negative[unique_label].empty:
                    dict_unique_label_topc_model_ran_bool_neg[unique_label] = False
                else:
                    dict_unique_label_topc_model_ran_bool_neg[unique_label] = True
                    try:
                        subset_df = dict_cat_var_unique_label_dfs_negative[unique_label]
                        # Extract relevant documents from the subset dataframe
                        subset_documents = subset_df[st.session_state['review_column']].tolist()
                        # Apply the LDA function to the subset
                        feature_names, lda, dict_topic_to_df, joined_processed_docs_lemmatized = func.run_lda_topic_modelling_exc_nlp_param(
                        vectorizer_method=vectorizer_method,
                        num_n_grams=num_n_grams,
                        remove_punctuation=remove_punctuation,
                        documents=subset_documents,
                        num_topics=num_topics,
                        df=subset_df,
                        survey_responses=st.session_state['review_column'],
                        stop_words=stop_words,
                        ) 
                        
                        dict_model_outputs = {}
                        dict_model_outputs['feature_names'] = feature_names
                        dict_model_outputs['lda'] = lda
                        dict_model_outputs['dict_topic_to_df'] = dict_topic_to_df
                        dict_model_outputs['joined_processed_docs_lemmatized'] = joined_processed_docs_lemmatized

                        dict_demographic_topic_modelling_outputs_neg[unique_label] = dict_model_outputs
                    except:
                        dict_demographic_topic_modelling_outputs_neg[unique_label] = 'Topic modelling did not run'

            #add if statement to check if demographics present and only run below if so
            for cat_var in dict_cat_var_unique_labels.keys():
                for unique_label in dict_cat_var_unique_labels[cat_var]:
                    if type(unique_label) == str:
                        #set up the expander sections for each unique demographic present
                        with st.expander(label=f"{cat_var} - {unique_label}"):
                            tab1, tab2 = st.tabs(['Positive', 'Negative'])
                            with tab1: #positive
                                #word cloud
                                st.subheader('Wordcloud:')
                                st.pyplot(dict_unique_label_pos_word_cloud[unique_label])
                                
                                #topic modelling
                                st.subheader('Topic Modelling')
                                #outlier comments
                                
                                if dict_unique_label_topc_model_ran_bool_pos[unique_label] == False:
                                    st.write('Topic modelling was not run.')
                                else:
                                    if dict_demographic_topic_modelling_outputs_pos[unique_label] != 'Topic modelling did not run':
                                        #top X terms for each topic
                                        for topic_idx, topic in enumerate(dict_demographic_topic_modelling_outputs_pos[unique_label]['lda'].components_):
                                            top_words_idx = topic.argsort()[:-10 - 1:-1]
                                            top_words = [dict_demographic_topic_modelling_outputs_pos[unique_label]['feature_names'][i] for i in top_words_idx]
                                            st.write(f"**Topic {topic_idx + 1}:** {', '.join(top_words)}")
                                        
                                        #subset dfs aligned to each topic
                                        for key in dict_demographic_topic_modelling_outputs_pos[unique_label]['dict_topic_to_df'].keys():
                                            st.subheader(key)
                                            st.dataframe(dict_demographic_topic_modelling_outputs_pos[unique_label]['dict_topic_to_df'][key])
                                    else:
                                        st.write('Topic modelling did not run')

                            with tab2: #negative
                                #word cloud
                                st.subheader('Wordcloud:')
                                st.pyplot(dict_unique_label_neg_word_cloud[unique_label])
                                
                                #topic modelling
                                st.subheader('Topic Modelling')
                                #outlier comments

                                if dict_unique_label_topc_model_ran_bool_neg[unique_label] == False:
                                    st.write('Topic modelling was not run.')
                                else:
                                    if dict_demographic_topic_modelling_outputs_neg[unique_label] != 'Topic modelling did not run':
                                        #top X terms for each topic
                                        for topic_idx, topic in enumerate(dict_demographic_topic_modelling_outputs_neg[unique_label]['lda'].components_):
                                            top_words_idx = topic.argsort()[:-10 - 1:-1]
                                            top_words = [dict_demographic_topic_modelling_outputs_neg[unique_label]['feature_names'][i] for i in top_words_idx]
                                            st.write(f"**Topic {topic_idx + 1}:** {', '.join(top_words)}")
                                        
                                        #subset dfs aligned to each topic
                                        for key in dict_demographic_topic_modelling_outputs_neg[unique_label]['dict_topic_to_df'].keys():
                                            st.subheader(key)
                                            st.dataframe(dict_demographic_topic_modelling_outputs_neg[unique_label]['dict_topic_to_df'][key])
                                    else:
                                        st.write('Topic modelling did not run')
                
            #func.subset_df_for_specific_langage(df, lang_col, language)
            

            #function call below currently in use, but not working as expected
            #breakdown into many smaller functions to find solution
            #sentiment_analysis.sentiment_analysis_page_demographic_group_variation_by_service(
            #    dict_service_sent_to_demo_results[st.session_state['analysis_scope']],
            #    st.session_state['dict_processed_data']['text_dataset_inc_na_rows'],#8/1 new line to test
            #    st.session_state['review_column'],
            #    st.session_state['demographic_columns'],
            #    num_topics)


        #-------------------------------------------------
        #if branch ends
        #-------------------------------------------------


else:           
    pass