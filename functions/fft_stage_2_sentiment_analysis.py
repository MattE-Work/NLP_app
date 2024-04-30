
#import libraries
import streamlit as st
import pandas as pd
import numpy as np

#import functions
import functions.functions as func

#debug_mode = st.radio(label='Turn on debug mode?', options=['Yes', 'No'], horizontal=True, index=1)

#------------------------------------------------
#<<< Functions for step 2 - overall analyses >>>
#------------------------------------------------
#@st.cache_data(ttl=1800)
def sentiment_analysis_page_overview():
    #debug toggle

    #st.title(':green[Qualitative Analysis]')
    intro_text = """This page provides various qualitative analyses of the text you 
             have prepared in step 1. 
             This page is structured into two main sections. First, the results / findings 
             for the text overall, followed by a breakdown by the demographic 
             or other categorical features you selected in the prep stage."""
    st.write(intro_text)
    if st.session_state['analysis_scenario'] == 'unknown_sentiment_with_demographics' or st.session_state['analysis_scenario'] == 'unknown_sentiment_no_demographics':
        st.write( """ No known sentiment was provided in the data prep stage, as a 
                          result, sentiment analysis has been run to **estimate** the sentiment based 
                          on the text provided. Where a positive/negative breakdown is provided
                          on this page, this is based on the output of the sentiment analysis. """)
    
    
    #check the analysis scenario that was run and render the page accordingly
    st.write(f"**Scenario:** {' '.join(st.session_state['analysis_scenario'].split('_'))}")
    st.title(':green[Findings for the overall dataset combined]')
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

        #return dict_keys_known_sentiment_with_demographics
        dict_keys_sentiment_demographics = dict_keys_known_sentiment_with_demographics

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
        #return dict_keys_known_sentiment_no_demographics
        dict_keys_sentiment_demographics = dict_keys_known_sentiment_no_demographics

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

        #return dict_keys_no_sentiment_no_demographics
        dict_keys_sentiment_demographics =  dict_keys_no_sentiment_no_demographics

    elif st.session_state['analysis_scenario'] == 'unknown_sentiment_with_demographics':
        list_outer_keys_no_sentiment_no_demographics = ["dict_processed_data" , "analysis_scenario"]
        list_nested_keys_dict_processed_data = ['all_data', 'unknown_sentiment_no_demographics']
        list_nested_keys_no_sentiment_no_demographics = ['Positive', 'Negative']
        #populate dict of the above lists for scenario: known_sentiment_with_demographics
        dict_keys_no_sentiment_with_demographics = {
            'list_outer_keys_no_sentiment_no_demographics': list_outer_keys_no_sentiment_no_demographics,
            'list_nested_keys_dict_processed_data': list_nested_keys_dict_processed_data,
            'list_nested_keys_no_sentiment_no_demographics': list_nested_keys_no_sentiment_no_demographics
        }
        #return dict_keys_no_sentiment_no_demographics
        dict_keys_sentiment_demographics = dict_keys_no_sentiment_with_demographics

    #first create a df for all processed data (which should ALWAYS be present)
    text_dataset = st.session_state['dict_processed_data']['all_data']

    #get the column names to containing the text we want to analyse
    review_col = st.session_state['review_column']
    review_col_no_stopwords = f"{review_col}_no_stopwords"
    
    if st.session_state['analysis_scenario'] == 'known_sentiment_no_demographics' or st.session_state['analysis_scenario'] == 'known_sentiment_with_demographics':
        binary_sentiment_col = f'{st.session_state["sentiment_col"]}_binary'
    else:
        binary_sentiment_col = 'Sentiment' #the col name to use if sentiment analysis run, in the absence of known sentiment
    #------------------------------------------
    #Overview of the data set
    #------------------------------------------
    #if st.session_state['analysis_scenario'] == 'known_sentiment_no_demographics' or st.session_state['analysis_scenario'] == 'known_sentiment_with_demographics':
    func.get_and_render_known_sentiment_overview('Overview of our dataset', text_dataset, review_col, review_col_no_stopwords, binary_sentiment_col)
    
    return dict_keys_sentiment_demographics

    
#---------------------------------------------------------
#Run topic modelling for each demographic selected for this
#---------------------------------------------------------
@st.cache_data(ttl=1800)
def run_topic_modelling_for_each_cat_variable():
    pass


#---------------------------------------------------------
@st.cache_data(ttl=1800)
def sentiment_analysis_page_demographic_group_variation(
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
        st.title(':green[Variation by demographic group]') 
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
            if True in df_or_demographic_results['significant']:
                df_subset_sig = df_or_demographic_results[df_or_demographic_results['significant'] == True]
                st.write('Significant Odds Ratio findings are summarised below. To view *all* results, click on the expander below.')
            else:
                st.write('All Odds Ratios returned **non-significant** results. The results can be viewed in the expander below.')
            
            with st.expander(label='Click for all OR ratio results'):
                st.dataframe(df_or_demographic_results)
        else:
            st.write('Insufficient data, Odds Ratios not run.')

#---------------------------------------------------------
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
    
#------------------------------------------------
#<<< Functions for step 3 - service level analyses >>>
#------------------------------------------------

@st.cache_data(ttl=1800)
def service_level_sentiment_analysis_page_overview(
    serice,
    analysis_scenario,

):
    #debug toggle

    st.title(f'Qualitative Analysis: {serice}')
    intro_text = """This page provides various qualitative analyses of the text you 
             have prepared in step 1. 
             This page is structured into two main sections. First, the results / findings 
             for the text overall, followed by a breakdown by the demographic 
             or other categorical features you selected in the prep stage."""
    st.write(intro_text)
    if st.session_state['analysis_scenario'] == 'unknown_sentiment_with_demographics' or st.session_state['analysis_scenario'] == 'unknown_sentiment_no_demographics':
        st.write( """ No known sentiment was provided in the data prep stage, as a 
                          result, sentiment analysis has been run to **estimate** the sentiment based 
                          on the text provided. Where a positive/negative breakdown is provided
                          on this page, this is based on the output of the sentiment analysis. """)
    
    
    #check the analysis scenario that was run and render the page accordingly
    st.write(f"**Scenario:** {' '.join(st.session_state['analysis_scenario'].split('_'))}")
    st.title('Findings for the overall dataset combined')
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
