
#import libraries
import streamlit as st
import numpy as np
import pandas as pd

#import functions
import functions.functions as func

#import test data
import data.test_data as data

#debug mode toggle
debug_mode = True

#test print - confirmed working
#st.dataframe(df_dummy_review_data.head())

def data_prep():
    #testing_options, df_testing_options = func.get_all_testing_scenarios(['Yes', 'No'], ['Yes', 'No'], ['Some', 'None'])
    #st.write(testing_options)
    #st.dataframe(df_testing_options)

    st.title(':green[Data Preparation Page]')
    st.write('This page contains a series of tools to facilitate rapid qualitative analysis of textual data.')
    st.write('Potential data sources could be Friends and Family Test comments, incident reports, compliment, complaints, comments, staff survey feedback, departmental reviews, etc.')

    #------------------------------------------
    st.subheader('Select your dataset')
    #Select the data set containing text data

    text_dataset_file = st.file_uploader(
            'Please select a .csv file containing your text data', 
            type=['.csv'], 
            label_visibility="visible")

    num_rows = 5
    first_or_last = 'last'

    if text_dataset_file == None:
        st.write("**Note:** No file uploaded. Using made-up test data for illustrative purposes.")
        test_data_selection = st.selectbox('Select the test dataset to use', options=['Reviews in English only', 'Reviews in multiple languages'], index=0)
        if test_data_selection == 'Reviews in multiple languages':
            text_dataset = data.create_dummy_dataset()
            func.preview_df(text_dataset, f'Click to preview {first_or_last} {num_rows} rows of the test dataset', first_or_last, num_rows)
        elif test_data_selection == 'Reviews in English only':
            text_dataset = data.create_dummy_dataset_english_only()
            func.preview_df(text_dataset, f'Click to preview {first_or_last} {num_rows} rows of the test dataset', first_or_last, num_rows)
    else:
        text_dataset = pd.read_csv(text_dataset_file)
        func.preview_df(text_dataset, f'Click to preview {first_or_last} {num_rows} rows of your chosen dataset', first_or_last, num_rows)
    
    df_data_quality = (text_dataset.isnull().sum().to_frame()).rename(columns={0:'Count Missing'})
    df_data_quality['missing_%'] = round((df_data_quality['Count Missing'] / text_dataset.shape[0])*100,1)
    with st.expander(f'Click to view overview of missing data (total number records: {text_dataset.shape[0]})'):
        st.dataframe(df_data_quality)
        st.write()

    st.subheader('Set up parameters')
    st.write('***Sentiment params:***')
    #determine if known sentiment present (ground truth to compare predictions to later)
    #if so, ID the col name for this data
    sentiment_col = []

    col1, col2, col3, col4 = st.columns(4)
    with col1: 
        str_df_has_sentiment_truth = func.radio_button(
            'Does your data contain known sentiment?', 
            ['Yes', 'No'], 
            True,
            1)

        #update session state with str_df_has_sentiment_truth
        #st.session_state['str_df_has_sentiment_truth'] = str_df_has_sentiment_truth

    #render additional params if there is a known sentiment in the dataset
    if str_df_has_sentiment_truth == 'Yes':
        #col1, col2, col3 = st.columns(3)
        with col2:
            sentiment_col = func.single_df_field_selector(
                text_dataset,
                'Select the field containing the sentiment labels',
                None
            )
            #update session state with the sentiment column
            #st.session_state['sentiment_col'] = sentiment_col
        
        with col3:
            positive_labels = func.multi_label_selector(
            text_dataset,
            sentiment_col,
            ':green[Select the positive labels]',
            None
            )
            #update session state with the labels
            #st.session_state['positive_labels'] = positive_labels
        with col4:
            negative_labels = func.multi_label_selector(
            text_dataset,
            sentiment_col,
            ':red[Select the negative labels]',
            None
            )
            #update session state with the labels
            #st.session_state['negative_labels'] = negative_labels


    st.write('***Analysis scope***')
    col1, col2, col3 = st.columns(3)
    with col1:
        str_df_has_service_label = func.radio_button(
                'Do you want to filter the data to run the analysis on just a subset of the records?',
                ['Yes', 'No'],
                True,
                1
            )
    
    list_services_present = []
    service_col = ''
    if str_df_has_service_label == 'Yes':
        with col2:
            service_col = func.single_df_field_selector(
                text_dataset,
                'Select the column you wish to filter on (E.g. "Service")?',
                None
            )
            #update session state with the sentiment column
            #st.session_state['service_col'] = service_col
            list_services_present = list(set(text_dataset[service_col]))
        
        with col3:
            filter_criteria = func.single_label_selector(
                text_dataset,
                service_col,
                f'Select the value to filter the data set by in the {service_col} column',
                None,
                placeholder_text='Choose option'
            )
            text_dataset_filtered = func.subset_df_for_specific_langage(
                text_dataset,
                service_col,
                filter_criteria
                )
            text_dataset = text_dataset_filtered.copy(deep=True)
    else:
        filter_criteria = '-- overall results --'
    #-----
    st.write('***Redacting records***')
    col1, col2, col3 = st.columns(3)
    with col1:
        has_redact_col = func.radio_button(
            'Does the dataset have a col to indicate rows to redact from publication?',
            list_options=['Yes', 'No'], 
            horizontal_bool=True,
            default_index=1)
        
        st.session_state['has_redact_col'] = has_redact_col

    if has_redact_col == 'Yes':
        with col2:
            redact_col = func.single_df_field_selector(text_dataset, 'Select the column redaction labels', help_string='This is the column in the data set containing the labels that can be used to remove rows from publication')
    
        with col3:
            redact_text = func.single_label_selector(
                text_dataset,
                redact_col,
                'Select the label indicating rows to remove from publication',
                f'This will remove all records containing this label in the redact column ({redact_col}) from outlier detection'
            )
        #update session state
        st.session_state['redact_col'] = redact_col
        st.session_state['redact_text'] = redact_text

    #positive_labels = []
    #negative_labels = []

    #else:
    #    st.session_state['sentiment_col'] = 'Sentiment'
    
    st.write('***Select the text to analyse***')
    #select the labels pertaining to positive or negative sentiment in the df
    review_column = []
    demographic_columns = []

    col1, col2 = st.columns(2)
    with col1:
        review_column = func.single_df_field_selector(
            text_dataset,
            'Select the field containing the text to analyse',
            None
        )
        
        #update session state with the review column
        #st.session_state['review_column'] = review_column

    with col2:
        demographic_columns = func.multi_df_field_selector(
            text_dataset,
            'Select demographic fields to analyse the text by',
            None
        )

        #update session state with the demographic column
        #st.session_state['demographic_columns'] = demographic_columns

    #if needed, add data selection here
        
    #--------------------------------------------------------
    #date range filter section to avoid memory challenge
    #currently commented out due to formatting inconsistencies in the source data. needs finishing. 

    #date_field = st.selectbox(label='Select the field containing the date the wait closed', options=text_dataset.columns)

    # Convert the 'datetime_column' to datetime type with explicit format
    #text_dataset['DateTime_column'] = pd.to_datetime(text_dataset[date_field], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    # Convert the 'date_column' back to string for comparison
    #text_dataset['DateTime_column_as_string'] = text_dataset['DateTime_column'].dt.strftime('%d/%m/%Y %H:%M:%S')
    #text_dataset['mismatched_dates'] = text_dataset['DateTime_column'] != text_dataset['DateTime_column_as_string']
    #errors = (text_dataset['DateTime_column'] != text_dataset['DateTime_column_as_string']).sum()
    #st.write(errors)
    #st.dataframe(text_dataset[text_dataset['mismatched_dates'] == True])

    #try:
    #    col1, col2, col3 = st.columns(3)
    #    with col1:
    #        date_field = st.selectbox(label='Select the field containing the date the wait closed', options=text_dataset.columns)

    #        #split the date field into two columns
    #        text_dataset[['date_column', 'time_column']] = text_dataset[date_field].str.split(' ', expand=True)

            #change date_field to date time type
    #        #text_dataset[date_field] = pd.to_datetime(text_dataset[date_field], errors='coerce')
    #        text_dataset[date_field] = np.datetime64(text_dataset[date_field])
            #text_dataset['date_column_DT'] = pd.to_datetime(text_dataset['date_column'], errors='coerce')

    #        subset_col_df = text_dataset[text_dataset[date_field, 'date_column_DT']]
    #        subset_col_df['dates_match'] = subset_col_df[date_field] == subset_col_df['date_column_DT']
    #        st.dataframe(subset_col_df)

    #        earliest_date = text_dataset['date_column_DT'].min().date()
    #        latest_date = text_dataset['date_column_DT'].max().date()

    #    with col2:
    #        start_date = np.datetime64(st.date_input(f"Select a **from** date between {earliest_date} and {latest_date}", value=None))
    #    with col3:
    #        end_date = np.datetime64(st.date_input(f"Select a **to** date between {earliest_date} and {latest_date}", value=None))

        
    #    df_filtered_date_range = text_dataset[
    #        ((text_dataset['date_column_DT'] >= start_date) & (text_dataset['date_column_DT'] <= end_date))  # Retain values within the specified range
    #    ]

    #except:
    #    st.write('The selected field name is not of an expected date type')
    #    st.stop()

    #text_dataset = df_filtered_date_range
    with st.expander(label=f'click to preview df with: {text_dataset.shape[0]} rows'):
        st.dataframe(text_dataset, height=250)
    #st.dataframe(df_filtered_date_range)
#--------------------------------------------------------

    
    translate_option = func.radio_button(
        'How do you want to handle non-English reviews?', 
        ["Translate to English and retain", "Don't translate and remove", "Don't translate and retain"], True, default_index=2)

    #summary of selections
    try:
        func.preview_param_selections(
            str_df_has_sentiment_truth,
            str_df_has_service_label,
            service_col,
            list_services_present,
            positive_labels,
            negative_labels,
            review_column,
            sentiment_col,
            demographic_columns,
            filter_criteria)
    except:
        func.preview_param_selections_no_sentiment(
        str_df_has_sentiment_truth,
        str_df_has_service_label,
        service_col,
        list_services_present,
        review_column,
        sentiment_col,
        demographic_columns,
        filter_criteria)

    #data prep finished - button to move to next stage
    #st.write(text_dataset.shape)
    button_data_prep = func.click_button('Confirm selections', 'Data_Prep', type_str='secondary')
    #st.write(text_dataset.shape)
    
    if button_data_prep:
        st.write("Parameters are set up! 👍🏻")
        #st.write(text_dataset[text_dataset[review_column].isna()].shape)

        #take copy of text_dataset to test how na's are being identified
        #text_dataset_inc_na_rows = text_dataset.copy(deep=True) #moved below after binary sentiment col added

        #st.dataframe(text_dataset[text_dataset[review_column].isna()])
        #text_dataset.dropna(subset=review_column, inplace=True) #removes Null values from the text in scope #moved below after binary sentiment col added
        #text_dataset = text_dataset[text_dataset[review_column] != 'Did not answer'] #Assumes all have this exact text in for a did not answer #moved below after binary sentiment col added
        
        #st.write(text_dataset.shape)

        #update the df with a _binary column, based on known sentiment labels provided above
        #text_dataset[f'{review_column}_binary'] = np.where((text_dataset[review_column] in positive_labels) | (text_dataset[review_column] in negative_labels), 'Positive', 'Negative')
        if str_df_has_sentiment_truth == 'Yes':
            text_dataset[f'{sentiment_col}_binary'] = np.where(
                text_dataset[sentiment_col].isin(positive_labels),'Positive',
                np.where(
                text_dataset[sentiment_col].isin(negative_labels), 'Negative', 'Other')
                )

 
            
        else:
            #no known sentiment, therefore run sentiment analysis
            #text_dataset = func.determine_sentiment()
            text_dataset = func.determine_sentiment(text_dataset, review_column, 0.05)
        
        #take copy of text_dataset to test how na's are being identified
        text_dataset_inc_na_rows = text_dataset.copy(deep=True)
        text_dataset.dropna(subset=review_column, inplace=True) #removes Null values from the text in scope
        text_dataset = text_dataset[text_dataset[review_column] != 'Did not answer'] #Assumes all have this exact text in for a did not answer

        st.write(f"The data set in use has {text_dataset.shape[0]} records.")

        #df_lang_checked, str_languages_present = func.check_what_languages_are_present(text_dataset, review_column)
        df_lang_checked, str_languages_present = func.check_what_languages_are_present_revised(text_dataset, review_column)
        st.write(str_languages_present)

        #update session state with the df with all langs present
        st.session_state['all_df_all_langs_present'] = df_lang_checked

        #st.write('preview of ORIGINAL df:')
        #st.dataframe(text_dataset)

        #st.write('')
        
        #translate if that option has been selected
        #test_dict_translation = func.create_translation_models_dict(text_dataset['language_code'])
        #st.write(test_dict_translation)


        if translate_option == "Translate to English and retain":
            #list_translated_reviews, text_dataset = func.translate_text(text_dataset, review_column, src_lang='auto', target_lang='en')
            #text_dataset = func.translate_text_revised(text_dataset, review_column) #working, but, uses googletrans and alternative method required
            #text_dataset = func.translate_text_revised_argo(text_dataset, review_column) #working, but, uses googletrans and alternative method required
            df_lang_checked, str_languages_present = func.check_what_languages_are_present_revised(text_dataset, review_column)
            st.write(str_languages_present)

            #update session state with the df with all langs present
            st.session_state['all_df_all_langs_present'] = df_lang_checked

            summary_string, text_dataset, test_df_argo = func.translate_text(text_dataset, 'language_code', 'Review', 'language')
            st.write(summary_string)

        elif translate_option == "Don't translate and remove":
            st.write(':red[NOTE: at this time subsequent stages only used English reviews. Method for translation to be added here.]')
            #filters df to english lang only. Needs updating to handle non-English languages.
            #text_dataset_en = func.filter_df_to_label(text_dataset, 'language', 'English') # original working when removing non English text
            
            df_lang_checked, str_languages_present = func.check_what_languages_are_present_revised(text_dataset, review_column)
            st.write(str_languages_present)

            #update session state with the df with all langs present
            st.session_state['all_df_all_langs_present'] = df_lang_checked
                
            text_dataset = func.filter_df_to_label(text_dataset, 'language', 'English')
        
        elif translate_option == "Don't translate and retain":
            pass
            #st.write('Only English reviews retained')
            #st.write(f"This is the shape: {text_dataset.shape}")
    
    #--------------------------------------------------------------------
    #<<< Edit df to retain only english language reviews - need to adjust to handle non-English reviews >>>
    #--------------------------------------------------------------------

        #test stopword removal
        text_dataset_en_sw_removed = func.remove_stopwords_from_df_col(
            text_dataset, #text_dataset_en previously and working when removing non-English reviews
            review_column, 
            f"{review_column}_no_stopwords")

        #TODO update this line when tranlation incorporated - functions below to subset text data refer to df variable
        df = text_dataset_en_sw_removed.copy(deep=True)

    #--------------------------------------------------------------------
    #<<< update the session state >>>
    #--------------------------------------------------------------------
        # ----------------------------
        #update session state 
        # << remove keys if already present >>
        if 'str_df_has_sentiment_truth' in st.session_state.keys():
            del st.session_state['str_df_has_sentiment_truth']
        if 'str_df_has_service_label' in st.session_state.keys():
            del st.session_state['str_df_has_service_label']
        if 'service_col' in st.session_state.keys():
            del st.session_state['service_col']
        if 'list_services_present' in st.session_state.keys():
            del st.session_state['list_services_present']
        if 'sentiment_col' in st.session_state.keys():
            del st.session_state['sentiment_col']
        if 'positive_labels' in st.session_state.keys():
            del st.session_state['positive_labels']
        if 'negative_labels' in st.session_state.keys():
            del st.session_state['negative_labels']
        if 'review_column' in st.session_state.keys():
            del st.session_state['review_column']
        if 'demographic_columns' in st.session_state.keys():
            del st.session_state['demographic_columns']
        if 'dict_processed_data' in st.session_state.keys():
            del st.session_state['dict_processed_data']
        if 'analysis_scenario' in st.session_state.keys():
            del st.session_state['analysis_scenario']
        if 'dict_service_to_dict_processed_dfs' in st.session_state.keys():
            del st.session_state['dict_service_to_dict_processed_dfs']

        #<< update session state with new keys >>
        #update session state with str_df_has_sentiment_truth
        st.session_state['str_df_has_sentiment_truth'] = str_df_has_sentiment_truth
        #update session state with str_df_has_sentiment_truth
        st.session_state['str_df_has_service_label'] = str_df_has_service_label
        
        #update session state with the sentiment column
        if str_df_has_service_label == 'Yes':
            st.session_state['service_col'] = service_col
            st.session_state['list_services_present'] = list_services_present
            st.session_state['analysis_scope'] = filter_criteria
        else:
            st.session_state['analysis_scope'] = '-- overall results --'

        if str_df_has_sentiment_truth == 'Yes':
            st.session_state['sentiment_col'] = sentiment_col
            st.session_state['positive_labels'] = positive_labels
            st.session_state['negative_labels'] = negative_labels
        else:
            st.session_state['sentiment_col'] = 'Sentiment'
        st.session_state['review_column'] = review_column
        st.session_state['demographic_columns'] = demographic_columns

        # ----------------------------


    #--------------------------------------------------------------------
    #<<< Set up dataframes depending on selections made in prep stage >>>
    #--------------------------------------------------------------------

        dict_processed_dfs = {}
        dict_processed_dfs['all_data'] = df
        dict_processed_dfs['text_dataset_inc_na_rows'] = text_dataset_inc_na_rows

        #st.dataframe(df)

        if str_df_has_sentiment_truth == 'Yes' and len(demographic_columns) == 0:
            analysis_method = 'known_sentiment_no_demographics'
            dict_df_selections = func.get_dict_subset_df_to_known_sentiment(
                    df,
                    sentiment_col,
                    positive_labels, 
                    negative_labels)
            dict_processed_dfs[analysis_method] = dict_df_selections

        elif str_df_has_sentiment_truth == 'Yes' and len(demographic_columns) > 0:
            analysis_method = 'known_sentiment_with_demographics'

            dict_df_selections = func.get_dict_subset_df_to_known_sentiment_and_demographics(
                    df,
                    sentiment_col,
                    positive_labels, 
                    negative_labels,
                    demographic_columns,
                    analysis_method)
            dict_processed_dfs[analysis_method] = dict_df_selections

        elif str_df_has_sentiment_truth == 'No' and len(demographic_columns) == 0:
            analysis_method = 'unknown_sentiment_no_demographics'

            dict_df_selections = func.get_dict_subset_df_to_known_sentiment(
                    df,
                    'Sentiment',
                    positive_labels, 
                    negative_labels)
            dict_processed_dfs[analysis_method] = dict_df_selections

        elif str_df_has_sentiment_truth == 'No' and len(demographic_columns) > 0:
            analysis_method = 'unknown_sentiment_with_demographics'
            #dict_df_selections = func.get_dict_subset_df_to_unknown_sentiment_with_demographics(
            #df, 
            #demographic_columns)
            #dict_processed_dfs[analysis_method] = dict_df_selections

            dict_df_selections = func.get_dict_subset_df_to_known_sentiment_and_demographics(
                    df,
                    'Sentiment',
                    [], 
                    [],
                    demographic_columns,
                    analysis_method)
            dict_processed_dfs[analysis_method] = dict_df_selections

        else:
            st.write('unaccounted for combination - check the code')

        st.session_state['dict_processed_data'] = dict_processed_dfs
        st.session_state['analysis_scenario'] = analysis_method

        #----------------------------------------------------------------
        #if service labels present, repeat above, for service level dfs
        #----------------------------------------------------------------
        if str_df_has_service_label == 'Yes':
            st.session_state['service_col'] = service_col
            st.session_state['list_services_present'] = list_services_present

            #create
            
            dict_service_to_dict_processed_dfs = {}
            for service in list_services_present:
                
                #subset source data to just THIS service
                df_service = func.subset_df_for_specific_langage(df, service_col, service)
                
                #create dicts for the service 
                dict_processed_dfs = {}
                dict_processed_dfs['all_data'] = df_service
                
                #then, run above functions for this subset service level df

                if str_df_has_sentiment_truth == 'Yes' and len(demographic_columns) == 0:
                    analysis_method = 'known_sentiment_no_demographics'
                    dict_df_selections = func.get_dict_subset_df_to_known_sentiment(
                            df_service,
                            sentiment_col,
                            positive_labels, 
                            negative_labels)
                    dict_processed_dfs[analysis_method] = dict_df_selections

                elif str_df_has_sentiment_truth == 'Yes' and len(demographic_columns) > 0:
                    analysis_method = 'known_sentiment_with_demographics'

                    dict_df_selections = func.get_dict_subset_df_to_known_sentiment_and_demographics(
                            df_service,
                            sentiment_col,
                            positive_labels, 
                            negative_labels,
                            demographic_columns,
                            analysis_method)
                    dict_processed_dfs[analysis_method] = dict_df_selections

                elif str_df_has_sentiment_truth == 'No' and len(demographic_columns) == 0:
                    analysis_method = 'unknown_sentiment_no_demographics'

                    dict_df_selections = func.get_dict_subset_df_to_known_sentiment(
                            df_service,
                            'Sentiment',
                            positive_labels, 
                            negative_labels)
                    dict_processed_dfs[analysis_method] = dict_df_selections

                elif str_df_has_sentiment_truth == 'No' and len(demographic_columns) > 0:
                    analysis_method = 'unknown_sentiment_with_demographics'
                    #dict_df_selections = func.get_dict_subset_df_to_unknown_sentiment_with_demographics(
                    #df, 
                    #demographic_columns)
                    #dict_processed_dfs[analysis_method] = dict_df_selections

                    dict_df_selections = func.get_dict_subset_df_to_known_sentiment_and_demographics(
                            df_service,
                            'Sentiment',
                            positive_labels, 
                            negative_labels,
                            demographic_columns,
                            analysis_method)
                    dict_processed_dfs[analysis_method] = dict_df_selections

                else:
                    st.write('unaccounted for combination - check the code')

                dict_service_to_dict_processed_dfs[service] = dict_processed_dfs
            st.session_state['dict_service_to_dict_processed_dfs'] = dict_service_to_dict_processed_dfs
