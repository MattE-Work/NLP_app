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


#---------------------------------------------
#<<< code starts >>>
#---------------------------------------------

#nlp = spacy.load("en_core_web_sm") #30mb -> least accurate
nlp = spacy.load("en_core_web_md") #120mb -> ?allegedly 94% accuracy - large model (~870mb!) would be more accurate, but slower = tradeoff

#app starts - set up page layout
st.set_page_config(initial_sidebar_state='collapsed', layout='wide')

menu = [
    'About this section', 
    'Example', 
    'Text comparison']

choice = st.sidebar.selectbox('Menu', menu, index=0)


# Specify the number of visible rows for dataframes
num_rows = 10
height_int = 30

#load nlp model

if choice == 'About this section':
    strat_func.about(nlp)

elif choice == 'Example':
    strat_func.text_comparison_example_page(nlp)
    
elif choice == 'Text comparison':
    st.title(':blue[Text Comparison Tools] 📚=📗❓')
    st.write('Use this page to utilise functionality to undertake text comparisons.')

    st.subheader('Select data files')
    #get file paths
    col1, col2 = st.columns(2)
    with col1:
        st.write(':blue[**Text A**]')
        reference_file_to_compare_filepath = st.file_uploader(label='Select the file with the text you want to compare to (E.g. Strategic Objectives)')

        #Get text A
        if reference_file_to_compare_filepath != None:
            df_ref_file_to_compare_against = pd.read_csv(reference_file_to_compare_filepath)
        else:
            with col1:
                st.write('**NOTE:** No file selected. Test data in use for Text A.')
            dict_strat_objectives_not_concatenated, dict_strat_objectives = strat_func.create_df_strat_objs()
            df_ref_file_to_compare_against = pd.DataFrame.from_dict(dict_strat_objectives_not_concatenated)
    
    with col2:
        st.write(':blue[**Text B**]')
        text_to_compare_with_ref_filepath = st.file_uploader(label='Select the file with the text to compare Text A to (E.g. survey responses)')
        
        #Get text B
        if text_to_compare_with_ref_filepath != None:
            df_survey_responses = pd.read_csv(text_to_compare_with_ref_filepath)
            if 'remove?' in df_survey_responses.columns:
                remove_flagged_records = st.selectbox(label='Remove flagged records?', options=['Yes', 'No'], index=1)
                if remove_flagged_records == 'Yes':
                    df_survey_responses = df_survey_responses[df_survey_responses['remove?'] != 'y'].reset_index().drop('index', axis=1)
        else:
            with col2:
                st.write('**NOTE:** No file selected. Test data in use for Text B.')
            df_survey_responses = strat_func.produce_staff_survey_dummy_data()

        #preview data sets (either selected files or the test / example data)

    with st.expander('Click to preview datasets in use'):
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(df_ref_file_to_compare_against, height=height_int * num_rows)
        with col2: 
            st.dataframe(df_survey_responses, height=height_int * num_rows)
    
    st.subheader('Set text comparison parameters')
    col1, col2 = st.columns(2)
    with col1:
        threshold = st.slider(label='Set the threshold to use', min_value=0.01, max_value=0.99, value=0.8, step=0.01, help='A higher threshold means only very similar texts will be considered a match')

    with col2:
        survey_responses_col_name = st.selectbox(label='Select the column from Text B you want to analyse', options=df_survey_responses.columns)
        
        #ensure the responses are interpreted as string
        df_survey_responses[survey_responses_col_name] = df_survey_responses[survey_responses_col_name].astype(str)
        
        #st.write(df_survey_responses.shape)

        df_survey_responses = df_survey_responses.dropna(subset=[survey_responses_col_name])
        #st.write('drop na')
        #st.write(df_survey_responses.shape)

    #button = st.button(label='Run analysis')

    #CONTROL FLOW - UNCOMMENT WHEN BUILT   
    #if button == False:
    #    st.stop()

    #subset Text B to just the column selected above
    df_survey_responses = pd.DataFrame(df_survey_responses[survey_responses_col_name])

    #user defines text pre-processing / cleaning parameters
    sent_analysis_remove_punc, sent_analysis_lemmatize, sent_analysis_remove_stopwords = strat_func.render_clean_text_user_inputs()

    aggregate_metric = strat_func.user_selection_of_summary_metric()

    #------------------------------------
    #------------------------------------


    explanation_sim_between_texts = ('This section runs SpaCy similarity analysis to compare the cleaned up Text A \
        with each line in Text B. This produces a similarity score for every survey response \
        for each aspect of the strategy. These scores are then summarised as the sum \
        of similarity scores, and the proportion of scores above the threshold, \
        for each strategy aspect, respectively. ')

    st.subheader('Analysing similarity between the texts', help=explanation_sim_between_texts)

    #get keys 
    dict_concat_strat_objectives = strat_func.concatenate_text_by_column(df_ref_file_to_compare_against)
    dict_cleaned_strat_objectives = {key: strat_func.clean_text_spacy_user_defined(value, sent_analysis_remove_punc, sent_analysis_lemmatize, sent_analysis_remove_stopwords) for key, value in dict_concat_strat_objectives.items()}
    
    sim_scores_tab_names = [
        'Semantic similarity limitations'        
    ]

    

    
    with st.expander(label='Click for similarity score outputs'):
        tab1, tab2, tab3, tab4 = st.tabs([
        'Semantic similarity limitations',
        'Classifying similarity alignment by distribution', 
        f'Results for: {survey_responses_col_name}_cleaned',
        f'Similarity scores for: {survey_responses_col_name}_cleaned'])

        with tab1:
            st.subheader('Context Sensitivity:')
            st.write('Semantic similarity scores may not fully capture the context or the nuanced differences between texts, especially if the texts use domain-specific language or have multiple meanings depending on the context.')
            
            st.subheader('False Positives/Negatives:')
            st.write('High similarity scores do not always imply meaningful alignment with strategic objectives, and low scores do not always indicate misalignment. The scores may sometimes overlook the deeper intent or specific details of the feedback.')
            
            st.subheader('Dependence on Text Quality:')
            st.write('The effectiveness of similarity analysis can be influenced by the quality of the input texts. Poorly structured sentences, spelling errors, or incorrect grammar can affect the accuracy of similarity scores.')

        #test prints
        #st.write(dict_concat_strat_objectives)
        #st.write(dict_cleaned_strat_objectives)
        cleaned_df = strat_func.preprocess_reviews_whole_df(df_survey_responses, sent_analysis_remove_punc, sent_analysis_lemmatize, sent_analysis_remove_stopwords)

        #calculate the similarity scores
        similarity_scores_dict, threshold_dict = strat_func.compare_each_question_to_each_strat_objective(dict_cleaned_strat_objectives, cleaned_df, nlp, threshold)
        
        #convert dicts to dfs
        df_similarity_scores = similarity_scores_dict[f'{survey_responses_col_name}_cleaned']
        df_threshold_classification = threshold_dict[f'{survey_responses_col_name}_cleaned']

        #create a df with categorized similarity alignment labels
        df_similarity_scores_categorized_alignment = strat_func.categorize_alignment_based_on_distribution(df_similarity_scores, survey_responses_col_name)
        
        with tab2:
            #st.dataframe(updated_df)
            #Chart the distribution of scores
            strat_func.plot_alignment_distribution(df_similarity_scores_categorized_alignment)       

        with tab3:
            st.dataframe(df_threshold_classification)
        
        with tab4:
            st.write("This score shows how closely the staff responses match our strategic goals. A higher score means a stronger alignment.")
            st.dataframe(df_similarity_scores)


    sum_scores_dict, proportion_matches_dict = strat_func.calculate_aggregate_scores(similarity_scores_dict, threshold)
    
    sim_scores_df = similarity_scores_dict[f"{survey_responses_col_name}_cleaned"]

    #test prints
    #st.write(similarity_scores_dict['doing_well_cleaned'])
    #st.write(sum_scores_dict)
    #st.write(proportion_matches_dict)
    
    #Render the plots
    col1, col2 = st.columns(2)
    with col1:
        strat_func.plot_similarity_metrics(sum_scores_dict, 'Sum of similarity scores')
    with col2:
        #strat_func.plot_similarity_metrics(proportion_matches_dict, 'Proportion above threshold')
        strat_func.plot_radar_chart(proportion_matches_dict, survey_responses_col_name)
    
    #----------------------------------------------
    #Sentiment analysis section
    #----------------------------------------------

    st.subheader('Sentiment analysis')
    st.write('Sentiment analysis is used to classify the response as positive or \
    negative based on the words used in the survey response.')

    #run sentiment analysis functions and get overall sentiment analysis df
    df_sentiment_results = strat_func.perform_sentiment_analysis(cleaned_df, f"{survey_responses_col_name}_cleaned")

    col1, col2 = st.columns(2)

    with col1:
        #plot the sentiment score distribution as a histogram
        sentiment_histogram = strat_func.plot_sentiment_histogram(df_sentiment_results)
        st.altair_chart(sentiment_histogram, use_container_width=True)

    with col2:
        #render sentiment score density plot - visualises sentiment scores as a continuous distribution
        strat_func.render_density_plot(df_sentiment_results, sentiment_column_name="Sentiment_score")

    #work up requirements and visualise heatmap
    
    #list comprehension to get the relevant column labels in the processed df with alignment labels
    target_columns = [f"{key}-Alignment" for key in dict_concat_strat_objectives.keys()]

    #concat sentiment analysis outputs with earlier similarity alignment df
    df_sentiment_results_by_strat_aspect = strat_func.concatenate_columns(df_sentiment_results, df_similarity_scores_categorized_alignment, target_columns)

    sentiment_column = 'Sentiment_score'      

    # Perform sentiment analysis by strategy aspect - squashed format
    #df_sentiment_by_aspect_squashed = strat_func.sentiment_analysis_by_strategy_aspect_compressed(df_sentiment_results_by_strat_aspect, target_columns, sentiment_column)
    df_sentiment_by_aspect_squashed = strat_func.sentiment_analysis_by_strategy_aspect_compressed(df_sentiment_results_by_strat_aspect, target_columns, sentiment_column, aggregate_metric)


    st.subheader('Average sentiment score by alignment classification')
    #visualise the output with heatmap
    heatmap_chart = strat_func.visualize_sentiment_heatmap_altair(df_sentiment_by_aspect_squashed, aggregate_metric)
    st.altair_chart(heatmap_chart, use_container_width=True)

    #----------------------------------------------
    #<<< Correlation Analysis section starts >>>
    #----------------------------------------------
    
    correlation_analysis_help_text = 'Correlation score above 1 indicates positive \
        correlation, less than 1 indicates negative correlation. The strength of \
        the association can be inferred by the absolute value, with values close \
        to zero indicating a weaker correlation and closer to -1 or 1 indicating \
        a stronger correlation.'

    st.subheader('Correlation analysis', help=correlation_analysis_help_text)
    st.write('Correlation analysis is used to determine the relationship between \
        two variables, quantifying the strength and direction of their association.')

    #get the df from earlier with similarity scores. 
    # Rename the col to avoid ValueError when calling concat function.    
    # Then drop the renamed column as not required. 
    df_similarity_scores = similarity_scores_dict[f'{survey_responses_col_name}_cleaned']
    df_similarity_scores_to_combine = df_similarity_scores.drop(f'{survey_responses_col_name}_cleaned', axis=1)
    #df_similarity_scores.rename(columns={f'{survey_responses_col_name}_cleaned': f'{survey_responses_col_name}_cleaned_'}, inplace=True)
    #df_similarity_scores.drop(f'{survey_responses_col_name}_cleaned_', axis=1, inplace=True)
    
    #create correlation for use in the correlation analsis stage
    df_correlation_analysis = strat_func.append_df_to_df(df_sentiment_results, df_similarity_scores_to_combine)
        
    #list comprehension to get the relevant column labels in the format required for our concat df for corr analysis
    #target_columns_corr_analysis = [col_name.split('-')[0] for col_name in target_columns]
    
    target_columns_corr_analysis = [f"{key}-similarity" for key in dict_concat_strat_objectives.keys()]

    dict_correlation_analysis_results = {}
    for col_name in target_columns_corr_analysis:
        score_columns = [sentiment_column, col_name]
        correlation_matrix = strat_func.run_correlation_analysis(df_correlation_analysis, score_columns, method='spearman')        
        dict_correlation_analysis_results[col_name] = correlation_matrix

    #test combined viz
    combined_heatmap_chart = strat_func.visualize_combined_correlation_heatmap(
        dict_correlation_analysis_results,
        sentiment_column,
        title="Combined Correlation Heatmap")
    st.altair_chart(combined_heatmap_chart, use_container_width=True)

    #check for significance in correlation
    all_results_df, filtered_results_df, sig_results_present = strat_func.run_correlation_analysis_with_significance_and_filter(
        df_correlation_analysis,
        target_columns_corr_analysis,
        sentiment_column
    )

    if sig_results_present == True:
        st.write('The following table summarises statistically significant correlations present')
        st.dataframe(filtered_results_df)
    else:
        st.write('No statistical difference(s) detected between sentiment scores and the comparison text')

    #run correlation analysis
    #correlation_matrix = strat_func.run_correlation_analysis(df_correlation_analysis, score_columns, method='spearman')

    #----------------------------------------------
    #----------------------------------------------

    st.subheader('PCA and K-Means')
    st.write('Now we have derived similarity scores per review text, we can \
    used these to attempt clustering the reviews. This *might* yield some insight \
        as to the nature of the reviews by strategy alignment.')

    #Create a version of the sim score df to only include the numerical values (similarity scores) and not the cleaned up source text
    #Because KMeans only runs on numeric data. Once clustered, we can rejoined the cluster results to the source text, for context. 
    df_sim_scores_only = sim_scores_df.iloc[:,1:]

    #get number of dimensions present
    num_dimensions_in_data = df_sim_scores_only.shape[1]
    undertake_pca = num_dimensions_in_data > 2

    #check if we have more than 2 dimensions in the dataset (in the use-case this was built for, we had 4 dimensions of a strategy
    # in order to render subsequent clustering outputs, PCA was undertaken, to enable rendering in 2 dimensions like X / Y scatter chart for example)
    st.write('***Principal Component Analysis (PCA)***')
    
    with st.expander(label='Click for PCA outputs'):
        if undertake_pca == False:
            st.write(f'We have {num_dimensions_in_data} dimensions in the data, as such PCA is not required.')
        else:
            #undertake PCA as there are >2 dimensions in our dataset
            st.write(f'PCA has been undertaken because there are currently {num_dimensions_in_data} \
            dimensions in the data and we want to reduce this to 2 dimensions whilst \
            retaining the majority of information in the source data.')

            #call PCA function to derive 2 principal components        
            pca_data, pca_model, explanation = strat_func.perform_pca(df_sim_scores_only)
            #provide feedback to the user on how effective PCA has been at maintaining the information from the source dataset
            st.write(explanation)

            df_pca_data = pd.DataFrame(pca_data)
            #df_sim_scores_only = df_pca_data

    explanation_k_means = "K-means clustering is an unsupervised machine learning \
        algorithm used for partitioning a dataset into a predefined number of \
        clusters. The algorithm works by iteratively assigning data points to \
        the nearest cluster centroid and then updating the centroids based on the \
        mean of the points assigned to each cluster. This process continues until \
        convergence, where the centroids no longer change significantly, or until \
        a specified number of iterations is reached. \
        \
        \nIn the context of survey data comparison with strategy objectives, K-means \
        clustering can be useful for identifying groups of survey responses that exhibit \
        similar patterns or characteristics. By clustering survey responses based on \
        their similarity to the strategic objectives, you can gain insights into \
        how well the survey responses align with the objectives and identify any \
        distinct clusters of responses that may require further analysis or action."

    st.write('***K-Means Clustering***')
    

    if undertake_pca == True:
        df_for_k_means = df_pca_data
    else:
        df_for_k_means = df_sim_scores_only
    
    with st.expander(label = 'Click for k-means outputs'):
        
        kmeans_tabs = [
            'Explanation',
            'Silhouette plot',
            'Updated dataset after clustering',
            'Evaluation metrics'
        ]

        tab1, tab2, tab3, tab4 = st.tabs(kmeans_tabs)
        with tab1:
            st.write(explanation_k_means)
    

        
        with tab2:
            # Assuming 'data' is your input data (numpy array or pandas DataFrame)
            # You can use the find_optimal_k function to identify the optimal value of K
            optimal_k, silhouette_scores = strat_func.find_optimal_k(df_for_k_means)
            
            #update the user with the value for k that been derived
            st.write(f"The identified optimal value of K to use for this analysis is :red[***{optimal_k}***]. You can view the associated plot below.")
            
            # Plot silhouette scores for different values of K
            fig = strat_func.plot_silhouette_scores(silhouette_scores)
            st.write("In the silhouette score vs. number of clusters plot, the aim is \
            to identify the number of clusters that maximizes the silhouette score. \
            The silhouette score measures the compactness and separation of the clusters, \
            with values ranging from -1 to 1. Higher silhouette scores indicate \
            better-defined clusters, where data points are closer to their \
            own cluster and farther from other clusters.")

            st.pyplot(fig)

        
        #optional expander to view outputs from PCA -> K-Means
        with tab3:
            # Once you have the optimal value of K, you can use the run_kmeans function to perform K-means clustering
            labels, centroids, kmeans_object = strat_func.run_kmeans(df_for_k_means, optimal_k)
            
            #st.write("Cluster Labels:", labels)
            #st.write("Cluster Centroids:", centroids)

            #rebuild the df for subsequent plotting
            #first, get a df with just the text review data
            review_col = pd.DataFrame(sim_scores_df.iloc[:,0:1])

            #append to this the dimension data (which may be the PCA columns if that was run)
            review_col_with_dimension_data = strat_func.append_df_to_df(review_col, df_for_k_means)

            #append cluster labels column from kmeans above
            sim_scores_df_with_cluster_labels = strat_func.append_cluster_labels(review_col_with_dimension_data, labels)
            
            st.dataframe(sim_scores_df_with_cluster_labels)

        #optional expander to view evaluation metrics for K-Means
        with tab4:
            st.write('The below metrics can be used to evaluate the K-Means performance on the dataset.')
            
            #obtain inertia
            inertia_explained = "Inertia measures the within-cluster sum of squared distances \
            from each point to its assigned cluster centroid. Lower inertia values \
            indicate tighter clusters. You can use the inertia value to compare different \
            cluster solutions with varying numbers of clusters and select the optimal \
            number of clusters based on the 'elbow method', where the rate of decrease \
            in inertia slows down."

            #obtain silhouette score
            silhouette_score_explained = "The silhouette score measures how similar \
            an object is to its own cluster compared to other clusters. It ranges \
            from -1 to 1, where a higher silhouette score indicates better-defined \
            clusters. You can calculate the average silhouette score across all samples \
            to evaluate the overall quality of the clustering."

            inertia_metric, silhouette_score_metric = strat_func.get_k_means_performance_metrics(kmeans_object, df_for_k_means, labels)
            
            st.subheader("Inertia score:", help=inertia_explained)
            st.write(f"The clusters have an inertia score of :red[**{round(inertia_metric,3)}**]. Scores closer to zero indicate tighter clusters. \
                This can be visually sense-checked with the scatter plot below.")

            st.subheader("Silhouette score:", help=silhouette_score_explained)
            st.write(f"The model has a silhouette score of :red[**{round(silhouette_score_metric,3)}**]. \
                \nA score close to 1 indicates that the data points are well-clustered and far from neighboring clusters.\
                \nA score close to 0 indicates overlapping clusters. \
                \nA score close to -1 indicates that the data points are assigned to the wrong clusters.")

            fig = strat_func.plot_clustered_data(sim_scores_df_with_cluster_labels, survey_responses_col_name, centroids)
            st.subheader("Visualizing Clustered Data")
            st.plotly_chart(fig)

    #----------------------------------------
    #<<< Topic Modelling section starts >>>
    #----------------------------------------
    #First combine the output dataframes from all previous stages
    sim_scores_df_with_cluster_labels_subset = sim_scores_df_with_cluster_labels.iloc[:,[0,-1]]

    #update the first df in list below to include the original source text 
    #to make interpreting outliers and themes easier
    df_similarity_scores.insert(1, f"{survey_responses_col_name}_original", df_survey_responses[survey_responses_col_name])

    list_results_dfs = [
    df_similarity_scores,
    df_threshold_classification,
    df_similarity_scores_categorized_alignment,
    df_sentiment_results,
    sim_scores_df_with_cluster_labels_subset]

    # Drop the text column from all but the first dataframe to prevent duplicate columns
    for i in range(1, len(list_results_dfs)):
        list_results_dfs[i] = list_results_dfs[i].drop(list_results_dfs[i].columns[0], axis=1)

    # Concatenate all dataframes horizontally
    combined_df = pd.concat(list_results_dfs, axis=1)
    #st.dataframe(combined_df)
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
                'Sentiment Score',
                'Similarity Scores', 
                'Alignment Labels', 
                'Sentiment Classification', 
                'Cluster Label',
                'Comparison Text Aspect']
        

            selected_criteria = st.multiselect('Select criteria for filtering:', criteria_options)

            sentiment_score_range = None
            comparison_text_aspect_alignment = None
            selected_alignment_labels = None
            selected_sentiment_classes = None
            selected_cluster_labels_int = None
            comparison_text_aspect_sim_score = None
            similarity_score_range = None

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
            
            col1, col2 = st.columns(2)
            if 'Alignment Labels' in selected_criteria:
                with col1:
                    comparison_text_aspect_alignment = st.multiselect('Select Comparison Text Aspect for Alignment filter:', list(dict_concat_strat_objectives.keys()))
                    if comparison_text_aspect_alignment:
                        with col2: #Applies this common label to all aspects
                            alignment_labels = ['High Alignment', 'Moderate Alignment', 'Low Alignment']
                            selected_alignment_labels = st.multiselect('Select Alignment Labels:', alignment_labels)
            
            col1, col2 = st.columns(2)
            if 'Sentiment Classification' in selected_criteria:
                with col1:
                    sentiment_classes = ['Positive', 'Neutral', 'Negative']
                    selected_sentiment_classes = st.multiselect('Select Sentiment Classification:', sentiment_classes)

            col1, col2 = st.columns(2)
            if 'Cluster Label' in selected_criteria:
                with col1:
                    num_clusters = optimal_k  # This should be dynamically determined from your KMeans model, e.g., kmeans.n_clusters
                    cluster_labels = list(range(num_clusters))
                    selected_cluster_labels = st.multiselect('Select Cluster Label:', cluster_labels)
                    selected_cluster_labels_int = [int(cluster_label) for cluster_label in selected_cluster_labels]
            
            col1, col2 = st.columns(2)
            if 'Similarity Scores' in selected_criteria:
                with col1:
                    comparison_text_aspect_sim_score = st.multiselect('Select Comparison Text Aspect:', list(dict_concat_strat_objectives.keys()))
                if comparison_text_aspect_sim_score:
                    with col2:
                        #Applied this common range to all aspects
                        similarity_score_range = st.slider(
                            "Select a range of similarity scores:",
                            min_value=0.0,  # Adjust based on your score range
                            max_value=1.0,  # Adjust based on your score range
                            value=(0.0, 1.0)  # Default range
                )

        #preview filtered df
        with tab3:
            #apply filtering
            df_filtered = strat_func.apply_user_filters(
            combined_df,
            sentiment_column,
            sentiment_score_range, 
            comparison_text_aspect_alignment, 
            selected_alignment_labels, 
            selected_sentiment_classes, 
            selected_cluster_labels_int, 
            comparison_text_aspect_sim_score, 
            similarity_score_range, 
            dict_concat_strat_objectives)

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

        


