

import streamlit as st

import pandas as pd
from transformers import pipeline

#import functions and test data
import functions.functions as func
import data.test_data as data


#app starts - set up page layout
st.set_page_config(initial_sidebar_state='collapsed', layout='wide')

func.overview_to_app()

 