import pandas as pd
import numpy as np 
import streamlit as st
from streamlit_extras.switch_page_button import switch_page


st.image("Snaps_To_Stats.png", width=2000)
st.sidebar.title("Exploratory Analysis With Random Forest Model")

st.markdown(
    """
    <h1 style="text-align: center; border-bottom: 3px solid #4CAF50; padding-bottom: 10px;">
        NFL Insights Hub
    </h1>
    """,
    unsafe_allow_html=True
)

