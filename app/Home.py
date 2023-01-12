# SWIZZLE - AI GENERATED MUSIC NOTATION FOR SONGS



# ----------import python packages----------
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import math
import time
import plotly.graph_objects as go

# ----------Page layout----------
st.set_page_config(layout = "wide")

# ----------import user packages----------
from preprocessing.funnel import Funnel
from preprocessing.prepro import PreProcessor

# ----------Setup state session in streamlit----------
if "page" not in st.session_state:
    st.session_state.page = 0
def nextpage(): st.session_state.page += 1
def restart(): st.session_state.page = 0
placeholder = st.empty()

# -----------------------------Page 1 (Home)-------------------------------
if st.session_state.page == 0:
    # Replace the content of page with several elements:
    with placeholder.container():
        
         # -----Create Sidebar------
        with st.sidebar:
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.subheader('About')
            st.markdown('Swizzle is a tool for musicians providing **AI generated music notation of songs**.')
            st.markdown("You can find the code on our [Github](https://github.com/swizzle-app/swizzle)")
        
        # -----Swizzle Logo------
        st.image('media/swizzle_logo.png', width=400,)
        st.markdown("---")
        
        left_column, right_column = st.columns(2)
         # -----Page layout - create 2 columns------
        with left_column:
            
            # -----Upload  WAV file------
            st.write('### Upload your WAV file')  
            file_uploader = st.file_uploader(label=" ", type=[".wav"]) #, ".wave", ".flac", ".mp3", ".ogg"])
            
            # -----Store variable with state session------
            st.session_state['file_uploader'] = file_uploader
            
            #-----Swizzle Button------
            st.write('')
            st.write('### Do the magic...') 
            st.write('') 
            if st.button("Swizzle it",):
                
                #-----Swizzle Spinner------
                with st.spinner('Swizzle it...'):
                    time.sleep(2)
                
                #-----Swizzle Spinner------
                latest_iteration = st.empty()
                bar = st.progress(0)
                for i in range(100):
                    latest_iteration.text(f'Iteration {i+1}')
                    bar.progress(i + 1)
                    time.sleep(0.02)
                
                #-----------Pre-Processing -----------
                # p = PreProcessor()
                # f = Funnel(p)
                # X= f.process_data(file_to_be_uploaded)

                #-----------Prediction Test-----------
                # with open(PATH,'rb') as f:
                # model.predict(X)

                #-----------Post-Processing-----------
                #postprocess_data(data)
                #Postpro = PostProcessor()
                st.write('')
                st.write('### Enjoy your guitar tabs!')
                st.write('')
                st.button("Guitar Tabs",on_click=nextpage,disabled=(st.session_state.page > 3))
         
             
# -----------------------------Page 2 (Guitar tabs)-------------------------------     
elif st.session_state.page == 1:
    with placeholder.container():
    
        # -----Create Sidebar------
        with st.sidebar:
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.subheader('About')
            st.markdown('Swizzle is a tool for musicians providing **AI generated music notation of songs**.')
            st.markdown("You can find the code on our [Github](https://github.com/swizzle-app/swizzle)")

        # ----------Create an dummy array for testing 6x21----------
        new = pd.DataFrame({'pos': [1, 2, 3, 4,5,6,7,8,9, 10, 11, 12,13,14,15,16,17,18,19,20,21,22,23,24],
                    'string': ['E','A','D','G','B','e','B','G','E','A','D','G','B','e','B','G','B','e','B','G','B','B','G','B'],
                    'fret': [1,2,3,4,5,6,5,4,1,2,3,4,5,6,5,4,5,6,5,4,2,5,4,2]})
        
        # -----Page layout and setup session state------
        st.image('media/swizzle_logo.png', width=400)
        st.markdown("---")
    
        left_column, right_column = st.columns(2)
        file_uploader = st.session_state['file_uploader']
        
        with left_column:
            
            #-----------Play song-----------
            st.write('### Play your song')
            st.write ("")
            st.audio(file_uploader, format="audio/wav", start_time=0, sample_rate=None)
        
            #-----------DISPLAY FILE-----------
            if file_uploader:
                st.write("Song: ", file_uploader.name)

        # ----------Setup variables for the for-loop----------
        j = len(new)/10
        j = math.ceil(j)
        l=10
        s=0
        tab=new[s:l]
        
        # -----Page layout and session state------
        st.write("---")
        left_column, center_column,right_column = st.columns(3)
        with left_column:
            st.write('### Guitar tabs')
        with right_column:
            st.button("Home",on_click=restart)
        
        # ----------Show guitar tabs using plotly express scatter plot----------
        for i in range(j):    
            tab=new[s:l]
            
            fig = px.scatter(tab,y="string", x="pos",text='fret', category_orders={"string": ["e", "B", "G", "D", "A","E"]},width=800, height=400,
            labels={"string": "","pos": ""})
            
            # -----Scatter Plot settings------
            fig.update_traces(marker_size=20)#, color='white')
            fig.update_traces(textposition="middle center")
            fig.update_traces(marker=dict(color='White'))
            fig.update_yaxes(gridcolor='black',showgrid=True)#, griddash='dash'
            fig.update_xaxes(gridcolor='white',showgrid=True,showticklabels=False)
            fig.update_layout(font=dict(family="Courier New, monospace",size=18,color="Black"))
            #These include "Arial", "Balto", "Courier New", "Droid Sans",, "Droid Serif", "Droid Sans Mono", "Gravitas One", "Old Standard TT", "Open Sans", "Overpass", "PT Sans Narrow", "Raleway", "Times New Roman".
            fig.update_layout(xaxis = dict(tickfont = dict(size=20)))
            fig.update_layout(yaxis = dict(tickfont = dict(size=20)))
            fig.add_hline(y="E",line_width=1)
            fig.add_hline(y="A",line_width=1)
            fig.add_hline(y="D",line_width=1)   
            fig.add_hline(y="G",line_width=1)
            fig.add_hline(y="B",line_width=1)
            fig.add_hline(y="e",line_width=1)
            fig.update_layout(font_family="Arial", showlegend=True)
            fig.update_yaxes(col=1,range=[-1,6+1])
            
            # -----Print Scatter Plot on streamlit------
            st.plotly_chart(fig)

            # ----------Print guitar tab patterns with 10 tones each----------
            l += 10
            s += 10
