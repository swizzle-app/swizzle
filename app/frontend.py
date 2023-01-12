# SWIZZLE - AI GENERATED MUSIC NOTATION FOR SONGS


# ----------import python packages----------
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
#import os
import math
import time
#import plotly.graph_objects as go
from tensorflow import keras  

# ----------Page layout----------
st.set_page_config(layout = "wide")

# ----------import user packages----------
from preprocessing.funnel import Funnel
from preprocessing.prepro import PreProcessor
from postprocessing.postpro import PostProcessor

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
            audio_file = st.file_uploader(label=" ", type=[".wav"]) #, ".wave", ".flac", ".mp3", ".ogg"])
            
            # -----Store variable with state session------
            st.session_state['audio_file'] = audio_file
            
            #-----Swizzle Button------
            st.write('')
            st.write('### Do the magic...') 
            st.write('') 
            if st.button("Swizzle it",):
                
                #-----Swizzle Spinner------
                with st.spinner('Swizzle it...'):
                
                    #-----------Pre-Processing -----------
                    p = PreProcessor()
                    f = Funnel(p)
                    X = f.process_data(audio_file)

                    st.session_state['pre_output'] = X

                    #-----------Prediction Test-----------
                    swizzle_model = keras.models.load_model("../app/model/swizzle_model")
                    y = swizzle_model.predict(X)
                    st.session_state['y'] = y

                    

                    #-----------Post-Processing-----------
                    postpro = PostProcessor()
                    post_pro_output = postpro.postprocess_data(y)
                    
                    # -----Store variable with state session------
                    st.session_state['post_pro_output'] = post_pro_output
                
                np.set_printoptions(threshold=np.inf)
                st.write(np.round(st.session_state.y,2))
                
                
                #-----------Guitar Tabs button-----------
                st.write('')
                st.write('### Enjoy your guitar tabs!')
                st.write('')
                st.button("Get guitar tabs",on_click=nextpage,disabled=(st.session_state.page > 3))
                     
# -----------------------------Page 2 (Guitar tabs)-------------------------------     
elif st.session_state.page == 1:
    with placeholder.container():
    
        # -----Create Sidebar------
        with st.sidebar:
            st.button("Go back home",on_click=restart)
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
        # new = pd.DataFrame({'pos': [1, 2, 3, 4,5,6,7,8,9, 10, 11, 12,13,14,15,16,17,18,19,20,21,22,23,24],
        #             'string': ['E','A','D','G','B','e','B','G','E','A','D','G','B','e','B','G','B','e','B','G','B','B','G','B'],
        #             'fret': [1,2,3,4,5,6,5,4,1,2,3,4,5,6,5,4,5,6,5,4,2,5,4,2]})
        
        # -----Page layout and setup session state------
        st.image('media/swizzle_logo.png', width=400)
        st.markdown("---")
        left_column, right_column = st.columns(2)
        audio_file = st.session_state['audio_file']
        post_pro_output = st.session_state['post_pro_output']
        
        #print(st.session_state.pre_output)
        #print(st.session_state.y)
        #print(post_pro_output)




        with left_column:
            
            #-----------Play song-----------
            st.write('### Play your song')
            st.write ("")
            st.audio(audio_file, format="audio/wav", start_time=0, sample_rate=None)
        
            #-----------DISPLAY FILE-----------
            if audio_file:
                st.write("Song: ", audio_file.name)

        # ----------Setup variables for the for-loop----------
        j = len(post_pro_output)/10
        j = math.ceil(j)
        u_bound=10
        l_bound=0
        #tab=post_pro_output[l_bound:u_bound]
        df=pd.DataFrame(post_pro_output, columns=["pos", "string", "fret"])

        # -----Page layout and session state------
        st.write("---")
        left_column, center_column,right_column = st.columns(3)
        with left_column:
            st.write('### Guitar tabs')
        #with right_column:
            #st.button("Home",on_click=restart)
        
        # ----------Show guitar tabs using plotly express scatter plot----------
        for i in range(j):    
            tab = df[l_bound:u_bound]
            
            #fig = px.scatter(tab,y=1, x=0,text=2,width=800, height=600,
            #labels={1: "",0: ""}) #category_orders={1: ["e", "B", "G", "D", "A","E"]}
            
            fig = px.scatter(tab,y="string", x="pos",text='fret',width=800, height=350,
            labels={"string": "","pos": ""}) #, category_orders={"string": ["e", "B", "G", "D", "A","E"]
            
            # -----Scatter Plot settings------
            fig.update_traces(marker_size=20)#, color='white')
            fig.update_traces(textposition="middle center")
            fig.update_traces(marker=dict(color='White'))
            fig.update_yaxes(gridcolor='black',showgrid=True)#, griddash='dash'
            fig.update_xaxes(gridcolor='white',showgrid=True,showticklabels=True)
            fig.update_layout(font=dict(family="Courier New, monospace",size=18,color="Black"))
            #These include "Arial", "Balto", "Courier New", "Droid Sans",, "Droid Serif", "Droid Sans Mono", "Gravitas One", "Old Standard TT", "Open Sans", "Overpass", "PT Sans Narrow", "Raleway", "Times New Roman".
            fig.update_layout(xaxis = dict(tickfont = dict(size=20,color="black")))
            fig.update_layout(yaxis = dict(tickfont = dict(size=20,color="black"), tickvals= [0,1,2,3,4,5],ticktext=["E","A","D","G","B","e"]))
            fig.add_hline(y=0,line_width=1)
            fig.add_hline(y=1,line_width=1)
            fig.add_hline(y=2,line_width=1)   
            fig.add_hline(y=3,line_width=1)
            fig.add_hline(y=4,line_width=1)
            fig.add_hline(y=5,line_width=1)
            #fig.add_vline(x=0,line_width=1)

            fig.update_layout(font_family="Arial", showlegend=True)
            fig.update_yaxes(col=1,range=[-1,6+1])
            
            fig.add_annotation(dict(font=dict(color='black',size=20),
                                        x=0,
                                        y=0.56,
                                        showarrow=False,
                                        text="T",
                                        textangle=0,
                                        xanchor='left',
                                        xref="paper",
                                        yref="paper"))
            fig.add_annotation(dict(font=dict(color='black',size=20),
                                        x=0,
                                        y=0.44,
                                        showarrow=False,
                                        text="A",
                                        textangle=0,
                                        xanchor='left',
                                        xref="paper",
                                        yref="paper"))
            fig.add_annotation(dict(font=dict(color='black',size=20),
                                        x=0,
                                        y=0.25,
                                        showarrow=False,
                                        text="B",
                                        textangle=0,
                                        xanchor='left',
                                        xref="paper",
                                        yref="paper"))
            # -----Print Scatter Plot on streamlit------
            st.plotly_chart(fig)

            # ----------Print guitar tab patterns with 10 tones each----------
            l_bound += 10
            u_bound += 10
