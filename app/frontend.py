                         ###################
                        #                  #
 #######               #  #  #####  #####  #   ###
#       #      #      #   #     #      #   #  #   #
 ###     #    # #    #    #    #      #    #  ####
    #     #  #   #  #     #   #      #     #  #
####       ##     ##      #  #####  #####  #   ###


#############################################
#                   IMPORTS                 #
#############################################


# ---------- python packages ----------
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import math
import librosa
from tensorflow import keras  

# ---------- swizzle packages ----------
from preprocessing.prepro import PreProcessor
from postprocessing.postpro import PostProcessor

# ---------- Page layout ----------
st.set_page_config(layout='wide', page_title='swizzle', initial_sidebar_state='expanded', menu_items={'About': '\"swizzle\" is a tool for musicians providing **AI generated music notation of songs**.\n \n Created with :heart: by Flo, Ivy, Matthias, and Sebastian.\n \n Check our GitHub [here](https://github.com/swizzle-app/swizzle)!'})


# ----------Setup state session in streamlit----------
if "page" not in st.session_state: st.session_state['page'] = 0
if "audiofile" not in st.session_state: st.session_state['audiofile'] = None
if "X" not in st.session_state: st.session_state['X'] = None
if "y_pred" not in st.session_state: st.session_state['y_pred'] = None
if "tabs" not in st.session_state: st.session_state['tabs'] = None


# ---------- Pagestatus check functions ----------
def nextpage(): 
    st.session_state.page += 1
    st.experimental_rerun()

def restart(): 
    st.session_state.page = 0


# ---------- Swizzle Logo ----------
l, c, r = st.columns([5,10,2])
with c:
    if st.get_option('theme.base') == 'dark':
        st.image('media/swizzle_logo_dark.png', width=450)
    else:
        st.image('media/swizzle_logo_light.png', width=450)

st.markdown("---")

placeholder = st.empty()
# ---------- Sidebar ----------
with st.sidebar:
    st.subheader('About')
    st.markdown('\"swizzle\" is a tool for musicians providing **AI generated music notation of songs**.')
    st.markdown("You can find the code on our [Github](https://github.com/swizzle-app/swizzle)")
    st.write("")
    st.write("")
    st.write("")
    l, c, r = st.columns([1, 4, 1])
    with c:
        back_button_placeholder = st.empty()

# ----------------------------- Page 1 (Home) -------------------------------
if st.session_state.page == 0:
    # Replace the content of page with several elements:
    with placeholder.container():

        # ----- Page layout - create 3 columns ------
        l, c, r = st.columns([2,10,2])

        with c:

            # ----- Upload  WAV file ------
            st.write('### Upload your audio file')  
            audio_file = st.file_uploader(label=" ", type=[".wav"]) #, ".wave", ".flac", ".mp3", ".ogg"])
            
            # -----Store variable with state session------
            st.session_state['audiofile'] = audio_file
            
            #----- Swizzle Button ------
            st.write('')
            st.write('### Get your tabs!') 
            st.write('')

            if st.button("swizzle it"):
                
                if not audio_file:
                    st.error("You forgot to upload your recording! Nothing to swizzle here... :(", icon="🤖")

                else:
                    #----- Swizzle Spinner ------
                    with st.spinner('🤖 swizzling it...'):
                    
                        #----------- Pre-Processing -----------
                        p = PreProcessor()
                        audio, _ = librosa.load(audio_file, sr=22050, dtype=np.float32)
                        p.preprocess_audio(audio, training=True)
                        X = p.output['windows']

                        # Store preprocessed data
                        st.session_state['X'] = X

                        #----------- Loading the model -----------
                        swizzle_model = keras.models.load_model("../app/model/swizzle_model", compile=False)
                        
                        y_pred = swizzle_model.predict(st.session_state['X'])
                        st.session_state['y_pred'] = y_pred

                        #----------- Post-Processing -----------
                        postpro = PostProcessor()
                        post_pro_output = postpro.postprocess_data(y_pred, remove_duplicates=True)
                        st.session_state['tabs'] = post_pro_output

                    #----------- get guitar tabs -----------
                    nextpage()
                     
# -----------------------------Page 2 (Guitar tabs)-------------------------------     
elif st.session_state.page == 1:

    with st.sidebar:
        with back_button_placeholder.container():
            st.button("← go back home", on_click=restart)

    with placeholder.container():
        
        # -----Page layout and setup session state ------
        l, c, r = st.columns([2, 10, 2])

        with c:
            
            #-----------Play song-----------
            st.write('### Play your song')
            st.write ("")
            st.audio(st.session_state['audiofile'], format="audio/wav", start_time=0, sample_rate=None)
        
            #-----------DISPLAY FILE-----------
            if st.session_state['audiofile']:
                st.write("Song: ", st.session_state['audiofile'].name)

        # ----------Setup variables for the for-loop----------
        n_tabs = st.session_state['tabs'].shape[0]/10
        n_tabs = math.ceil(n_tabs)
        u_bound= 10
        l_bound= 0
        df = pd.DataFrame(st.session_state['tabs'], columns=["pos", "string", "fret"])

        # -----Page layout and session state------
        st.write("---")
        l, c, r = st.columns([2, 10, 2])
        with c:
            st.write('### Guitar tabs')
        
        # ----------Show guitar tabs using plotly express scatter plot----------
            for i in range(n_tabs):    
                # get next 10 notes
                tab = df[l_bound:u_bound].reset_index(drop=True)
                l_bound += 10
                u_bound += 10

                # pad tab if less than 10 notes
                if tab.shape[0] < 10:
                    # get padding needed
                    pad = 10 - tab.shape[0]
                    # get last position in tab
                    last_pos = tab.iloc[-1, 0]
                    # calculate next positions needed
                    next_pos = list(range(last_pos+1, last_pos+pad+1))
                    # construct filler with empty frets
                    filler = pd.DataFrame(np.transpose([next_pos, [''] * pad, [1] * pad]), columns=['pos', 'fret', 'string'])
                    tab = pd.concat([tab, filler], axis=0).reset_index(drop=True)

                # create scatter plot
                fig = px.scatter(tab, y="string", x="pos", text='fret', height=275, labels={"string": "","pos": ""})

                # style scatter plot
                fig.update_traces(marker_size=20, textposition="middle center", marker=dict(color='White'))
                fig.update_yaxes(showgrid=False)
                fig.update_xaxes(showgrid=False, showticklabels=False)
                fig.update_layout(font=dict(family="Arial", size=18, color="Black"),
                                  showlegend=False, paper_bgcolor='white', plot_bgcolor='white',
                                  margin=dict(l=0, r=30, t=50, b=50),
                                  xaxis=dict(tickfont = dict(size=20, color="black")),
                                  yaxis=dict(tickfont = dict(size=20, color="black"), tickvals=[0, 1, 2, 3, 4, 5], ticktext=["E", "A", "D", "G", "B", "e"]))
                for i in range(6):
                    fig.add_hline(y=i,line_width=1)
                
                # add "TAB"
                tab = ['T', 'A', 'B']
                for idx, t in enumerate(tab):
                    fig.add_annotation(dict(font=dict(color='black',size=20),
                                                x=0,
                                                y=-idx+3.5,
                                                showarrow=False,
                                                text=t,
                                                textangle=0,
                                                xanchor='left',
                                                xref="paper",
                                                yref="y"))

                # print the chart
                st.plotly_chart(fig, use_container_width=True)