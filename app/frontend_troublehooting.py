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
        
        left_column, right_column = st.columns([5,1])
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
                    #st.write(X[:5])
                    #-----------Prediction Test-----------
                    swizzle_model = keras.models.load_model("../app/model/swizzle_model_v3", compile=False)
                    y = swizzle_model.predict(X)
                    st.session_state['y'] = y
                    #np.set_printoptions(threshold=np.inf)
                    #st.write(np.round(y[0:50][:],1))
                    #test =pd.DataFrame(y.tolist(), columns=['A','B','C',"D","E","F"])
                    #test =pd.DataFrame(y)
                    pd.describe_option('display')
                    #st.table(y.shape)                    #test = pd.DataFrame(data=y)
                    
                    st.write("---")
                    st.write("## Troubleshooting")
                    st.write("### Shape of the CNN output array: ",y.shape)
                    #st.write(np.round(y[1][0][15],2))
                    #st.write(np.round(y[2][0],2))
                    #st.write(np.round(y[:1][1],2))
                    
                    for i in range(int(len(y)/20)):
                        
                        st.write("---")
                        st.write("Position: ",i,"  String: 1","  Fret: ",np.argmax(y[i][0][1:20]))#,(y[i][0][np.argmax(y[i][0][1:20])]))
                            
                        st.write("Position: ",i,"  String: 2","  Fret: ",np.argmax(y[i][1][1:20]))#,(y[i][1][np.argmax(y[i][1][1:20])]))
                        st.write("Position: ",i,"  String: 3","  Fret: ",np.argmax(y[i][2][1:20]))#,(y[i][2][np.argmax(y[i][2][1:20])]))
                        st.write("Position: ",i,"  String: 4","  Fret: ",np.argmax(y[i][3][1:20]))#,(y[i][3][np.argmax(y[i][3][1:20])]))
                        st.write("Position: ",i,"  String: 5","  Fret: ",np.argmax(y[i][4][1:20]))#,(y[i][4][np.argmax(y[i][4][1:20])]))
                        st.write("Position: ",i,"  String: 6","  Fret: ",np.argmax(y[i][5][1:20]))#,(y[i][5][np.argmax(y[i][5][1:20])]))
                        st.write(np.round(y[i][0:6],decimals = 2))

                        #x=np.max(y[i][0][1:20])
                            #x=np.max(y[i][0],axis =0)
                           
                            
                            #st.write(x)
                            #st.write(y[i][0][x])
                            #st.write(y[[i][0][1:20]].max())

                        #st.write(y[i][0:6])
                        #st.write(np.round(y[i][0:6],3))
                        #st.write("Position: ",i)
                        #y= ndarray.round(decimals=2, out=None)
                        #j=0
                        #if (y[i][0][0]) <0.9 or y[i][1][0] <0.9 or y[i][2][0] <0.9 or y[i][3][0]<0.5 or y[i][4][0] <0.9 or y[i][5][0] <0.9:
                            #st.write("Position: ",i)
                            
                            #st.write("Position: ",i,"  String: 0","  Fret: ",np.argmax(y[i][0][1:20]))
                            #for j in range(20):
                             #   if y[i][0][j] == np.argmax(y[i][0],axis=0):
                             #       st.write(y[i][0][j])
                        #st.write("1.Spalte 1.Reihe: ",y[i][0][0]) #1.Spalte 1.Reihe
                        #st.write("2.Reihe  3.Spalte: ",y[i][1][2]) #2.Reihe  3.Spalte
                        #st.write("4.Reihe  3.Spalte",y[i][3][2]) #4.Reihe 3.Spalte
                        #y= np.round(y,5)
                        #1. Array , 2.Reihe, 3. Spalte
                        
                        # for y[i][j][0] in range(6):
                        #     for k in range(1,21):
                        #         if y[i][j][k] >0.5 == True:
                        #             st.write("Position: ",i)
                        #             #st.write("Position: ",y, " ",j," ",k)
                        #             #temp =y[i][j][k]
                        #             #st.write( "Value: ",temp) 
                                    
                        #st.write(y[i][0:6])
                        
                        #st.write(np.round(y[0]))
                        #st.write(np.round(y[0]),decimals = 5)
                        
                        

                        
                        #st.write(y[i][0:6])
                            #st.write(y[i][0][0]<0.5)
                            #st.write(y[i][0][0]>0.5)
                            #st.write(y[i][0][0])
                            #st.write(y[i][1][0])
                            #st.write(y[i][2][0])
                            # st.write(y[i][3][0])
                            # st.write(y[i][4][0])
                            # st.write(y[i][5][0])
                            #st.write(y[i][0:6])
                            #for j in range (int((y[i][j][0]))):
                                #st.write(np.where(y[i][j][0] <0.5) )
                                #st.write((y[i][j][0])[np.where((y[i][j][0])<0.5)]  )
                    #st.write(np.round(y[10][0],2))
                    #st.write(np.round(y[30][0],2))
                    #st.write(np.round(y[100][0],2))
                    #f=y.transpose(2,0,1)
                    #st.write(f.shape)
                    #e = A.transpose(2,0,1).reshape(c,-1)
                    #reshaped_array=np.reshape(y,(6,21))
                    #st.write(reshaped_array.shape)
                    #st.write(reshaped_array)
                    #test.to_csv("test.csv")
                    
                    #newarr = y.reshape(y.shape[0], (y.shape[1]*y.shape[2]))
                    #st.write(newarr[0:5])
                    # for i in range(len(y)):

                    #     #if y[i][0][0]<0.5:
                    #         #st.write(np.round(y,1))
                    #     for j in range (6):
                    #         string= y[0][0][j]
                    #         if string <0.5:
                    #             st.write(y[i])        
                    #pd.DataFrame(y).to_csv("../app/test.csv")
                    #y.to_csv("../app/test.csv", header=None)

                    #-----------Post-Processing-----------
                    postpro = PostProcessor()
                    post_pro_output = postpro.postprocess_data(y)
                    
                    # -----Store variable with state session------
                    
                    st.session_state['post_pro_output'] = post_pro_output
                    #st.write("### Postprocessing output: ",post_pro_output[:50])
                    #np.set_printoptions(threshold=np.inf)

            #left_column, right_column = st.columns(2)
            #st.write("### Check CNN versus postprocessing output")
            #with left_column:
                    #cnn=pd.DataFrame(y)
                    #st.table(y)
            #st.write(np.round(st.session_state.y[0:10],2))
            #with right_column:
            #st.table(post_pro_output)
                
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
        left_column, right_column1 = st.columns(2)
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

            # new = pd.DataFrame({'pos': [1, 2, 3, 4,5,6,7,8,9, 10, 11, 12,13,14,15,16,17,18,19,20,21,22,23,24],
            #          'string': ['E','A','D','G','B','e','B','G','E','A','D','G','B','e','B','G','B','e','B','G','B','B','G','B'],
            #          'fret': [1,2,3,4,5,6,5,4,1,2,3,4,5,6,5,4,5,6,5,4,2,5,4,2]})
            
            # st.write(new)
            # st.dataframe(new)
        
        # ----------Setup variables for the for-loop----------
        j = len(post_pro_output)/10
        j = math.ceil(j)
        u_bound=10
        l_bound=0
        #tab=post_pro_output[l_bound:u_bound]
        df=pd.DataFrame(post_pro_output, columns=["pos", "string", "fret"])

        # -----Page layout and session state------
        st.write("---")
        left_column2, center_column,right_column2 = st.columns(3)
        
        with right_column2:
            st.write('### Postprocessing')
            st.write("")
            st.write("")
            st.write("")
            #st.markdown(df.style.set_table_styles(styles).to_html(),unsafe_allow_html=True)
            #df.style.hide_index()
            st.table(df[1:])
        with left_column2:
            st.write('### Guitar tabs')
        #with right_column:
            #st.button("Home",on_click=restart)
            #st.table(post_pro_output)
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
        
        ######################## FOR TEST####################
       
        #with right_column2:
            #pd.set_option('expand_frame_repr', True)
            # pd.set_option("display.max_rows", None, "display.max_columns", None)
            
            
            # new = pd.DataFrame({'pos': [1, 2, 3, 4,5,6,7,8,9, 10, 11, 12,13,14,15,16,17,18,19,20,21,22,23,24],
            #          'string': ['E','A','D','G','B','e','B','G','E','A','D','G','B','e','B','G','B','e','B','G','B','B','G','B'],
            #          'fret': [1,2,3,4,5,6,5,4,1,2,3,4,5,6,5,4,5,6,5,4,2,5,4,2]})
            
            # st.write(new)
            # st.write("")
            # st.write("")
            # st.write("")
            # st.table(new)
            
            #st.write(post_pro_output)
            #print(my_arr.tolist())
        ######################## FOR TEST####################