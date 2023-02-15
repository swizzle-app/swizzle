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
import logging
import plotly
import plotly.express as px
import math
import json
import pandas as pd
import numpy as np


class Plotter():

    def __init__(self, verbose: int = 3) -> None:
        # setup logger
        FORMAT = "[%(levelname)8s][%(filename)s:%(lineno)4s - %(funcName)20s() ] %(message)s"
        logging.basicConfig(format=FORMAT)
        verbosity = {0: logging.CRITICAL, 1: logging.ERROR, 2: logging.WARNING, 3: logging.INFO, 4: logging.DEBUG}
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(verbosity[verbose])


    def make_plots(self, data) -> json:
        """Returns list of JSON object from postprocessing outcomes."""

        self.logger.info("Making plots.")

        # split up data in parts of 10
        n_tabs = data.shape[0]/10
        n_tabs = math.ceil(n_tabs)
        u_bound= 10
        l_bound= 0
        df = pd.DataFrame(data, columns=["pos", "string", "fret"])

        # plot list
        j = []

        # generate plots and store in json
        for i in range(n_tabs):    
            self.logger.info(f"Plot {i+1}/{n_tabs}")
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
                                            

            j.append(plotly.io.to_json(fig, validate=True, pretty=False, engine="json") )

        self.logger.info("Done.")

        return j
