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
import numpy as np
import librosa
import librosa.display
import jams
import os
import logging


#############################################
#                   CONSTANTS               #
#############################################
AUDIOPATH = '../data/raw/'
RECPATHS = {'mm': 'audio_mono-mic/', 'mp': 'audio_mono-pickup_mix/', 'pd': 'audio_hex-pickup_debleeded/', 'po': 'audio_hex-pickup_original/'}
RECMODES = {'mm': '_mic', 'mp': '_mix', 'pd': '_hex_cln', 'po': '_hex'}
LABELPATH = '../data/raw/annotation/'
OUTPUTPATH = '../data/output/'
STANDARDE = [40, 45, 50, 55, 59, 64]


class PostProcessor:

    def __init__(self, verbose: int = 3):

        # setup logger
        FORMAT = "[%(levelname)8s][%(filename)s:%(lineno)4s - %(funcName)20s() ] %(message)s"
        logging.basicConfig(format=FORMAT)
        verbosity = {0: logging.CRITICAL, 1: logging.ERROR, 2: logging.WARNING, 3: logging.INFO, 4: logging.DEBUG}
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(verbosity[verbose])

    
    def read_data():
        pass