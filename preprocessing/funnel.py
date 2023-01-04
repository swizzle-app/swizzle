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
from prepro import PreProcessor
import logging
import os

OUTPUTPATH = '../data/output/'

class Funnel():

    def __init__(self, verbose: int = 0, outputpath: str = OUTPUTPATH):
        
        self.alive = True

         # fix outputpath string
        if not outputpath.endswith('/'): 
            outputpath += '/'

        # save outputpath
        self.outputpath = outputpath

        # setup logger
        FORMAT = "[%(levelname)8s][%(filename)s:%(lineno)4s - %(funcName)20s() ] %(message)s"
        logging.basicConfig(format=FORMAT)
        verbosity = {0: logging.CRITICAL, 1: logging.ERROR, 2: logging.WARNING, 3: logging.INFO, 4: logging.DEBUG}
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(verbosity[verbose])

        self.logger.info('Funnel created.')

    
    def hello_world(self):
        """Check if Funnel has been created

        Returns:
            bool: True if initialized correctly.
        """
        return self.alive


    def training(self, save: bool = False, subset: float = 1):

        # create PreProcessor object
        p = PreProcessor()

        # get filenames in folder
        filenames = p.get_filenames()

        # only take subset of data
        if subset < 1:
            ubound = np.round(len(filenames) * subset, 0).astype(int)
            filenames = filenames[:ubound]

        X = []
        y = []

        # got through all files
        for idx, f in enumerate(filenames):

            self.logger.info(f"Processing file: {f} ({idx+1}/{len(filenames)})")

            # get audio and label file for first filename
            audio, labels = p.load_files(f)

            # Preprocess audio and labels
            p.preprocess_audio(audio)
            p.preprocess_labels(labels)

            # store each point in X and y
            for datapoint in p.output.get('windows'):
                X.append(datapoint)

            for labelpoint in p.output.get('windowlabels'):
                y.append(labelpoint)

        if save: self._save_output(X, y)
        
        return X, y


    def process_data(self, data, save: bool = False):
        
        # create preprocessor object
        p = PreProcessor()

        self.logger.info(f"Processing data.")

        # load data into librosa object
        data = librosa.load(data, sr=None)

        # preprocess data
        p.preprocess_audio(data)

        self.logger.info(f"Success!")

        # save
        if save: 
            self._save_outputself.p.output.get('windows')

        # return X
        return self.p.output.get('windows')


    def _save_output(self, data = "", labels = ""):
        """Saves output from training to .npz files

        Args:
            data (np.ndarray): preprocessed audio data
            labels (np.ndarray): preprocessed label data
        """

         # create directory
        if not os.path.exists(self.outputpath):
            os.makedirs(self.outputpath)

        # save files
        if data != "":
            np.savez(self.outputpath + "training" + "_" + "data" + ".npz", data)
            self.logger.info(f'Data was saved under {self.outputpath}')
        else:
            self.logger.warning('No data to save!')

        if labels != "":
            np.savez(self.outputpath + "training" + "_" + "labels" + ".npz", labels)
            self.logger.info(f'Labels were saved under {self.outputpath}')
        else:
            self.logger.warning('No labels to save!')


if __name__ == "__main__":
    f = Funnel(verbose=4)

    # create training data
    X, y = f.training(save=True, subset=0.05)