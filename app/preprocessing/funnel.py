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
import logging
import os


#############################################
#                   CONSTANTS               #
#############################################
OUTPUTPATH = '../data/output/'
AUDIOPATH = '../data/raw/'
RECMODES = ['mm', 'mp', 'pd', 'po']


class Funnel:

    def __init__(self, preprocessor, verbose: int = 0, outputpath: str = OUTPUTPATH, f: str = ""):

         # fix outputpath string
        if not outputpath.endswith('/'): 
            outputpath += '/'

        self.p = preprocessor

        # save outputpath
        self.outputpath = outputpath
        self.filter = f

        # help
        self.possible_recmodes = {'all': 'all recording modes', 'mm': 'mono-mic', 'mp': 'mono-pickup_mix', 'pd': 'hex-pickup_debleeded', 'po': 'hex-pickup_original'}

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
        return print(f"I'm alive!")


    def get_training_data(self, r: bool = True, save: bool = False, rec_modes: list = RECMODES, subset: float = 1, filter: str = None, remove_noise: float = 0.95):
        """Generates training data (X and y) from the GuitarSet dataset. Input can be filtered by filenames (e.g. "solo" or "comp").
        Check out attribute "possible_recmodes" for keywords that can be given to rec_modes.

        Args:
            r (bool, optional): Whether the data should be returned. Defaults to True.
            save (bool, optional): Whether to save output as npz files. Defaults to False.
            rec_modes (list, optional): What recording modes to consider. Defaults to all of them (['mm', 'mp', 'pd', 'po'])
            subset (float, optional): Fraction of input to process. Defaults to 1.
            filter (str, optional): Filter for input filenames. Defaults to None.
            remove_noise (float, optional): Remove fraction of frames that have no notes played. Defaults to 0.95.

        If r is set to True, returns:
            list: X (data)
            list: y (labels)
        """

        # checking recording modes
        if 'all' in rec_modes:
            rec_modes = RECMODES

        # get filenames in folder
        filenames = self.p.get_filenames()

        # filter data
        if filter != None:
            len_orig = len(filenames)
            filenames = [f for f in filenames if filter in f]
            self.logger.info(f"Filtering files for '{filter}' ({len(filenames)}/{len_orig} files match filter).")

        # only take subset of data
        if subset < 1:
            ubound = np.round(len(filenames) * subset, 0).astype(int)
            filenames = filenames[:ubound]
            self.logger.info(f"Taking {subset*100}% of the data.")

        X = []
        y = []

        # got through all files
        for idx, f in enumerate(filenames):

            self.logger.info(f"Processing file: {f} ({idx+1}/{len(filenames)})")

            # go through all recording modes wanted
            for rmidx, rm in enumerate(rec_modes):

                self.logger.info(f"--> recording mode: {rm} ({rmidx+1}/{len(rec_modes)})")

                # get audio and label file for first filename
                audio, labels = self.p.load_files(f, rec_mode=rm)

                # Preprocess audio and labels
                self.p.preprocess_audio(audio)
                self.p.preprocess_labels(labels)

                # remove empty frames
                if remove_noise > 1: remove_noise = 1
                if remove_noise < 0: remove_noise = 0
                remove_noise = float(remove_noise)
                if remove_noise > 0:
                    self.p.remove_noise(fraction=remove_noise)

                # store each point in X and y
                for datapoint in self.p.output.get('windows'):
                    X.append(datapoint)

                for labelpoint in self.p.output.get('windowlabels'):
                    y.append(labelpoint)

        # save output into npz files if needed
        if save: 
            self.logger.info("Saving data.")
            self._save_output(data=X, labels=y, suffix=str(int(remove_noise*100)))
        
        # return values if needed
        if r: 
            self.logger.info("Returning data.")
            return X, y


    def process_data(self, data, save: bool = False):
        """Process user audio data provided from the frontend

        Args:
            data (file-like): 
            save (bool, optional): Whether to save the output to a .npz file. Defaults to False.

        Returns:
            list: X (data)
        """

        self.logger.info(f"Processing data.")

        # load data into librosa object
        data, _ = librosa.load(data, sr=None)

        # preprocess data
        self.p.preprocess_audio(data, training=False)

        self.logger.info(f"Success!")

        # save
        if save: 
            self._save_output(data=self.p.output.get('windows'))

        # return X
        return self.p.output.get('windows')


    def _save_output(self, data = "", labels = "", suffix = ""):
        """Saves output from training to .npz files

        Args:
            data (np.ndarray): preprocessed audio data
            labels (np.ndarray): preprocessed label data
        """

         # create directory
        if not os.path.exists(self.outputpath):
            os.makedirs(self.outputpath)

        # set filter string
        f = ""
        if self.filter == "":
            f = 'all'
        else:
            f = self.filter

        # save files
        if data != "":
            np.savez(self.outputpath + "training" + "_" + "data" + "_" + f + "_" + suffix + ".npz", data)
            self.logger.info(f'Data was saved under {self.outputpath}')
        else:
            self.logger.warning('No data to save!')

        if labels != "":
            np.savez(self.outputpath + "training" + "_" + "labels_" + f + "_" + suffix + ".npz", labels)
            self.logger.info(f'Labels were saved under {self.outputpath}')
        else:
            self.logger.warning('No labels to save!')


if __name__ == "__main__":
    f = Funnel(verbose=4)
    print(f"Alive: {'Yes!' if f.hello_world() else 'No :('}")