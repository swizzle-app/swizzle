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
import logging
import os


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

    
    def postprocess_data(self, y: np.array, test: np.array = np.zeros((0))) -> np.array:
        """Processes predictions from swizzle. Returns list with position, string, fret.

        Args:
            y (np.array): Prediction array from swizzle
            test (np.array, optional): Testoutput for method testing. Defaults to np.zeros((0)).

        Returns:
            np.array: List in with shape (n, 3) and columns ('position', 'string', 'fret').
        """

        self.logger.info("Starting postprocessing.")
        self.logger.info("Checking shape.")

        # checking the shape of input
        if y.shape[1] == 6 and y.shape[2] == 21:

            self.logger.info(f"Received {y.shape[0]} frames.")

            r = []
            pos = 0
            next_pos = False

            # map probability outputs of the cnn to 1 and 0 using argmax
            y_corr = np.zeros_like(y)

            for fidx, frame in enumerate(y):
                for sidx, string in enumerate(frame):
                    y_corr[fidx][sidx][np.argmax(string)] = 1

            # input: n x (6, 21)
            # convert frame to midi note (string_base_note + fret)
            # store midi note and frame index in midi_curr, idx_curr
            # continue until new midi note: midi_new, idx_new
            # get index of last seen midi_curr
            # remove positions idx_curr+1 to idx_new

            midi_curr = 0

            # loop over every frame
            for fidx, frame in enumerate(y_corr):
                self.logger.info(f"Processing frame {fidx+1}/{y.shape[0]}.")
               # loop over every string
                for sidx, string in enumerate(frame):
                    # extract fret indices
                    fret_idx = np.where(string == 1)
                    self.logger.debug(f"Fret indices for string {sidx} at position {pos}: {fret_idx}")
                    # reshape data to output shape
                    if np.sum(fret_idx) > 0:
                        # extract empty string midi value
                        esmv = STANDARDE[sidx]
                        # check if note is new
                        if esmv + fret_idx - 1 != midi_curr:
                            # allow for position to count up
                            next_pos = True
                            # append frame if new note found
                            for i in fret_idx:
                                # safety loop if multiple elements per string
                                # this shouldn't happen, but who knows
                                for j in i:
                                    r.append([pos, sidx, np.squeeze(j)-1])
                            # set midi_curr new note
                            midi_curr = esmv + fret_idx - 1
                        # just do nothing if note is already known
                        else:
                            next_pos = False
                
                # next position
                if next_pos: 
                    pos += 1
                    next_pos = False

            self.logger.info("Done.")

            if test.size > 0:
                self.logger.debug("Testing output against test data:")
                if np.array(r).shape == test.shape:
                    self.logger.debug("Shape: passed.")
                    self.logger.debug(f"Content: {'passed' if np.all(r == test) else 'failed'}.")
                else:
                    self.logger.debug("Shape: failed.")

            return np.array(r)

        else:
            self.logger.error("Data is in the wrong shape (expects (n, 6, 21).")
    

def test():

    mock_data = np.array([
                           [
                            [0, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0.2, 0, 0.2, 0, 0, 0.3, 0, 0, 0, 0.4, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                           ],
                           [
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           ],
                           [
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0.9, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0.8, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                           ],
                           [
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0.9, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0.8, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                           ],
                           [
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0.9, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0.8, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                           ],
                           [
                            [0, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0.2, 0, 0.2, 0, 0, 0.3, 0, 0, 0, 0.4, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                           ],
                        ])

    mock_results = np.array([  # P, S, F
                                [0, 0, 0],
                                # A string: not played
                                [0, 2, 2],
                                [0, 3, 8],
                                [0, 4, 9],
                                [0, 5, 0],
                                [1, 0, 0],
                                # A string: not played
                                [1, 2, 2],
                                # G string: not played
                                [1, 4, 8],
                                [1, 5, 0],
                                [2, 0, 0],
                                # A string: not played
                                [2, 2, 2],
                                [2, 3, 8],
                                [2, 4, 9],
                                [2, 5, 0],
                           ])

    print(f"Mock data shape: {mock_data.shape}")
    print(f"Mock results shape: {mock_results.shape}")

    p = PostProcessor(verbose=4)
    results = p.postprocess_data(mock_data, mock_results)

    print(results)


def singlesong():
    # cwd = os.getcwd()
    # print(cwd)
    y = np.load('app/model/model_output_singlesong.npy')
    pp = PostProcessor()
    
    print(pp.postprocess_data(y))


if __name__ == "__main__":
    test()
    # singlesong()