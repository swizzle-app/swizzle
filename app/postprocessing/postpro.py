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

            # make values in y integers, so they are only 1 or 0 (the CNN outputs floats between 0 and 1)
            y = np.round(y)


            # loop over every frame
            for idx, frame in enumerate(y):
                self.logger.info(f"Processing frame {idx+1}/{y.shape[0]}.")
                # loop over every string
                for string, frets in enumerate(frame):
                    # extract fret indices
                    fret_idx = np.where(frets == 1)
                    self.logger.debug(f"Fret idices for string {string} at position {pos}: {fret_idx}")
                    # reshape data to output shape
                    if np.sum(fret_idx) > 0:
                        next_pos = True
                        for i in fret_idx:
                            # safety loop if multiple elements per string
                            # this shouldn't happen, but who knows
                            for j in i:
                                r.append([pos, string, np.squeeze(j)-1])
                
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
    

if __name__ == "__main__":

    mock_data = np.array([
                           [
                            [0, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
                            [1, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0.8, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                           ]
                        ])

    mock_results = np.array([
                                [0, 0, 0],
                                # A string: not played
                                [0, 2, 2],
                                # G string: not played
                                [0, 4, 9],
                                [0, 5, 0],
                                [1, 0, 0],
                                # A string: not played
                                [1, 2, 2],
                                # G string: not played
                                [1, 4, 9],
                                [1, 5, 0]
                           ])

    # print(f"Mock data shape: {mock_data.shape}")
    # print(f"Mock results shape: {mock_results.shape}")

    p = PostProcessor(verbose=4)
    results = p.postprocess_data(mock_data, mock_results)