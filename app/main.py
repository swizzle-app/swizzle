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
from preprocessing.funnel import Funnel
from preprocessing.prepro import PreProcessor

import os
import logging

def generate_training_data(verbose: int = 0, r: bool = False, save: bool = True, subset: float = 1, filter: str = None, remove_noise: float = 0.95):
    """Generates Preprocessor and Funnel objects to process the dataset. Returns training data.

    Args:
        verbose (int, optional): Verbosity of function (0-4). Defaults to 0.
        r (bool, optional): Whether to return the data generated. Defaults to False.
        save (bool, optional): Wheter to save the data generated (*.npz). Defaults to True.
        subset (float, optional): What fraction of the data to use (0-1). Defaults to 1 (all data).
        filter (str, optional): Filter the data (e.g. "solo" or "comp"). Defaults to None
    """

    # setup logger
    FORMAT = "[%(levelname)8s][%(filename)s:%(lineno)4s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(format=FORMAT)
    verbosity = {0: logging.CRITICAL, 1: logging.ERROR, 2: logging.WARNING, 3: logging.INFO, 4: logging.DEBUG}
    logger = logging.getLogger(__name__)
    logger.setLevel(verbosity[verbose])

    # check if correct cwd
    cwd = os.getcwd()
    if cwd.endswith("app"):

        logger.info("-"*50)
        logger.info("Starting training data generation.")
        logger.info("-"*50)

        p = PreProcessor(verbose=verbose)
        f = Funnel(p, verbose=verbose, f=filter)

        if r == False and save == False:
            logger.info("Data will neither be returned nor saved. Maybe your forgot to set your output (parameters r and/or save)?")

        # if data should be returned, return function's return
        if r:
            return f.get_training_data(r=r, save=save, subset=subset, filter=filter, remove_noise=remove_noise)
        
        # else just save the files
        else:
            f.get_training_data(r=r, save=save, subset=subset, filter=filter, remove_noise=remove_noise)

        logger.info("-"*50)
        logger.info("Finished training data generation.")
        logger.info("-"*50)
    
    else:
        logger.warning(f"Wrong working directory (currently in : {os.getcwd().split('/')[-1]})")
        logger.warning(f"Trying to get to correct working directory...")

        if "app" in os.listdir(cwd):
            os.chdir("app")
            logger.warning(f"Done.")
            print(r, save, subset, filter)
            generate_training_data(verbose=verbose, r=r, save=save, subset=subset, filter=filter)

        else:
            logger.warning("Couldn't find \"app\" directory. Please change to it manually")


if __name__ == "__main__":

    r = False
    save = True
    filter = '' # solo, comp or empty
    
    generate_training_data(verbose=4, r=r, save=save, subset=1, filter=filter, remove_noise=0.95)