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

def generate_training_data(verbose: int = 0):
    """Generates Preprocessor and Funnel objects to process the dataset. Returns training data.
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

        p = PreProcessor(verbose=4)
        f = Funnel(p, verbose=4)

        # PRODUCTION
        # f.get_training_data(r=False, save=True, subset=1)
        
        # TESTING
        logger.info("-"*50)
        logger.info("TEST FUNCTION")
        logger.info("-"*50)
        f.get_training_data(r=False, save=False, subset=0.01)

        logger.info("-"*50)
        logger.info("Finished training data generation.")
        logger.info("-"*50)
    
    else:
        logger.warning(f"Wrong working directory (currently in : {os.getcwd().split('/')[-1]})")
        logger.warning(f"Trying to get to correct working directory...")

        if "app" in os.listdir(cwd):
            os.chdir("app")
            logger.warning(f"Done.")
            generate_training_data(verbose=4)

        else:
            logger.warning("Couldn't find \"app\" directory. Please change to it manually")


if __name__ == "__main__":
    generate_training_data(verbose=4)