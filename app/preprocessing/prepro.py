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
#                   TO DO                   #
#############################################
#                                           #
# 1. Collect filenames in JAMS folder   ✓   #
# 2. Load wave and jams files           ✓   #
# 3. Create spectrogram                 ✓   #
# 4. Extract windows                    ✓   #
# 5. Extract window labels              ✓   #
# 6. Store data in object               ✓   #
#                                           #
#############################################


#############################################
#                   CONSTANTS               #
#############################################
AUDIOPATH = '../data/raw/'
RECPATHS = {'mm': 'audio_mono-mic/', 'mp': 'audio_mono-pickup_mix/', 'pd': 'audio_hex-pickup_debleeded/', 'po': 'audio_hex-pickup_original/'}
RECMODES = {'mm': '_mic', 'mp': '_mix', 'pd': '_hex_cln', 'po': '_hex'}
LABELPATH = '../data/raw/annotation/'
OUTPUTPATH = '../data/output/'
STANDARDE = [40, 45, 50, 55, 59, 64]


class PreProcessor():

    def __init__(self, audiopath: str = AUDIOPATH, labelpath: str = LABELPATH, outputpath: str = OUTPUTPATH,
                 tuning: list = STANDARDE, frets: int = 19,
                 hop_length: int = 512, bins: int = 192, bins_per_octave: int = 24, sr: int = 22050, normalize: bool = True, 
                 window_width: int = 9, verbose: int = 3) -> None:
        """Generates a preprocessing object

        Args:
            audiopath (str, optional): Path to audiofiles relative to cwd. Defaults to '../data/raw/'.
            labelpath (str, optional): Path to annotation files relative to cwd. Defaults to '../data/raw/annotation/'.
            outputpath (str, optional): Path to data output. Defaults to '../data/output/'.
            tuning (list, optional): Tuning of guitar. Defaults to [40, 45, 50, 55, 59, 64].
            frets (int, optional): Frets of guitar. Defaults to 19.
            hop_length (int, optional): hop_length parameter for CQT. Defaults to 512.
            bpo (int, optional): Bins per octave parameter for CQT. Defaults to 24.
            sr (int, optional): Sampling rate parameter for CQT. Defaults to 22050.
            normalize (bool, optional): Normalize amplitudes of audiofiles. Defaults to True.
            verbose (int, optional): Verbosity level of logger. Defaults to 3 (level: debug).
        """

        # set input and output paths
        self.audiopath = audiopath
        self.labelpath = labelpath
        self.outputpath = outputpath

        # guitar settings
        self.tuning = tuning
        self.n_strings = len(tuning)
        self.n_frets = frets

        # audio settings
        self.sr = sr
        self.normalize = normalize
        self.audiolength = 0

        # spectrogram settings
        self.hop_length = hop_length
        self.bins = bins
        self.bins_per_octave = bins_per_octave

        # preprocessing settings
        self.winwidth = window_width

        # output settings
        self.curr_file = ""
        self.curr_rm = ""
        self.n_classes = self.n_frets + 2
        self.output = {}

        # setup logger
        FORMAT = "[%(levelname)8s][%(filename)s:%(lineno)4s - %(funcName)20s() ] %(message)s"
        logging.basicConfig(format=FORMAT)
        verbosity = {0: logging.CRITICAL, 1: logging.ERROR, 2: logging.WARNING, 3: logging.INFO, 4: logging.DEBUG}
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(verbosity[verbose])


    def get_filenames(self, path: str = "") -> list:
        """Returns a list of filenames in a given directory

        Args:
            path (str, optional): Path to the directory. Defaults to '../data/raw/annotation/'.

        Returns:
            list: List containing sorted filenames of files in directory
        """
        # default path
        if not path:
            path = self.labelpath

        self.logger.info(f'Getting filenames from path {path}.')
        return np.sort(os.listdir(path))


    def load_files(self, filename: str, rec_mode: str = 'mm'):
        """Takes filename and looks for the respective audio and jams file in AUDIOPATH and LABELPATH. Returns librosa and jams object.

        Args:
            filename (str): Filename of jams file (with or without extension).
            rec_mode (str): Either mm, mp, pd, or po (mono_mic, mono_pickup, pickup_debleeded, or pickup_original). Defaults to mm.

        Returns tuple:
            audio (np.ndarray): librosa object containing audio
            labels (jams.JAMS): jams object containing labels
        """

        # get recording-style path and mode
        rp = RECPATHS[rec_mode]
        rm = RECMODES[rec_mode]

        # generate filenames
        filename = filename.split('.')[0]
        audiofile = self.audiopath + rp + filename + rm + '.wav'
        labelfile = self.labelpath + filename + '.jams'

        # save current filename and recording mode for output files
        self.curr_file = filename
        self.curr_rm = rm

        # extract audio
        audio, _ = librosa.load(audiofile, sr=self.sr, dtype=np.float32)

        # save audio length
        self.audiolength = librosa.get_duration(y=audio, sr=self.sr, hop_length=self.hop_length)

        self.logger.info(f"Loading files {audiofile} and {labelfile}.")
        return audio, jams.load(labelfile)


    def preprocess_audio(self, data: np.ndarray):
        """Normalizes audio, calculates CQT and hands data over to _get_windows() for sliding window generation. Data is stored in self.output['data'].

        Args:
            data (np.ndarray): Audio data as np.ndarray.

        """
        # Ensure data has dtype float
        data = data.astype(float)

        # Normalize
        if self.normalize:
            data = librosa.util.normalize(data)

        # ConstantQ transformation
        data = np.abs(librosa.cqt(data, sr=self.sr, hop_length=self.hop_length, n_bins=self.bins, bins_per_octave=self.bins_per_octave))

        # Swapping axes so arrays are "across frequency bins" instead of "along timepoints"
        self.output['data'] = np.swapaxes(data, 0, 1)

        # get data windows
        self._get_windows(self.output['data'])
        

    def preprocess_labels(self, labels: jams.JAMS):
        
        notes_played = []

        # get midi notes played per string
        for string, data in enumerate(labels.annotations['note_midi']):
            for observation in range(len(data['data'])):
                notes_played.append([string, np.round(data['data'][observation][2]), data['data'][observation][0]])
        
        # sort ascending by time
        notes_played = np.array(sorted(notes_played, key=lambda x: x[-1]))
        
        # store labels
        self.output['labels'] = labels

        # get label windows
        self._get_windowlabels(notes_played)


    def _pad_frame(self, data: np.ndarray, pad_left: bool):
        """Padding function to account for windows which left or right bounds are < 0 or > len(data)

        Args:
            input (list): frame to be padded
            width (int, optional): Window width. Defaults to 9.
            left (bool, optional): Left or right padding. Defaults to True.

        Returns:
            list: the padded input.
        """
        orig_width = len(data)
        padding = self.winwidth - orig_width

        
        if padding == 0:
            return data

        else:
            self.logger.info(f" --> Padding a window of width {orig_width}. Padding size: {padding}.")
            
        if pad_left:
            data = np.concatenate([[np.zeros((self.bins,))] * padding, data], axis=0)

        else:
            data = np.concatenate([data, [np.zeros((self.bins,))] * padding], axis=0)

        return data


    def _get_windows(self, data):
        """Sliding window function to extract windows with set width from an input array. Windows are stored in self.output['windows'].

        Args:
            data (np.ndarray): preprocessed audio data

        """
        
        # initialize windows array
        windows = np.zeros(shape=(len(data), self.bins, self.winwidth))

        # calculate half-width of window
        half_width = self.winwidth//2


        # Data is in format: TIME x FREQUENCY
        # Loop over the different timepoints and save windows that are 9 timepoints wide,
        # centered around the current timepoint. If no 9 timepoints are available
        # (left and right edge), pad the windows accordingly.
        #
        # Pseudocode:
        # 
        # 1. get current timepoint
        # 2. calculate left and right timepoints (-+ half_width)
        # 3. if either is < 0 or > len(data) --> padding
        # 4. Padding means adding winwidth - lbound zero-arrays

        # f_idx/fbin: different frequency bins
        # t_idx/tpoint: different timepoints
        for t_idx in range(len(data)):
            # set left and right bounds, so that item at t_idx is centered
            lbound = t_idx - half_width
            rbound = t_idx + half_width + 1

            self.logger.info(f"Extracting window {t_idx}/{len(data)-1}. Frames: {lbound, rbound}", )

            # if bounds > 0 and < len(data), simply extract window
            if lbound >= 0 and rbound < len(data):
                windows[t_idx] = data[lbound:rbound].T
            
            # if left bound below zero, pad left
            elif lbound < 0:
                windows[t_idx] = self._pad_frame(data[0:rbound], True).T

            # if right bound greater than input length, pad right
            elif rbound >= len(data):
                windows[t_idx] = self._pad_frame(data[lbound:], False).T

        self.output['windows'] = windows


    def _get_windowfrets(self, windowlabels, n_windows: int):
        """Converts MIDI note values to fret positions. Windows are stored in self.output['windowlabels'].

        Args:
            windowlabels (list): list with windowlabels
            n_windows (int): number of windows
        """

        # initialize windows array
        windows = np.zeros(shape=(n_windows, self.n_strings, self.n_classes))
    
        # 1: get string played
        # 2: get empty string midi value (esmv)
        # 3: subtract esmv from played note midi value
        # 3.1: if fret played is 0,
        # 4: replace respective value in fretboard
        # 5: set first value to 1, if all other values are 0

        for widx, window in enumerate(windowlabels):
            self.logger.info(f"Current window: {widx, np.reshape(window, (len(window)*2,))}")
            if window.size > 0:
                # empty string midi value from string played (0-5)
                for item in window:
                    esmv = self.tuning[item[0].astype(int)]
                    # convert played note to fret
                    fret = (item[1] - esmv + 1).astype(int)
                    # set fret to 1 in fretboard
                    windows[widx][item[0].astype(int)][fret] = 1
        
            # if no note was played in window, set first values to 1
            elif window.size == 0: 
                for idx in range(len(windows[widx])):
                    windows[widx][idx][0] = 1 

        for widx in range(len(windows)):
            for idx, string in enumerate(windows[widx]):
                if sum(string) == 0:
                    windows[widx][idx][0] = 1
            
        # store in output
        self.output['windowlabels'] = windows

    def _get_windowlabels(self, labels):
        """Sliding window function to extract windows with set width from an input array.

        Args:
            labels (np.ndarray): preprocessed label data

        """

        # get number of windows
        if 'windows' in self.output:
            n_windows = len(self.output.get('windows'))
        else:
            self.logger.warning('No data windows found.')
            n_windows = 0

        # calculate half-width of window
        half_width = self.winwidth//2

        # Data has columns: STRING MIDI STARTTIME
        # Check if note was played or is still ringing within a window's start and endtime.
        # Collect all notes for which this is true and save as windowlabels.
        # Repeat this for all windows

        windows = []

        for w in range(n_windows):
            lbound = ((w-half_width) / n_windows * self.audiolength)
            rbound = ((w+half_width) / n_windows * self.audiolength)

            self.logger.debug(f"Current timeframe: {np.round(lbound, 3):.3f} - {np.round(rbound, 3):.3f} / {np.round(self.audiolength, 3):.3f}")
            mask = ((labels[:, 2] >= lbound) & (labels[:, 2] <= rbound))
            windows.append(labels[mask, :-1])
            self.logger.debug(f"Found labels: {labels[mask, :-1]}")
        
        self._get_windowfrets(windows, n_windows)



    def save_output(self, path: str = "", suffix: list = ['data', 'labels']):
        """Saves data and labels stored as .npz files.

        Args:
            path (str, optional): Path to save files to. If directory doesn't exist, it will be created. Defaults to '../data/output/'.
            suffix (list, optional): Suffixes for data and labels files. Defaults to ['data', 'labels'].
        """
        # create directory
        if not os.path.exists(self.outputpath):
            os.makedirs(self.outputpath)

        # get path
        if not path:
            path = self.outputpath

        # fix path string
        if not path.endswith('/'): 
            path += '/'

        # save files if data is present
        if 'windows' in self.output:
            np.savez(path + self.curr_file + self.curr_rm + "_" + suffix[0] + ".npz", self.output.get('windows'))
            self.logger.info(f'Data was saved under {path}')
        else:
            self.logger.warning('No data to save!')

        if 'windowlabels' in self.output:
            np.savez(path + self.curr_file + self.curr_rm + "_" + suffix[1] + ".npz", self.output.get('windowlabels'))
            self.logger.info(f'Labels were saved under {path}')
        else:
            self.logger.warning('No labels to save!')
        

    def _test(self):

        if os.path.exists(self.outputpath + '00_BN1-129-Eb_solo_mic_data_notebook.npz'):
            test_array = np.load(self.outputpath + '00_BN1-129-Eb_solo_mic_data_notebook.npz')['arr_0'][0]
            self.logger.debug(f"Testing windows output: {'passed' if np.sum(np.round(self.output.get('windows')[0], 3) == np.round(test_array, 3)) == test_array.shape[0] * test_array.shape[1] else 'failed.'}")
        else:
            self.logger.warning("Windows testfile not found.")

        if os.path.exists(self.outputpath + '00_BN1-129-Eb_solo_mic_labels_notebook.npz'):
            test_array = np.load(self.outputpath + '00_BN1-129-Eb_solo_mic_labels_notebook.npz')['arr_0'][0]
            self.logger.debug(f"Testing labels output: {'passed' if np.sum(self.output.get('windowlabels')[0] == test_array) == test_array.shape[0] * test_array.shape[1] else 'failed.'}")
        else:
            self.logger.warning("Labels testfile not found.")


def main(verbose: int = 3):
    
    # create PreProcessor object
    p = PreProcessor(verbose=4)

    # get all solo file names
    files = p.get_filenames()
    files = [f for f in files if 'solo' in f]
    
    # get audio and label file for first filename
    audio, labels = p.load_files(files[0])

    # Preprocess audio and labels
    p.preprocess_audio(audio)
    p.preprocess_labels(labels)

    # save output
    p.save_output()

    # check whether output is correct
    p._test()

if __name__ == '__main__':
    main()