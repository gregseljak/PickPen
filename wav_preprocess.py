import numpy as np
import os
import matplotlib.pyplot as plt
import toml_config
from scipy.io import wavfile
import logging 

logger = logging.getLogger(__name__)

class WavPrep():
    """
    Load samples and treat them as needed before handing off to pkpn_train
    """
    def __init__(self, loaddir = None):
        
        if loaddir is not None:
            self.load(loaddir)
        self._rs = np.random.RandomState(self.seed)

    def load(self, indir):
        
        loaddir = indir
        if not (indir.startswith("./")):
           loaddir += "./" 
        if not (os.path.isdir(loaddir)):
            logger.error(" WavPrep err - directory " + loaddir + " not found ")
            return None
        abs_indir = os.path.abspath(indir) + "/"

        nbs = 0
        for file in os.listdir(abs_indir):
            if file.endswith(".wav"):
                if nbs == 0:
                    nbb = ((wavfile.read(abs_indir+file))[1]).shape
                    if len(nbb) == 1:
                        nbc = 1
                    else:
                        nbc = nbb[0]
                    nbb = nbb[-1]
                nbs += 1

        self.xvals = np.empty((nbs, nbc, nbb))
       
        for file in os.listdir(abs_indir):
            if file.endswith(".toml"):
                self.tconfig = toml_config.TConfig(abs_indir + file, "pkpn_synthgen")
                for section in self.tconfig.dict:
                    for item in self.tconfig.dict[section]:
                        setattr(self, str(item), getattr(self.tconfig, str(item)))
                self.tconfig.parent = self

            elif file.endswith(".npz"):
                self.yvals = np.load(abs_indir+file)
            elif file.endswith(".wav"):
                delimit1 = -1*file[::-1].find("_")
                sample_idx = int(file[delimit1: -4])
                bitrate, data = wavfile.read(abs_indir+file)
                self.xvals[sample_idx, :,:] = data
                logger.info(" loaded " + file + f" to xval[{ sample_idx }]")
            else:
                logger.info(" file " + file + " not loaded; unrecognized name convention")

    def fnote(self, note):
        """ equal temper
        # accepts midi note value ex fnote(69); return 440.0
        # or equivalently snn     ex fnote("A4"); return 440.0
        """
        halftone_idx = note
        if isinstance(note, str):
            halftone_idx = self.midi_idx(note)
        return self.a4freq * 2 **((halftone_idx - 69)/12) 

    def plot_samples(self, nb_plot=2):
        time = self.time
        for i in range(nb_plot):
            for channel in self.samples[i]:
                plt.plot(time, channel, color=("C"+str(i)), label=self.yvals[i])
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", type=int, default=3,
    help=("set logging level: 0 critical, 1 error, "
            "2 warning, 3 info, 4 debug, default=info"))
    logging_translate = [logging.CRITICAL, logging.ERROR, logging.WARNING,
                         logging.INFO, logging.DEBUG]
    parser.add_argument("-i", default=None, type=str,
        help="training/validation dataset")
    logger = None
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging_translate[args.v])
    logger = logging.getLogger(__name__)


    prep = WavPrep(args.i)
    
if __name__ == "__main__":
    main()