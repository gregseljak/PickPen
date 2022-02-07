import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import toml_config
from scipy.io import wavfile
import seglearn

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

        nbs = 0 # number of sample files
        for file in os.listdir(abs_indir):
            if file.endswith(".wav"):
                if nbs == 0:
                    nbb = ((wavfile.read(abs_indir+file))[1]).shape # np.ndims() changes based on whether the file is mono or stereo
                    if len(nbb) == 1:
                        nbc = 1 # number of channels
                    else:
                        nbc = nbb[0]
                    nbb = nbb[-1]   # number of bits per channel
                nbs += 1

        self.xvals = np.empty((nbs, nbc, nbb)) # samples, channels, bits; pytorch convention
       
        for file in os.listdir(abs_indir):
            if file.endswith(".toml"):
                self.tconfig = toml_config.TConfig(abs_indir + file, "pkpn_synthgen")
                for section in self.tconfig.dict:
                    for item in self.tconfig.dict[section]:
                        setattr(self, str(item), getattr(self.tconfig, str(item)))
                self.tconfig.parent = self

            elif file.endswith(".npz"):
                self.yvals = (np.load(abs_indir+file, allow_pickle=True))["yvals.npy"]
            elif file.endswith(".wav"):
                delimit1 = -1*file[::-1].find("_")
                sample_idx = int(file[delimit1: -4])
                bitrate, data = wavfile.read(abs_indir+file)
                self.xvals[sample_idx, :,:] = data
                logger.debug(" loaded " + file + f" to xval[{ sample_idx }]")
                if bitrate != self.bitrate:
                    logger.warning(f" Directory .toml file bitrate [{self.bitrate}] differs from {file} metadata bitrate [{bitrate}] ")
            else:
                logger.debug(" file " + file + " not loaded; unrecognized name convention")
            

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

    def _segment(self, start_bit, nb_bits, listen_window):
        """
        returns 1) xvals: the x values from [start_bit, start_bit+nb_bits] for each sample
                2) yvals: a translated list of events in the score that fall within the listen subinterval
                            returned in relative midi matrix format
                ex// all the bits   from 22050b to 44100b (0.50s to 1.00s)
                     all the events from  0.75s to 0.875s
        """
        nb_notes = self.range[1] - self.range[0] + 1
        t0 = (start_bit + listen_window[0]) / self.bitrate * 1000
        tf = (start_bit + listen_window[1]) / self.bitrate * 1000   # in milliseconds
#        print(f" t0, tf {(t0, tf)}")
        yvals = np.zeros((self.nb_samples, nb_notes))
        xvals = np.zeros((self.nb_samples, 1, nb_bits))
        for i in range(self.nb_samples):
            xvals[i,0,:] = self.xvals[i,0,start_bit:start_bit+nb_bits]
            for event in self.yvals[i,:]:
                if (event["time0"] >= t0 and event["time0"] < tf):
                    yvals[i, (event["pitch"] - self.range[0])] = event["vol"]
        return xvals, yvals

    def render_segments(self, window_bits = 44100*0.50, sensitivity=44100*0.125, center = 0.5):
        """ returns arrays of segmented subsamples from the wavs """
        bit0 = int(center*window_bits - sensitivity)
        bitf = int(center*window_bits + sensitivity)
        if (bitf > window_bits) or (bit0 < 0):
            logging.warning(f""" {self.__str__}. render_segments() sensitivity window {bit0}:{bitf}
                exceeds segment window {window_bits}; \n Defaulting to segment boundary """)
            if bitf > window_bits :
                bitf = window_bits - 1
            if bit0 < 0:
                bit0 = 0
        nb_segs = int(((self.xvals.shape[2])//window_bits))
        self.nb_samples = len(self.xvals)
        window = int(window_bits)
        xvals = np.zeros((int(nb_segs*self.nb_samples),1, window))
        yvals = np.zeros((int(nb_segs*self.nb_samples), self.range[1] - self.range[0] + 1))
        for j in range(nb_segs):
            xvals[j:: nb_segs], yvals[j::nb_segs,:] = self._segment(int(window)*j, window, (bit0,bitf))
        print(np.max(yvals))
        yvals = np.expand_dims(yvals, axis=1)
        yvals = yvals/np.max(yvals)
        return xvals, yvals
        
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
    x, y = prep.render_segments()
    print(x.shape)
    print(y.shape)
if __name__ == "__main__":
    main()
