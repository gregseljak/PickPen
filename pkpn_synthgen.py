import numpy as np
import matplotlib.pyplot as plt
import logging
import toml
import os
from datetime import datetime
#import model_library
import toml_config

class Wavegen():

    def __init__(self, tconfig):
        """ meant to be initialized using a TConfig object from toml_config.py """
        self.state_from_tconfig(tconfig)
        self.a4freq = 440                # in Hz; industry standard
        self._rs = np.random.RandomState(self.seed)
        self.samples = None
        self._yvals = None
        self.keydict = {"A4":0, "Bb4":1, "B4":2, "C4":3, "C#4":4, "D4":5, "D#":6,
        "E4":7, "F4":8, "F#4":9, "G4":10,"G#4":11, "A5":12}

    def state_from_tconfig(self, tconfig):
        """ inherited from configuration file:
        self.tconfig = tconfig
        self.seed = tconfig.seed
        self.score = tconfig.score
        self.bitrate = tconfig.bitrate    # in Hz
        self.duration = tconfig.duration  # in seconds
        self.nb_channels = tconfig.nb_channels
        self.noise = tconfig.noise
        self.noisetype = tconfig.noisetype
        self.nb_samples = tconfig.nb_samples
        self.monotone = tconfig.monotone """
        self.tconfig = tconfig
        for section in self.tconfig.dict:
            for item in self.tconfig.dict[section]:
                setattr(self, str(item), getattr(tconfig, str(item)))

    @property
    def time(self):
        return self._time
    
    @time.getter
    def time(self):
        self._time = np.linspace(0, self.duration, self.bitrate*self.duration)
        return self._time

    @property
    def yvals(self):
        return self._yvals
    
    @yvals.setter
    def yvals(self, value):
        self._yvals = value
        self.nb_samples = self._yvals.shape[0]

    def draw_yvals(self):
        logger.info("drawing yvals")
        #self.yvals = self._rs.uniform(0,1,self.nb_samples)
        self.yvals = np.arange(self.range[0], self.range[1], 1)

    def generate_samples(self):
        if self.yvals is None:
            logger.error("generate_samples() called with no set yvals")
            return None
        self.samples = np.empty((self.nb_samples, self.nb_channels, self.bitrate*self.duration))
        for i in range(self.samples.shape[0]):
            freq = self.yvals[i] * 440
            src_wave = self._waveform(freq)
            for j in range(self.samples.shape[1]):
                self.samples[i,j,:] = self._add_noise(src_wave)
        self.samples = self.samples/np.max(self.samples)
        return self.samples

    def _waveform(self, freq):
        wave = np.cos(2*np.pi*self.time*freq)
        return wave

    def _add_noise(self, wave):
        bitlen = wave.shape[-1]
        if self.noisetype == "normal":
            return wave + self._rs.normal(0,self.noise, bitlen)
        else:
            return wave
    
    def fnote(self, halftone):
        # equal temper
        halftone_idx = halftone
        if isinstance(halftone, str):
            halftone_idx = self.keydict[halftone]
        return self.a4freq * 2 **(halftone_idx/12)

    def plot_samples(self, nb_plot=2):
        time = self.time
        for i in range(nb_plot):
            for channel in self.samples[i]:
                plt.plot(time, channel, color=("C"+str(i)), label=self.yvals[i])
        return None

    def export_npz(self, comment=None):
        
        if self.samples is None:
            logging.error("self.samples is None; nothing to export")
            return None
        outdir = "cos_n" + str(self.nb_samples)
        if comment:
            outdir += comment
        os.mkdir(outdir)
        np.savez("./" + outdir + "/xdata.npz", self.samples)
        np.save("./" + outdir + "/ydata.txt", self.yvals)
        self.tconfig.update_tconfig(self)
        self.tconfig.save_toml("./"+outdir+"/config.toml")
        logger.info(f" Dumped to {outdir}")

c_scale = np.array([261.63, 293.66, 329.63, 349.23, 392.00,
440.00, 493.88, 523.25]) # from c4 to c5,  inclusive

logger = logging.getLogger(__name__)
def main():
    import argparse
    parser = argparse.ArgumentParser()
    # =====================
    # Parse arguments. metavar is the name of the corresponding TConfig variable name
    parser.add_argument("-n", type=int, metavar="nb_samples",
                           help="(int) the  number of samples")
    parser = argparse.ArgumentParser(description=("Create synthetic samples"))
    parser.add_argument("-v", type=int, default=3,
                        help=("set logging level: 0 critical, 1 error, "
                              "2 warning, 3 info, 4 debug, default=info"))
    parser.add_argument("-t", type=str, default=None,
                        help="See toml_config.py")
    args = parser.parse_args()
    logging_translate = [logging.CRITICAL, logging.ERROR, logging.WARNING,
                         logging.INFO, logging.DEBUG]
    logger = None
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging_translate[args.v])
    logger = logging.getLogger(__name__)

    # Load the configuration file, and override with terminal commands
    # These will also write to the config file saved with the model
    config = toml_config.TConfig()
    if args.t:
        config = toml_config.TConfig(args.t)
        config.translate_args(parser)
        logger.info(f" args: {args.t}")

    wavg = Wavegen(config)
    logger.debug(config.summary_str())
    wavg.draw_yvals()
    wavg.generate_samples()
    wavg.export_npz()
    #plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

