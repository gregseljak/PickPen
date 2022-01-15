import numpy as np
import matplotlib.pyplot as plt
import logging
import toml
import os
import datetime
#import model_library
import toml_config

class Wavegen():

    def __init__(self, tconfig):
        self.score = tconfig.score
        self._rs = np.random.RandomState(tconfig.seed)
        self.bitrate = tconfig.bitrate    # in Hz
        self.duration = tconfig.duration  # in seconds
        self.nb_channels = tconfig.nb_channels
        self.noise = tconfig.noise
        self.noisetype = tconfig.noisetype
        self.nb_samples = tconfig.nb_samples
        self.samples = None
        self.monotone = tconfig.monotone
        self.outdir = "wavesample"
        self.yvals = None
        self.tconfig = None

    @property
    def time(self):
        return self._time
    
    @time.getter
    def time(self):
        self._time = np.linspace(0, self.duration, self.bitrate*self.duration)
        return self._time

    def draw_yvals(self):
        logger.info("drawing yvals")
        self.yvals = self._rs.uniform(0,1,self.nb_samples)

    def generate_samples(self, nb_samples=10):
        if self.yvals is None:
            logger.error("generate_samples() called with no set yvals")
            return None
        self.samples = np.empty((nb_samples, self.nb_channels, self.bitrate*self.duration))
        for i in range(self.samples.shape[0]):
            freq = self.yvals[i] * 440
            src_wave = self._waveform(freq)
            for j in range(self.samples.shape[1]):
                self.samples[i,j,:] = self._add_noise(src_wave)
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

    def plot_samples(self, nb_plot=2):
        time = self.time
        for i in range(nb_plot):
            for channel in self.samples[i]:
                plt.plot(time, channel, color=("C"+str(i)))
        return None

    """def export_npz(self, outdir=None):
        # under construction
        outdir = self.outdir
        if outdir is not None:
            filename = outdir
        if self.samples is None:
            logging.error("self.samples is None; nothing to export")
            return None

        os.mkdir(filename)
        np.savez(self.samples, "./" + filename + "/" + filename)
        np.save(self.yvals, "./" + filename + "/Y_" + filename)

        outdir = config.fname_out
        now = datetime.now()
        dt_string = now.strftime("_%dd%mm_%HH%MM")
        while (os.path.isdir(outdir)):
            dt_string = now.strftime("_%dd%mm_%HH%MM")
        outdir += dt_string
        os.mkdir(outdir)
        np.savez("./" + outdir + "/" + outdir, self.samples,)
        np.savez(outdir+ "/"+ companion_name, loss=epoch_loss, ste=epoch_ste, lr=config.lr, lor_fnm = config.fname_in)
        config.save_toml(outdir+"/"+outdir+".toml")
        logger.info(f" Dumped to {outdir}")
        """
logger = logging.getLogger(__name__)
def main():
    import argparse

    # =====================
    # Parse arguments. metavar is the name of the corresponding TConfig variable name
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
    wavg.plot_samples(2)
    plt.show()
if __name__ == "__main__":
    main()
