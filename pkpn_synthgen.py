import numpy as np
import matplotlib.pyplot as plt
import logging
import toml
import os

class Wavegen():

    def __init__(self, seed=12345, nb_channels=1, bitrate=14400, monotone=440):
        self._rs = np.random.seed(seed)
        self.bitrate = bitrate  # in Hz
        self.duration = 1       # in seconds
        self.nb_channels=nb_channels
        self.noise = 0
        self.noise_type = None
        self.nb_samples = 0
        self.samples = None
        self.monotone = monotone
        self.outfile = "wavesample.npz"

    @property
    def time(self):
        return self._time
    
    @time.getter
    def time(self):
        self._time = np.linspace(0, self.duration, self.bitrate*self.duration)
        return self._time

    def generate_samples(self, nb_samples=10, outfile="sample"):
        self.samples = np.empty((nb_samples, self.nb_channels, self.bitrate*self.duration))
        for i in range(self.samples.shape[0]):
            src_wave = self._waveform()
            for j in range(self.samples.shape[1]):
                self.samples[i,j,:] = self._add_noise(src_wave)
        return None

    def _waveform(self):
        wave = np.cos(2*np.pi*self.time*self.monotone)
        return wave

    def _add_noise(self, wave):
        
        if self.noise_type:
            pass
        else:
            return wave

    def plot_samples(self, nb_plot=2):
        time = self.time
        for i in range(nb_plot):
            for channel in self.samples[i]:
                plt.plot(time, channel, color=("C"+str(i)))
        return None

    def export_npz(self, outfile=None):
        filename = self.outfile
        if outfile is not None:
            filename = outfile
        if self.samples is None:
            logging.error("self.samples is None; nothing to export")
            return None
        np.savez(filename)


wavg = Wavegen()
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", default=150, type=int, help='epochs')
    parser.add_argument("-f", type=str, default=None, help=" -log10 of learning rate (ex. 2 -> 0.01)")
    wavg.generate_samples()
    wavg.plot_samples(2)
    plt.show()
if __name__ == "__main__":
    main()
