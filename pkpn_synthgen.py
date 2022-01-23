import numpy as np
import matplotlib.pyplot as plt
import logging
import toml
import os
from datetime import datetime
#import model_library
import toml_config
import subprocess

c_scale = np.array([261.63, 293.66, 329.63, 349.23, 392.00,
440.00, 493.88, 523.25]) # from c4 to c5,  inclusive

class Wavegen():

    def __init__(self, tconfig):
        """ meant to be initialized using a TConfig object from toml_config.py """
        self.state_from_tconfig(tconfig)
        self.a4freq = 440                # in Hz; industry standard
        self._rs = np.random.RandomState(self.seed)
        self._scores = None
        self._yvals = None # 2d: [sample, atk volume]
        self.duration = 2000 # duration in ms
        self.bpm = 120
        self.nb_intervals = 4  #= self.bpm # intervals per sample
        self.notedict = {"C":0, "C#":1, "D":2, "D#":3,
            "E":4, "F":5, "F#": 6, "G":7, "G#": 8, "A":9, "Bb":10, "B":11}
        self.csv_header = '''0,0, Header,1,2,480,
1,0, Start_track,,,,
1,0, Time_signature,4,2,24,8
1,0, Tempo,500000,,,
1,0, End_track,,,,
2,0, Start_track,,,,
2,0, Instrument_name_t," ""Guitar""",,,
2,0, Program_c,1,0,,\n'''
        self.csv_footer = '2,' + str(self.duration) + ''', End_track,,,,
            0,0, End_of_file,,,,'''
        self.prange = tconfig.range

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
        tconfig.parent = self


    @property
    def yvals(self):
        return self._yvals
    
    @yvals.setter
    def yvals(self, value):
        self._yvals = value
        if self.nb_samples != self._yvals.shape[0]:
            logger.warning(f" Wavegen nb_samples = {self.nb_samples} != yvals.shape[0] = {self._yvals.shape[0]}")
            self.nb_samples = self._yvals.shape[0]
            logger.warning(f" Wavegen nb_samples has been updated to {self.nb_samples}")
        self.nb_samples = self._yvals.shape[0]

    def draw_yvals(self):
        logger.info("drawing yvals")
        self.yvals = np.zeros((self.nb_samples, self.nb_intervals, self.prange[-1] - self.prange[0] + 1))
        for sample in self.yvals:
            for interval in range(self.nb_intervals):
                pitch = int(self._rs.uniform(self.prange[0], self.prange[1]+1)) - self.range[0]
                sample[interval, pitch] = 1
        
    def generate_scores(self):
        self._score = ["" for _ in range(self.nb_samples)]
        interval_len = int(self.duration/self.nb_intervals)
        time = np.arange(0,self.duration + interval_len, interval_len)
        print(f"len(time) {len(time)}")
        for i in range(self.nb_samples):
            for t in range(len(time)-1):
                for p in range(len(self.yvals[i,t])):
                    pitch = self.prange[0] + p
                    if self.yvals[i][t][p] > 0:
                        self._score[i] += "2, " + str((time[t]))   + ", note_on_c , 1, " + str(pitch) + ", 81" + "\n"
                        self._score[i] += "2, " + str((time[t+1])) + ", note_on_c , 1, " + str(pitch) + ", 0"  + "\n"
                        logger.info(f" Sample {i} : {pitch} at {time[t]}")

    def export_csv(self, outdir, towav=True):
        absolute_outdir = "/home/greg/PickPen/midi_conversion/"+ outdir + "/"
        os.mkdir(absolute_outdir)
        for i in range(self.nb_samples):
            fullscore = self.csv_header + self._score[i] + self.csv_footer
            filename = "sample_" + str(i)
            output = open(absolute_outdir + filename +".csv", "w")
            output.write(fullscore)
            output.close()
            with open(absolute_outdir + 'yvals', 'wb') as f:
                np.savez(f, self.yvals)
            self.tconfig.update_tconfig()
            self.tconfig.save_toml(outdir)
            if towav:
                subprocess.run(["midi_conversion/midicsv11/csv2wav.sh", outdir, filename])

    def midi_idx(self, notename):
        octave = notename[-1]
        noteval = self.notedict[notename[:-1]]
        return 12*octave + noteval

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
    wavg.nb_samples = 4
    wavg.draw_yvals()
    print(wavg.yvals)
    wavg.generate_scores()
    wavg.export_csv("output")

if __name__ == "__main__":
    main()

