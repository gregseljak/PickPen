import numpy as np
import matplotlib.pyplot as plt
import logging
import toml
import os
from datetime import datetime
import pandas as pd
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
        self.grain = 0.25  # quarter note
        self.nb_events_per_grain = 4
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
        
        self.yvals = np.empty((self.nb_samples, self.nb_events_per_grain), dtype = dict)
        logger.info("drawing yvals")
        prange = self.prange[-1] - self.prange[0] + 1
        for sample in self.yvals:
            for i in range(len(sample)):
                time0 = int(i)*500 
                time1 = time0 + 500 #+ int(np._rs.gaussian(0,5))
                pitch = int(self._rs.uniform(prange) + self.range[0]) 
                sample[i] = {"time0":time0, "time1":time1, "pitch":pitch, "vol":81}
        
    def generate_scores(self):
        """ This is quite brutal but the csvmidi software requires that the events be chronologically"""
        self._score = []
        for i in range(self.nb_samples):
            nb_events = len(self.yvals[i])
            scoreframe = np.zeros((nb_events*2, 3))
            for j in range(nb_events):
                event = self.yvals[i,j]
                scoreframe[int(2*j)] = np.array([event["time0"], event["pitch"], event["vol"]]) # when to turn on
                scoreframe[int(2*j+1)] = np.array([event["time1"], event["pitch"], 0])          # when to turn off
            scoreframe = scoreframe[scoreframe[:,0].argsort()] # chronological
            samplescore = ""
            for event in scoreframe:
                if event[-1] == 0: # "off" event
                    samplescore += "2, " + str(event[0]) + ", note_off_c , 1, " + str(event[1]) + ", 0"  + "\n"
                else:              # "on" event
                    samplescore += "2, " + str(event[0])   + ", note_on_c , 1, " + str(event[1]) + ", " + str(event[2]) + "\n"
            self._score.append(samplescore)

    def export_csv(self, outdir, towav=True):
        absolute_outdir = "/home/greg/PickPen/midi_conversion/"+ outdir + "/"
        os.mkdir(absolute_outdir)
        for i in range(self.nb_samples):
            fullscore = self.csv_header + self._score[i] + self.csv_footer
            filename = "sample_" + str(i)
            with open(absolute_outdir + filename +".csv", "w") as output:
                output.write(fullscore)
                output.close()
            if towav:
                subprocess.run(["midi_conversion/midicsv11/csv2wav.sh", outdir, filename])
        np.savez(absolute_outdir + 'yvals', yvals = self.yvals)
        #self.tconfig.update_tconfig()
        self.tconfig.save_toml(absolute_outdir)

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
    wavg.nb_samples = 10
    wavg.draw_yvals()
    wavg.generate_scores()
    wavg.export_csv("n10_output")

if __name__ == "__main__":
    main()

