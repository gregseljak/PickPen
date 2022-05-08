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
        self.duration = 1000 # duration in ms
        self.bpm = 120
        self.max_events_per_file = 2
        self.notedict = {"C":0, "C#":1, "D":2, "D#":3,
            "E":4, "F":5, "F#": 6, "G":7, "G#": 8, "A":9, "Bb":10, "B":11}
        self.csv_header = '''0,0, Header,1,2,480,
1,0, Start_track,,,,
1,0, Time_signature,4,2,24,8
1,0, Tempo,500000,,,
1,0, End_track,,,,
2,0, Start_track,,,,
2,0, Instrument_name_t," ""Guitar""",,,
2,0, Program_c,1,0,,
2, 0.0, note_on_c , 1, 69.0, 1.0
2, 1.0, note_off_c , 1, 69.0, 0.0 \n'''
# Last 2 lines: csv2midi requires a nonzero-volume event to start the recording
        self.csv_footer = '2,' + str(self.duration) + ''', End_track,,,,
            0,0, End_of_file,,,,'''
        self.prange = tconfig.range

    def state_from_tconfig(self, tconfig):
        self.tconfig = tconfig
        self.tconfig.parent = self
        self.tconfig.update_parent()

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
        
        self.yvals = np.empty((self.nb_samples, self.max_events_per_file), dtype = dict)
        logger.info("drawing yvals")
        innerints = (self.duration-499)//500
        for sample in self.yvals:
            nbevents = self._rs.randint(0, self.max_events_per_file+1)
            for i in range(len(sample)):
                if i <= nbevents:
                    time0 = self._rs.randint(0,innerints+1)*500 + 250 + int(self._rs.normal(0,5))
                    time1 = min(time0 + 500, self.duration) + int(self._rs.normal(0,5))
                    pitch = self._rs.randint(self.range[0],self.range[1]+1)
                    sample[i] = {"time0":time0, "time1":time1, "pitch":pitch, "vol":99}
                else:
                    sample[i] = {"time0":-1, "time1":-1, "pitch":-1, "vol":-1}
        
    def generate_scores(self):
        """ This is quite brutal but the csvmidi software requires that the events be chronologically"""
        self._score = []
        for i in range(self.nb_samples):
            nb_events = len(self.yvals[i])
            scoreframe = np.zeros((nb_events*2, 3))
            for j in range(nb_events):
                event = self.yvals[i,j]
                if event["time0"] == -1:
                    continue
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
                duration = self.duration_string()
                subprocess.run(["midi_conversion/midicsv11/csv2wav.sh", outdir, filename, duration])
        np.savez(absolute_outdir + 'yvals', yvals = self.yvals)
        #self.tconfig.update_tconfig()
        self.tconfig.save_toml(absolute_outdir)

    def midi_idx(self, notename):
        octave = notename[-1]
        noteval = self.notedict[notename[:-1]]
        return 12*octave + noteval

    def duration_string(self):
        """takes duration in ms (self.duration)
            and returns a string in {hh:mm:ss.ms} fmt"""
        timecount = self.duration
        hours = timecount//(60*60*1000)
        timecount -= hours*60*60*1000
        minutes = timecount//(60*1000)
        timecount -= minutes*60*1000
        seconds = timecount//1000
        timecount -= seconds*1000
        hrmnsc = (str(hours), str(minutes), str(seconds))
        drstr = ""
        for unit in hrmnsc:
            while len(unit) < 2:
                unit = "0" + unit
            drstr += (unit +":")
        drstr = drstr[:-1] + "." + str(timecount)
        return drstr

logger = logging.getLogger(__name__)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    # =====================
    # Parse arguments. metavar is the name of the corresponding TConfig variable name
    parser = argparse.ArgumentParser(description=("Create synthetic samples"))
    parser.add_argument("-n", type=int, metavar="nb_samples",
                        help="(int) the  number of samples")
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
    print(f"wavg.nb_samples {wavg.nb_samples}")
    wavg.draw_yvals()
    wavg.generate_scores()
    export_location = "diphone_fullrange"
    wavg.export_csv(export_location)
    print(f"finished conversion: {export_location}")
if __name__ == "__main__":
    main()

