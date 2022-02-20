import torch
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import toml_config
import wav_preprocess

logger = logging.getLogger(__name__)

class PkJudge():

    def __init__(self, modeldir, evaldata):
        self.load_dir(modeldir)
        self.evalwavs = wav_preprocess.WavPrep(evaldata, volthreshold=None)
        self.volthreshold = [0.60,]
        #print((self.evalwavs.render_segments()[0]).shape)
        self.xvals, self.groundy = self.evalwavs.render_segments()
        self.predy = np.zeros((self.groundy.shape))
        self.ground_rnd = self.predy.copy()
        self.predy_rnd = self.predy.copy()
    
    def load_dir(self, dirname):
        loaddir = dirname
        if not loaddir.startswith("./"):
            loaddir += "./" 
        if not os.path.isdir(loaddir):
            logger.error(" WavPrep err - directory " + loaddir + " not found ")
            return None
        loaddir = os.path.abspath(loaddir) + "/"
        for file in os.listdir(loaddir):
            if file.endswith(".pth"):
                self.model = torch.load(loaddir+file)
            if file.endswith(".toml"):
                self.model.config = toml_config.TConfig(loaddir+file)
        return None

    def evaluate(self):
        volthreshold = [self.model.config.volthreshold]     # careful
        """print(volthreshold)
        print(self.yvals.shape[-1])
        target = np.expand_dims(self.xvals[0], axis=0)
        print(f" target shape{target.shape}")
        sample = self.model.forward(torch.from_numpy(target).float()).detach().numpy()
        print(sample)"""
        self.predy = self.model.forward(torch.from_numpy(self.xvals).float())
        self.predy = self.predy.detach().numpy()
        print(f" predicting between {np.min(self.predy)} and {np.max(self.predy)}")
        print(f" predy shape {self.predy.shape}")
        if volthreshold is not None:
            groundval = np.zeros(self.groundy.shape)
            predvalue = groundval.copy()
            for level in self.volthreshold:
                groundval += (self.groundy > np.ones(self.groundy.shape)*level )/len(volthreshold)
                predvalue += (self.predy > np.ones(self.groundy.shape)*level )/len(volthreshold)
            self.ground_rnd = groundval
            self.predy_rnd = predvalue
        errs = (self.ground_rnd != self.predy_rnd)
        total = np.sum(errs)
        print(f" {self.groundy[0]} || {self.predy[0]}")
        print(f" {self.ground_rnd[0]} || {self.predy_rnd[0]}")
        logger.info(f" Evaluate.evaluate: {total} miscategorizations in {self.groundy.shape[0]} segments")
        by_segment = (np.sum(errs, axis=2))
        print(f" by_segment.shape {by_segment.shape}")
        print(by_segment[0])
        return self.groundy, self.predy, self.ground_rnd, self.predy_rnd

    def view_segment(self):
        fig, ax = plt.subplots(2,2)
        print("="*8)
        print(f"{self.groundy[0:4,0,:]}")
        ax[0,0].plot(self.xvals[0,0,:])
        ax[0,1].plot(self.xvals[1,0,:])
        ax[1,0].plot(self.xvals[2,0,:])
        ax[1,1].plot(self.xvals[3,0,:])
        plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", type=int, default=3,
        help=("set logging level: 0 critical, 1 error, "
                "2 warning, 3 info, 4 debug, default=info"))
    parser.add_argument("-m", default=None, type=str,
        help="model for evaluation")
    parser.add_argument("-d", default=None, type=int, 
        help='evaluation data')
    config = toml_config.TConfig()
    args = parser.parse_args()

    logging_translate = [logging.CRITICAL, logging.ERROR, logging.WARNING,
                         logging.INFO, logging.DEBUG]
    logger = None
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging_translate[args.v])
    logger = logging.getLogger(__name__)
    goof1 = "./models/2002_0844"
    goof2 = "./midi_conversion/n20_output/"
    #judge = PkJudge(args.m, args.d)
    judge=PkJudge(goof1, goof2)
    judge.view_segment()
    return judge.evaluate()

if __name__ == "__main__":
    ground, pred, grnd, prnd = main()