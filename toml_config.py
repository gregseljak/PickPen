#!/bin/env python3
import toml
import os
if __name__ != "__main__":
    import __main__ as main



CONF_LEARN = {
    'general': {
        'fname_in': ("./cscale.cos", 'Number of grid points in the horizontal direction.'),
        'model_name': "starter",
        'combine_channels': "true, false: degenerate channels?",
    },
    'hyperparameters': {
        'kernel_size': (41, 'first convolutional kernel size'),
        'lr': (0.01, 'Learning Rate'),
        'epochs': (100),
        'split': (0.8),
        'batch_size': (64),
    },
}

CONF_MAKE = {
    "general" : { 
        "score" : 12345,
        "nb_samples" : 256,
        "nb_channels" : 1,
        "duration" : 3000,
        "bitrate" : 14400,
        "seed": 12345,
    },
    "variations" : {
        "range" : [68, 81],
        "y_2" : 0,
        "noise"       : 0.05,
        "noisetype" : "normal",
        "monotone" : True,
    },
}



class TConfig():
    def __init__(self, parfile=False, filetype=None):
        self.configs = {"pkpn_train" : CONF_LEARN, #default
            "pkpn_synthgen" : CONF_MAKE,
            }
        if filetype != None:
            self.filetype = filetype
        else:
            try:        
                self.filetype = str(os.path.basename(main.__file__))
            except AttributeError:
                self.filetype="pkpn_train"
            if self.filetype.endswith(".py"):
                self.filetype = self.filetype[:-3]
            if self.filetype not in self.configs.keys():
                self.filetype = "pkpn_train"
        if parfile:
            self.dict = toml.load(parfile)
        else:
            self.dict = self.configs[self.filetype]
        self._update_members()
        self.parent=None

    def _update_members(self):
        """Read a TOML par file and update tconfig attributes accordingly."""
 
        for section, opts in self.dict.items():
            if section not in self.configs[self.filetype]:
                print('Unknown section {}.'.format(section))
                continue
            for opt, val in opts.items():
                if opt not in self.configs[self.filetype][section]:
                    print('Unknown option {}.{}.'.format(section, opt))
                    continue
                setattr(self, opt, val)
    
    def translate_args(self, parser):
        import argparse
        """
        # Example implementation:
        config = TConfig("default.toml")
        parser = argparse.ArgumentParser(description=("Accept override args"))
        parser.add_argument("-i", type=str, metavar="input",
                        help=("We want a simple argparse target of -'i'"
                        "but in the toml file the variable is called 'input'"
                        "So set metavar='input'")
        config.translate_args(parser)
        """
        dicto, myargs = {}, []
        args=parser.parse_args()
        for action in parser._actions:
            if ((type(action) is argparse._StoreAction) and action.metavar != None and action.dest != None):
                dicto[action.dest] = [action.metavar]       # populate a args.__dict__ : config.__dict__ dictionary
            if ((action.metavar != None) and (args.__dict__[action.dest] != None)):
                self.__dict__[action.metavar] = args.__dict__[action.dest] # update config.__dict__ values
        return dicto

    def summary_str(self):
        """" debugging """
        contents = ""
        for item, value in self.__dict__.items():
            if item != "dict":
                contents += (f" {item} :: {value} \n")
        return contents
        
    def update_tconfig(self):
        """ scan target for matching attribute names and update """
        for section in self.dict:
            for item in self.dict[section]:
                if hasattr(self.parent, str(item)):
                    setattr(self, str(item), getattr(self.parent, str(item)))

    def save_toml(self, directory="./", filename = "config.toml"):
        """updates the state dictionary to match attributes, then dumps it"""

        for section in self.dict:   
            for item in self.dict[section]:
                self.dict[section][item] = getattr(self, item)
        if not directory.endswith("/"):
            directory += "/"
        with open(directory + filename, "w") as toml_file:
            toml.dump(self.dict, toml_file)

