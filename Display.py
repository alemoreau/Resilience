from Parameters import *

class Display:
    "General display"
    def __init__(self, displays={}):
        self.displays = displays

    def set_display(self, name, display):
        self.displays[name] = display
        
    def __call__(self, name, data, parameters = Parameters({})):
        if not name in self.displays:
            print "No ", name, " found in display"
            return
        self.displays[name](data, parameters)
