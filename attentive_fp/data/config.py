
#COnfguration is just a definned dictionnary with a structure
from features import ATOM_CLASSES,BOND_CLASSES


PARAM_TEMPLATE = {
    "canonical":{"type":bool,"value":True},
    "atom":{"type":str,"value":"intermediate","values":ATOM_CLASSES.keys()},
    "bond":{"type":str,"value":"intermediate","values":BOND_CLASSES.keys()}
}

class Configuration:
    def __init__(self):
        pass

    
    def _check(param,value):
        if not isinstance(param,PARAM_TEMPLATE["type"]):
            raise ValueError("Parameter {} should be of type {}.".format(param,PARAM_TEMPLATE[param]["type"]))
        if "values" in PARAM_TEMPLATE[param] and value not in PARAM_TEMPLATE[param]["values"]:
            raise ValueError("Parameter {} should be of type.".format(PARAM_TEMPLATE[param]["type"]))

    def __setitem__(self,item,values):

        