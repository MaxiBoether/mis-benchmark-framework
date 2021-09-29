import numpy as np
import uuid
import torch
import torch.nn.functional as F

from dgl.distributed import shared_mem_utils

from model import GCN
import importlib
from logzero import logger


def find_module(full_module_name):
    """
    Returns module object if module `full_module_name` can be imported. 

    Returns None if module does not exist. 

    Exception is raised if (existing) module raises exception during its import.
    """
    try:
        return importlib.import_module(full_module_name)
    except ImportError as exc:
        if not (full_module_name + '.').startswith(exc.name + '.'):
            raise

def _load_model(prob_maps, weight_file=None, cuda_dev=None):
    model = GCN(1, # 1 input feature - the weight
                32, #32 dimensions in hidden layers
                prob_maps,  #probability maps
                20, #20 hidden layers
                F.relu,
                0)

    if cuda_dev is not None:
        model = model.to(cuda_dev)

    if weight_file:
        if cuda_dev:
            model.load_state_dict(torch.load(weight_file))
        else:
            model.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))

    return model

def _locked_log(lock, msg, loglevel):
    if lock:
        if loglevel == "DEBUG":
            with lock:
                logger.debug(msg)
        elif loglevel == "INFO":
            with lock:
                logger.info(msg)          
        elif loglevel == "WARN":
            with lock:
                logger.warn(msg)
        elif loglevel == "ERROR":
            with lock:
                logger.error(msg)
        else:
            with lock:
                logger.error(f"The following message was logged with unknown log-level {loglevel}:\n{msg}")
    else:
        if loglevel == "DEBUG":
            logger.debug(msg)
        elif loglevel == "INFO":
            logger.info(msg)          
        elif loglevel == "WARN":
            logger.warn(msg)
        elif loglevel == "ERROR":
            logger.error(msg)
        else:
            logger.error(f"The following message was logged with unknown log-level {loglevel}:\n{msg}")

### Currently not used ###

def _copy_graph_to_shared_mem(g, graph_name = None):
    if graph_name is None:
        graph_name = str(uuid.uuid4())

    new_g = g.shared_memory(graph_name, formats='coo')
    # We should share the node/edge data to the client explicitly instead of putting them
    # in the KVStore because some of the node/edge data may be duplicated.
    new_g.ndata['ts_label'] = shared_mem_utils._to_shared_mem(g.ndata['ts_label'],
                                       shared_mem_utils._get_ndata_path(graph_name, 'ts_label'))

    new_g.ndata['id_map'] = shared_mem_utils._to_shared_mem(g.ndata['id_map'],
                                       shared_mem_utils._get_ndata_path(graph_name, 'id_map'))

    new_g.ndata['weight'] = shared_mem_utils._to_shared_mem(g.ndata['weight'],
                                       shared_mem_utils._get_ndata_path(graph_name, 'weight'))

    return new_g

def _simple_sat_is_validation(G):
    literals = np.unique(G.ndata['literal'][(G.ndata['ts_label'].detach().squeeze(1) == 1)].numpy())

    for literal in literals:
        if -literal in literals:
            print(f"Oh no! We have both {literal} and {-literal} in our assignment!")
    return literals
    
def _locked_print(lock, msg):
    if lock:
        with lock:
            print(msg)
    else:
        print(msg)