import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import random
import time

from model import HindsightLoss
from utils import _load_model

from tqdm import tqdm
from logzero import logger

def train(args):
    self_loop = args.self_loops
    cuda = bool(args.cuda_devices)
    prob_maps = args.model_prob_maps
    
    if cuda:
        cuda_dev = args.cuda_devices[0]
        if len(args.cuda_devices) > 1:
            logger.warn("More than one cuda device was provided, using " + cuda_dev)
    else:
        cuda_dev = None

    training_graphs = []

    dgl_graphs = dgl.load_graphs(str(args.input / f"graphs_{'weighted' if args.weighted else 'unweighted'}.dgl"))[0]
    random.shuffle(dgl_graphs)

    logger.info("Loading training graphs.")
    for g in tqdm(dgl_graphs):
        if cuda:
            g = g.to(cuda_dev)

        if self_loop:
            g = dgl.remove_self_loop(g)
            g = dgl.add_self_loop(g)
        else:
            g = dgl.remove_self_loop(g)

        training_graphs.append(g)

    logger.info(f"Loaded {len(training_graphs)} graphs for training.")

    model = _load_model(prob_maps, weight_file=args.pretrained_weights, cuda_dev=cuda_dev)
    loss_fcn = HindsightLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=5e-4)

    num_epochs = args.epochs
    status_update_every = max(1, int(0.1 * len(training_graphs)))
    
    for epoch in range(num_epochs + 1):
        logger.info(f"Epoch {epoch}/{num_epochs}")
        epoch_losses = list()
        for gidx, graph in enumerate(tqdm(training_graphs)):
            features = graph.ndata['weight']
            labels = graph.ndata['label']
            model.train()
            # forward
            output = model(graph, features)
            loss = loss_fcn(output, labels)
            epoch_losses.append(float(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if gidx % status_update_every == 0:
                logger.info(f"Epoch {epoch}/{num_epochs}, Graph {gidx}/{len(training_graphs)}: Average Epoch Loss = {np.mean(epoch_losses)}, Last Training Loss = {loss}")

        torch.save(model.state_dict(), args.output / f"{int(time.time())}_intermediate_model{prob_maps}_{epoch}_{np.mean(epoch_losses):.2f}.torch")

    logger.info(f"Final: Average Epoch Loss = {np.mean(epoch_losses)}, Last Training Loss = {loss}")
    torch.save(model.state_dict(), args.output / f"{int(time.time())}_final_model{prob_maps}.torch")
