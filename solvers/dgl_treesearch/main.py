import argparse
import logzero

from pathlib import Path
from logzero import logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DGL-based tree search.")
    parser.add_argument("operation", type=str, help="Operation to perform.", choices=["train", "solve"])
    parser.add_argument("input", type=Path, action="store", help="Directory containing input graphs (to be solved/trained on).")
    parser.add_argument("output", type=Path, action="store",  help="Folder in which the output (e.g. json containg statistics and solution will be stored, or trained weights)")

    parser.add_argument("--loglevel", type=str, action="store", default="DEBUG", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Verbosity of logging (DEBUG/INFO/WARNING/ERROR)")

    parser.add_argument("--time_limit", type=int, nargs="?", action="store", default=600, help="Time limit in seconds")
    parser.add_argument("--num_threads", type=int, nargs="?", action="store", default=8, help="Maximum number of threads to use.")
    parser.add_argument("--cuda_devices", type=int, nargs="*", action="store", default=[], help="Which cuda devices should be used (distributed around the threads in round-robin fashion). If not given, CUDA is disabled.")
    parser.add_argument("--self_loops", action="store_true", default=False, help="Enable self loops addition (in input data) for GCN-based model.")
    parser.add_argument("--max_prob_maps", type=int, action="store", default=32, help="Maximum number of probability maps to explore during tree-search (ignored for training).")
    parser.add_argument("--model_prob_maps", type=int, action="store", default=32, help="Number of probability maps the model was/should be trained for.")
    parser.add_argument("--pretrained_weights", type=Path, nargs="?", action="store", help="Pre-trained weights to be used for solving/continuing training.")
    parser.add_argument("--lr", type=float, action="store", default=0.001, help="Learning rate (for training)")
    parser.add_argument("--epochs", type=int, action="store", default=20, help="Number of epochs to train for")
    parser.add_argument("--reduction", action="store_true", default=False, help="Enable reduction of graph (kernelization).")
    parser.add_argument("--local_search", action="store_true", default=False, help="Enable local search if time left.")
    parser.add_argument("--queue_pruning", action="store_true", default=False, help="Enable pruning of queue.")
    parser.add_argument("--weighted", action="store_true", default=False, help="Whether input is weighted or unweighted.")
    parser.add_argument("--noise_as_prob_maps", action="store_true", default=False, help="Use uniform noise instead of GNN output.")
    parser.add_argument("--weighted_queue_pop", action="store_true", default=False, help="Choose element from queue with probability inverse proportional to number of unlabelled vertices in it.")

    args = parser.parse_args()

    ### Set logging ###
    if args.loglevel == "DEBUG":
        logzero.loglevel(logzero.DEBUG)
    elif args.loglevel == "INFO":
        logzero.loglevel(logzero.INFO)
    elif args.loglevel == "WARNING":
        logzero.loglevel(logzero.WARNING)
    elif args.loglevel == "ERROR":
        logzero.loglevel(logzero.ERROR)
    else:
        print(f"Unknown loglevel {args.loglevel}, ignoring.")

    logger.debug(f"DGL Treesearch got launched with arguments {args}")

    # imports are done here because importing DGL is slow
    if args.operation == "solve":
        from solve import solve
        solve(args)
    else:
        from train import train
        train(args)