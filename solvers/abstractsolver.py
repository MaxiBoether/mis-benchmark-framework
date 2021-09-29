from abc import ABC, abstractmethod, abstractstaticmethod
import pathlib

class MWISSolver(ABC):

    @abstractmethod
    def load_weights(self, model_state_path):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def directory(self):
        pass

    @abstractstaticmethod
    def _prepare_instance(source_instance_file, cache_directory, **kwargs):
        pass

    @classmethod
    def _prepare_instances(C, instance_directory: pathlib.Path, cache_directory: pathlib.Path, **kwargs):
        for graph_path in instance_directory.rglob("*.gpickle"):
            C._prepare_instance(graph_path.resolve(), cache_directory, **kwargs)

    @abstractmethod
    def train(self, train_data_path: pathlib.Path, results_path: pathlib.Path, parameters):
	    pass

    @abstractmethod
    def solve(self, solve_data_path: pathlib.Path, results_path: pathlib.Path, parameters):
	    pass