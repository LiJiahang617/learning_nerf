import os
import importlib.util


def make_network(cfg):
    spec = importlib.util.spec_from_file_location(cfg.network_module, cfg.network_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    network = module.Network()
    return network
