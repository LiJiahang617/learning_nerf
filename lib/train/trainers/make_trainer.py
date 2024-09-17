from .trainer import Trainer
import importlib.util


def _wrapper_factory(cfg, network, train_loader=None):
    # 使用 importlib 加载模块
    spec = importlib.util.spec_from_file_location(cfg.loss_module, cfg.loss_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # 实例化loss中的 NetworkWrapper 类
    network_wrapper = module.NetworkWrapper(network, train_loader)
    return network_wrapper

def make_trainer(cfg, network, train_loader=None):
    network = _wrapper_factory(cfg, network, train_loader)
    return Trainer(network)
