datamodules = {}

def register(name):
    def decorator(cls):
        datamodules[name] = cls
        return cls
    return decorator

def make(name, config):
    dataset = datamodules[name](config)
    return dataset

from . import dm