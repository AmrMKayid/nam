import torch


class Model(torch.nn.Module):

    def __init__(self, config, name):
        super(Model, self).__init__()
        self._config = config
        self._name = name

    def forward(self):
        raise NotImplementedError

    def __str__(self):
        return f'{self.__class__.__name__}(name={self._name})'

    @property
    def config(self):
        return self._config

    @property
    def name(self):
        return self._name
