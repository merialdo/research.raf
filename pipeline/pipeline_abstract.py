from abc import ABC, abstractmethod


class AbstractPipeline(ABC):
    """
    Generic step in a pipeline
    """

    @abstractmethod
    def run(self, data):
        pass

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def need_input(self):
        pass

    @abstractmethod
    def need_output(self):
        pass