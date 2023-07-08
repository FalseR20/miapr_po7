import abc


class ActFun(abc.ABC):
    @abc.abstractmethod
    def f(self, value):
        pass

    @staticmethod
    @abc.abstractmethod
    def d(self, value):
        pass
