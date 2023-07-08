import numpy as np
from uninn.activation_functions.act_fun import ActFun


class SigmoidActFun(ActFun):
    def f(self, value):
        return 1.0 / (1.0 + np.exp(-value))

    def d(self, value):
        return value * (1.0 - value)
