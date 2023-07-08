from uninn.activation_functions.act_fun import ActFun


class LinearActFun(ActFun):
    def f(self, value):
        return value

    def d(self, value):
        return 1.0
