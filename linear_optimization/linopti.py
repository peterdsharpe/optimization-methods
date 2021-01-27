class LinOpti:
    def __init__(self):
        self.variables = []
        self.constraints = []
        self.objective = []

    def variable(self):
        var = LinOptiVariable(opti=self, opti_variable_index=len(self.variables))
        self.variables.append(var)
        return var


class LinOptiVariable:
    def __init__(self, opti, opti_variable_index):
        self.opti = opti
        self.opti_variable_index = opti_variable_index

    def __mul__(self, other):
        return f"other = {other}"

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.__mul__(1/other)


lo = LinOpti()
x = lo.variable()
y = lo.variable()
