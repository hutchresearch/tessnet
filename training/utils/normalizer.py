import numpy as np

class LinearNormalizer:
    def __init__(self, data_min, data_max, norm_max, norm_min):
        self.data_min = data_min
        self.data_max = data_max

    def normalize(self, data):
        return data * (self.data_max - self.data_min) + self.data_min

    def denormalize(self, data):
        return (data - self.data_min) / (self.data_max - self.data_min)


class ScaleNormalizer:
    def __init__(self, one_value):
        self.scale = one_value

    def normalize(self, data):
        return data / self.scale

    def denormalize(self, data):
        return data * self.scale


class LogNormalizer:
    def __init__(self, one_value):
        # one value is the value that equals one after normalization
        self.base = np.sqrt(one_value + 1)

    def normalize(self, data):
        return np.emath.logn(self.base, data + 1.0) - 1.0

    def denormalize(self, data):
        return self.base ** (data + 1) - 1
