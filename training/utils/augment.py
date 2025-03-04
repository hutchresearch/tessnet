import numpy as np
import random


# Base Class
class Augmentor:
    def __init__(self, proportion, stable=True):
        self.proportion = proportion
        self.augmentations = []
        self.stable = stable

    def __call__(self, data: np.ndarray, index):
        rs = np.random.RandomState(index) if self.stable else np.random
        for augmentation in self.augmentations:
            data = augmentation(data, rs)

        return data

    def __add__(self, other) -> "Augmentor":
        if type(other) == Augmentor:
            self.augmentations += other.augmentations
            return self

        elif isinstance(other, _ConcreteAugmentor):
            self.augmentations.append(other)
            return self

        else:
            raise (TypeError(f"Cannot add {self} to {other}."))

    def __str__(self) -> str:
        return str(self.augmentations)


class _ConcreteAugmentor:
    def __call__(self, data: np.ndarray, *args, **kwargs) -> np.ndarray:
        return self._augment(data, *args, **kwargs)

    def _augment(self, data, rs):
        raise NotImplementedError()



# NoiseAugmentor
class NoiseAugmentor(_ConcreteAugmentor):
    """A data augmentor that adds noise to the data with a specified standard deviation"""

    def __init__(self, settings):
        super().__init__()
        self.p = settings.p
        self.noise_mag_range = settings.mag

    def _augment(self, data, rs):
        if rs.random() < self.p:
            noise_mag = rs.uniform(self.noise_mag_range[0], self.noise_mag_range[1])
            data += rs.normal(scale=noise_mag, size=data.shape)
        return data


# ValueShiftAugmentor
class ValueShiftAugmentor(_ConcreteAugmentor):
    """A data augmentor that shifts all values in the data up or down with the same value drawn from specified standard deviation each time"""
    def __init__(self, settings):
        super().__init__()
        self.p = settings.p
        self.shift_mag_range = settings.mag

    def _augment(self, data, rs):

        if rs.random() < self.p:
            shift_mag = rs.uniform(self.shift_mag_range[0], self.shift_mag_range[1])
            data += shift_mag

        return data


# FlippingAugmentor
class FlippingAugmentor(_ConcreteAugmentor):
    def __init__(self, settings, axis):
        self.axis = axis
        self.p = settings.p
        super().__init__()

    def _augment(self, data, rs):

        if rs.random() < self.p:
            data = np.flip(data, axis=self.axis)

        return data


def get_augmentors(aug_cfg):
    augmentors = []

    for k, settings in aug_cfg.items():
        if k.startswith('augmentor'):
            augmentor = Augmentor(settings.proportion, aug_cfg.stable)
            if settings.get('noise', False):
                augmentor += NoiseAugmentor(settings.noise)
            if settings.get('value_shift', False):
                augmentor += ValueShiftAugmentor(settings.value_shift)
            if settings.get('flip_x', False):
                augmentor += FlippingAugmentor(settings.flip_x, 1)
            if settings.get('flip_y', False):
                augmentor += FlippingAugmentor(settings.flip_y, 2)

            augmentors.append(augmentor)

    return augmentors