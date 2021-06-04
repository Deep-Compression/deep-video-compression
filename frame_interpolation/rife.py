import torch
from torch.nn import functional as F

from rife.train_log.RIFE_HDv3 import Model


class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Also, the decorated class cannot be
    inherited from. Other than that, there are no restrictions that apply
    to the decorated class.

    To get the singleton instance, use the `instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.

    """

    def __init__(self, decorated):
        self._decorated = decorated

    def instance(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)


@Singleton
class RifeModel:
    def __init__(self):
        self.model = Model()
        self.model.load_model('rife/train_log', -1)

        self.device = torch.device("cuda")


def rife_interpolation(first_frame, last_frame, depth=1):
    device = RifeModel.instance().device

    first_frame = (torch.tensor(first_frame.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    last_frame = (torch.tensor(last_frame.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

    n, c, h, w = first_frame.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    first_frame = F.pad(first_frame, padding)
    last_frame = F.pad(last_frame, padding)

    img_list = [first_frame, last_frame]

    for i in range(depth):
        tmp = []

        for j in range(len(img_list) - 1):
            mid = RifeModel.instance().model.inference(img_list[j], img_list[j + 1])
            tmp.append(img_list[j])
            tmp.append(mid)

        tmp.append(last_frame)
        img_list = tmp

    images = []

    for image in img_list[1:-1]:
        images.append((image[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])

    return images
