from abc import abstractmethod
import logging
import math
logger = logging.getLogger(__name__)


class Scheduler(object):
    def __init__(self, epoch, iteration, initial, start=-1, max_value=1):
        self._epoch = epoch
        self._iter = iteration
        self.start = start
        self.value = initial
        self.max_value = max_value

    @abstractmethod
    def step():
        pass


class StepScheduler(Scheduler):
    def __init__(self, epoch, iteration, sc_kwargs):
        initial = sc_kwargs['initial']
        start = sc_kwargs['start']
        max_value = sc_kwargs['max']
        rate = sc_kwargs['rate']
        frequency = sc_kwargs['frequency']
        super(StepScheduler, self).__init__(epoch, iteration,
                                            initial, start, max_value)
        self.rate = rate
        self.frequency = frequency

    def step(self):
        e = self._epoch
        if e >= self.start and self.start >= 0:
            frac = (e - self.start) // self.frequency
            self.value = min(self.rate ** frac, self.max_value)
        self._epoch += 1


class SigmoidScheduler(Scheduler):
    def __init__(self, epoch, iteration, sc_kwargs):
        initial = sc_kwargs['initial']
        start = sc_kwargs['start']
        max_value = sc_kwargs['max']
        speed = sc_kwargs['speed']
        super(StepScheduler, self).__init__(epoch, iteration,
                                            initial, start, max_value)
        self.speed = speed

    def step(self):
        if self._epoch >= self.start and self.start >= 0:
            self.value = 1 - self.speed / (self.speed +
                                           math.exp(self._iter / self.speed))
            self.value = min(self.value, self.max_value)
        self._iter += 1

