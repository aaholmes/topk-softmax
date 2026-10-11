import math

import torch

from .module import SoftTopKSoftmax

_KINDS = ("exp", "cosine", "linear")
_MEAN_DECAY = 0.9  # of the running mean used for the early stop


class TauScheduler:
    """Anneals tau from tau_start to tau_end and writes it into SoftTopKSoftmax modules.

    Call step() once per training step, next to the learning-rate scheduler. After s calls:

      hold    tau = tau_start while s <= int(hold_frac * total_steps), so the network first
              trains with a near-softmax.
      decay   then tau falls to tau_end at s = total_steps and stays there. kind="exp" is a
              straight line in log tau, "cosine" a half cosine in log tau, "linear" a straight
              line in tau.

    early_stop_eps: once the hold is over, if a running mean of the mass outside the top k falls
    below this value, tau is frozen where it is for the remainder of training. Lowering tau further
    would only shrink the gradients. The mass is the `mass_outside` argument of step(), or by
    default the mean of `stats["mass_outside_mean"]` over the modules that track stats.

    st_frac: for the last int(st_frac * total_steps) steps the modules are switched to mode="st",
    so training sees exact zeros before inference uses mode="hard". With st_frac=0 the scheduler
    never changes a module's mode.

    state_dict() and load_state_dict() let a resumed run continue the schedule.
    """

    def __init__(self, modules, tau_start, tau_end, total_steps, kind="exp",
                 hold_frac=0.05, early_stop_eps=None, st_frac=0.0):
        if isinstance(modules, SoftTopKSoftmax):
            modules = [modules]
        self.modules = list(modules)
        if not tau_end > 0:
            raise ValueError(f"tau_end must be positive, got {tau_end}")
        if not tau_start >= tau_end:
            raise ValueError(f"tau_start must be at least tau_end, got {tau_start} < {tau_end}")
        if not total_steps >= 1:
            raise ValueError(f"total_steps must be at least 1, got {total_steps}")
        if kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
        if not 0 <= hold_frac < 1:
            raise ValueError(f"hold_frac must be in [0, 1), got {hold_frac}")
        if not 0 <= st_frac <= 1:
            raise ValueError(f"st_frac must be in [0, 1], got {st_frac}")
        if early_stop_eps is not None and not early_stop_eps > 0:
            raise ValueError(f"early_stop_eps must be positive, got {early_stop_eps}")
        if st_frac > 0 and any(int(m.k) != m.k for m in self.modules):
            raise ValueError('st_frac > 0 switches modules to mode="st", which needs an integer k')

        self.tau_start, self.tau_end, self.total_steps, self.kind = tau_start, tau_end, total_steps, kind
        self.early_stop_eps = early_stop_eps
        self._hold = int(hold_frac * total_steps)
        self._st_from = total_steps - int(st_frac * total_steps) if st_frac > 0 else None

        self._step = 0
        self._frozen = False
        self._mean = None
        self.tau = float(tau_start)
        self._write()

    def _curve(self, s):
        if s <= self._hold:
            return float(self.tau_start)
        if s >= self.total_steps:
            return float(self.tau_end)
        x = (s - self._hold) / (self.total_steps - self._hold)
        if self.kind == "linear":
            return self.tau_start + (self.tau_end - self.tau_start) * x
        if self.kind == "cosine":
            x = (1 - math.cos(math.pi * x)) / 2
        return self.tau_start * (self.tau_end / self.tau_start) ** x

    def _write(self):
        st = self._st_from is not None and self._step >= self._st_from
        with torch.no_grad():
            for m in self.modules:
                m.tau.fill_(self.tau)  # in place, so compiled modules see the new value
                if st:
                    m.mode = "st"

    def _observed_mass(self):
        values = [float(m.stats["mass_outside_mean"]) for m in self.modules if m.stats]
        return sum(values) / len(values) if values else None

    def step(self, mass_outside=None):
        self._step += 1
        if self.early_stop_eps is not None and not self._frozen:
            if mass_outside is None:
                mass_outside = self._observed_mass()
            if mass_outside is not None:
                mass_outside = float(mass_outside)
                self._mean = mass_outside if self._mean is None else (
                    _MEAN_DECAY * self._mean + (1 - _MEAN_DECAY) * mass_outside)
                self._frozen = self._step > self._hold and self._mean < self.early_stop_eps
        if not self._frozen:
            self.tau = self._curve(self._step)
        self._write()

    def state_dict(self):
        return {"step": self._step, "tau": self.tau, "frozen": self._frozen, "mean": self._mean}

    def load_state_dict(self, state):
        self._step, self.tau = state["step"], state["tau"]
        self._frozen, self._mean = state["frozen"], state["mean"]
        self._write()
