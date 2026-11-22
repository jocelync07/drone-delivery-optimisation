"""Delivery problems: a depot, a fleet of drones and requests with time windows.

`generate()` makes a random but repeatable problem from a seed, so every
experiment can be rerun exactly.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Request:
    id: int
    x: float
    y: float
    tw_start: float      # earliest delivery time, minutes after midnight
    tw_end: float        # latest on-time delivery
    service: float = 5.0  # minutes spent at the drop-off


@dataclass(frozen=True)
class Instance:
    requests: tuple[Request, ...]
    depot: tuple[float, float] = (50.0, 50.0)
    n_drones: int = 3
    battery: float = 180.0    # most distance one drone can fly in a day
    speed: float = 1.0        # distance units per minute
    day_start: float = 8 * 60
    day_end: float = 16 * 60
    _dist: dict = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        # Pre-compute every distance once; index 0 is the depot.
        points = {0: self.depot, **{r.id: (r.x, r.y) for r in self.requests}}
        table = {(a, b): math.dist(pa, pb) for a, pa in points.items() for b, pb in points.items()}
        object.__setattr__(self, "_dist", table)

    @property
    def ids(self) -> list[int]:
        return [r.id for r in self.requests]

    def request(self, rid: int) -> Request:
        return self.requests[rid - 1]

    def dist(self, a: int, b: int) -> float:
        """Distance between two stops; 0 means the depot."""
        return self._dist[(a, b)]


def generate(seed: int = 0, n_requests: int = 20, n_drones: int = 3, battery: float = 180.0) -> Instance:
    """Random requests on a 100 x 100 map, each with a 1-3 hour delivery window."""
    rng = random.Random(seed)
    requests = []
    for i in range(1, n_requests + 1):
        start = rng.randrange(8 * 60, 14 * 60, 15)
        width = rng.choice([60, 90, 120, 180])
        requests.append(Request(i, round(rng.uniform(0, 100), 1), round(rng.uniform(0, 100), 1),
                                start, min(start + width, 16 * 60)))
    return Instance(tuple(requests), n_drones=n_drones, battery=battery)


def fmt_time(minutes: float) -> str:
    m = int(round(minutes))
    return f"{m // 60:02d}:{m % 60:02d}"
