"""One scoring function used by every method: GA, ACO, the adversarial search and the game.

A plan is a list of routes, one per drone; a route is the request ids in visit
order. Each drone leaves the depot at the start of the day, waits if it arrives
before a window opens, and is late if it arrives after the window closes.

Payoff = revenue for each delivery - energy - lateness penalty
         - penalty for each request left unserved - penalty for each broken limit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from .instance import Instance

REVENUE = 30.0            # per delivered request
ENERGY_PER_UNIT = 0.5     # per distance unit flown
LATE_PER_MINUTE = 1.0
UNSERVED_PENALTY = 20.0   # per request nobody delivered
VIOLATION_PENALTY = 1000.0  # per route over the battery or past the end of the day

Route = Sequence[int]


@dataclass
class RouteStats:
    distance: float = 0.0
    finish: float = 0.0
    late_minutes: float = 0.0
    arrivals: dict = field(default_factory=dict)  # request id -> arrival time
    battery_ok: bool = True
    horizon_ok: bool = True


@dataclass
class Score:
    payoff: float
    distance: float
    served: int
    on_time: int
    late_minutes: float
    unserved: int
    violations: int
    makespan: float
    arrivals: dict


def simulate(inst: Instance, route: Route) -> RouteStats:
    stats = RouteStats(finish=inst.day_start)
    t, here = inst.day_start, 0
    for rid in route:
        r = inst.request(rid)
        leg = inst.dist(here, rid)
        stats.distance += leg
        t += leg / inst.speed
        t = max(t, r.tw_start)          # wait for the window to open
        stats.arrivals[rid] = t
        stats.late_minutes += max(0.0, t - r.tw_end)
        t += r.service
        here = rid
    if route:
        leg = inst.dist(here, 0)
        stats.distance += leg
        t += leg / inst.speed
    stats.finish = t
    stats.battery_ok = stats.distance <= inst.battery + 1e-9
    stats.horizon_ok = t <= inst.day_end + 1e-9
    return stats


def evaluate(inst: Instance, plan: Iterable[Route], credited: Optional[set] = None,
             universe: Optional[Iterable[int]] = None) -> Score:
    """Score a plan.

    credited:  if given, only these requests earn revenue (used when two fleets
               visit the same request and only the first to arrive is paid).
    universe:  the requests this plan is responsible for; any not credited count
               as unserved. Defaults to all requests.
    """
    plan = [list(r) for r in plan]
    visited = [rid for route in plan for rid in route]
    if len(visited) != len(set(visited)):
        raise ValueError("A request appears more than once in the plan")

    routes = [simulate(inst, r) for r in plan if r]
    arrivals = {k: v for s in routes for k, v in s.arrivals.items()}
    paid = set(visited) if credited is None else set(visited) & credited
    universe = set(inst.ids if universe is None else universe)

    late = sum(max(0.0, arrivals[rid] - inst.request(rid).tw_end) for rid in paid)
    on_time = sum(arrivals[rid] <= inst.request(rid).tw_end + 1e-9 for rid in paid)
    distance = sum(s.distance for s in routes)
    violations = sum((not s.battery_ok) + (not s.horizon_ok) for s in routes)
    unserved = len(universe - paid)

    payoff = (REVENUE * len(paid) - ENERGY_PER_UNIT * distance - LATE_PER_MINUTE * late
              - UNSERVED_PENALTY * unserved - VIOLATION_PENALTY * violations)
    return Score(round(payoff, 4), distance, len(paid), on_time, late, unserved, violations,
                 max((s.finish for s in routes), default=inst.day_start), arrivals)


def can_append(inst: Instance, route: Route, rid: int) -> bool:
    """Would the drone still get home within its battery and before the day ends?"""
    stats = simulate(inst, list(route) + [rid])
    return stats.battery_ok and stats.horizon_ok


def split_into_routes(inst: Instance, order: Sequence[int], n_drones: Optional[int] = None) -> list[list[int]]:
    """Turn a visiting order into feasible routes, first-fit.

    Each request in turn goes to the first drone that can still fit it (battery
    and end of day); one that no drone can fit is skipped and stays unserved.
    This is the decoder the genetic algorithm uses.
    """
    n_drones = inst.n_drones if n_drones is None else n_drones
    routes: list[list[int]] = [[] for _ in range(n_drones)]
    for rid in order:
        for route in routes:
            if can_append(inst, route, rid):
                route.append(rid)
                break
    return routes
