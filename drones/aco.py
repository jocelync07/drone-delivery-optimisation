"""Ant colony optimisation.

Each ant builds a whole plan drone by drone. From its current stop it picks the
next request with probability proportional to pheromone^alpha x desirability^beta,
choosing only among requests it can still fit (battery and end of day).
Desirability favours requests that are close and need little waiting or
lateness. After each iteration pheromone evaporates, and the best ant of the
iteration and the best plan so far add pheromone along their routes.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from .evaluate import evaluate
from .ga import RunResult
from .instance import Instance


@dataclass
class ACOConfig:
    ants: int = 30
    iterations: int = 200
    alpha: float = 1.0        # weight of pheromone
    beta: float = 3.0         # weight of desirability
    evaporation: float = 0.1
    deposit: float = 1.0


def build_plan(inst: Instance, pheromone: dict, rng: random.Random, config: ACOConfig,
               allowed: Optional[set] = None, n_drones: Optional[int] = None) -> list[list[int]]:
    todo = set(inst.ids if allowed is None else allowed)
    plan = []
    for _ in range(inst.n_drones if n_drones is None else n_drones):
        route, here = [], 0
        t, flown = inst.day_start, 0.0  # time ready to leave `here`, distance flown so far
        while True:
            options, weights = [], []
            for rid in sorted(todo):  # sorted: same choices for the same seed
                r = inst.request(rid)
                leg, home = inst.dist(here, rid), inst.dist(rid, 0)
                arrive = t + leg / inst.speed
                done = max(arrive, r.tw_start) + r.service
                # Only requests the drone can still reach and get home from in time.
                if flown + leg + home > inst.battery or done + home / inst.speed > inst.day_end:
                    continue
                wait, late = max(0.0, r.tw_start - arrive), max(0.0, arrive - r.tw_end)
                desirability = 1.0 / (1.0 + leg + 0.5 * wait + 2.0 * late)
                options.append(rid)
                weights.append(pheromone[(here, rid)] ** config.alpha * desirability ** config.beta)
            if not options:
                break
            rid = rng.choices(options, weights)[0]
            r = inst.request(rid)
            leg = inst.dist(here, rid)
            t = max(t + leg / inst.speed, r.tw_start) + r.service
            flown += leg
            route.append(rid)
            todo.discard(rid)
            here = rid
        plan.append(route)
    return plan


def edges(plan: list[list[int]]):
    for route in plan:
        if not route:
            continue  # an unused drone lays no pheromone
        stops = [0] + route + [0]
        yield from zip(stops, stops[1:])


def run_aco(inst: Instance, seed: int = 0, config: ACOConfig = ACOConfig(),
            allowed: Optional[set] = None, n_drones: Optional[int] = None) -> RunResult:
    """allowed limits the ants to some requests (used for a fleet that keeps to its own zone)."""
    rng = random.Random(seed)
    nodes = [0] + inst.ids
    pheromone = {(a, b): 1.0 for a in nodes for b in nodes if a != b}
    result = RunResult("ACO", [], float("-inf"))
    universe = None if allowed is None else allowed

    for _ in range(config.iterations):
        iteration_best, iteration_payoff = None, float("-inf")
        for _ in range(config.ants):
            plan = build_plan(inst, pheromone, rng, config, allowed, n_drones)
            payoff = evaluate(inst, plan, universe=universe).payoff
            result.evaluations += 1
            if payoff > iteration_payoff:
                iteration_best, iteration_payoff = plan, payoff
            if payoff > result.payoff:
                result.payoff, result.plan = payoff, plan
            result.history.append((result.evaluations, result.payoff))

        for key in pheromone:
            pheromone[key] = max(1e-3, pheromone[key] * (1 - config.evaporation))
        for plan, amount in ((iteration_best, config.deposit), (result.plan, config.deposit / 2)):
            for edge in edges(plan):
                pheromone[edge] += amount
    return result
