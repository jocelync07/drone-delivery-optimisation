"""Genetic algorithm.

Each individual is an order of all requests. The decoder in evaluate.py fills
drones in that order while battery and time allow, so every individual decodes
to a plan that respects the limits; the GA only has to learn a good order.

Operators: tournament selection, order crossover (OX1), swap or reversal
mutation, and elitism (the best individuals survive unchanged).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .evaluate import evaluate, split_into_routes
from .instance import Instance


@dataclass
class GAConfig:
    population: int = 60
    max_evaluations: int = 6000   # distinct plans scored; the same budget ACO gets
    crossover_rate: float = 0.9
    mutation_rate: float = 0.3
    tournament: int = 3
    elites: int = 2


@dataclass
class RunResult:
    method: str
    plan: list
    payoff: float
    history: list = field(default_factory=list)  # (evaluations used, best payoff so far)
    evaluations: int = 0


def order_crossover(a: list, b: list, rng: random.Random) -> list:
    """Copy a slice from parent a, then fill the rest in parent b's order."""
    i, j = sorted(rng.sample(range(len(a)), 2))
    child = [None] * len(a)
    child[i:j + 1] = a[i:j + 1]
    taken = set(a[i:j + 1])
    fill = (g for g in b if g not in taken)
    return [g if g is not None else next(fill) for g in child]


def mutate(order: list, rng: random.Random) -> list:
    order = order[:]
    i, j = sorted(rng.sample(range(len(order)), 2))
    if rng.random() < 0.5:
        order[i], order[j] = order[j], order[i]
    else:
        order[i:j + 1] = reversed(order[i:j + 1])
    return order


def run_ga(inst: Instance, seed: int = 0, config: GAConfig = GAConfig()) -> RunResult:
    rng = random.Random(seed)
    result = RunResult("GA", [], float("-inf"))
    cache: dict[tuple, float] = {}

    def fitness(order: list) -> float:
        key = tuple(order)
        if key not in cache:
            plan = split_into_routes(inst, order)
            cache[key] = evaluate(inst, plan).payoff
            result.evaluations += 1
            if cache[key] > result.payoff:
                result.payoff, result.plan = cache[key], plan
            result.history.append((result.evaluations, result.payoff))
        return cache[key]

    def tournament(pop: list) -> list:
        return max(rng.sample(pop, config.tournament), key=fitness)

    population = [rng.sample(inst.ids, len(inst.ids)) for _ in range(config.population)]
    stale = 0
    while result.evaluations < config.max_evaluations and stale < 200:
        before = result.evaluations
        population.sort(key=fitness, reverse=True)
        children = population[:config.elites]
        while len(children) < config.population:
            a, b = tournament(population), tournament(population)
            child = order_crossover(a, b, rng) if rng.random() < config.crossover_rate else a[:]
            if rng.random() < config.mutation_rate:
                child = mutate(child, rng)
            children.append(child)
        population = children
        stale = stale + 1 if result.evaluations == before else 0  # stop if nothing new is being tried
    return result
