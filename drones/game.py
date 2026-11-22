"""Game theory: should two rival fleets keep to their own zones or compete everywhere?

Each fleet is responsible for one half of the map (A: west, B: east) and is
penalised for any request in its zone that it does not deliver. It chooses:

- Share:   plan routes (with ACO) only inside its own zone.
- Compete: plan routes over the whole map, poaching the rival's customers.

Both plans run at the same time. When both fleets visit a request, the one that
arrives first is paid; the other still pays for the trip. This gives a 2x2
payoff matrix, which is checked for dominant strategies, Nash equilibria and the
best joint outcome. A repeated version then tests strategies that react to the
rival's last move.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from .aco import ACOConfig, run_aco
from .evaluate import evaluate
from .instance import Instance

STRATEGIES = ("Share", "Compete")


def zones(inst: Instance) -> dict[str, set]:
    west = {r.id for r in inst.requests if r.x < inst.depot[0]}
    return {"A": west, "B": set(inst.ids) - west}


def plan_for(inst: Instance, fleet: str, strategy: str, drones: int, seed: int, config: ACOConfig) -> list:
    allowed = zones(inst)[fleet] if strategy == "Share" else set(inst.ids)
    return run_aco(inst, seed=seed, config=config, allowed=allowed, n_drones=drones).plan


def resolve(inst: Instance, plan_a: list, plan_b: list) -> tuple[float, float]:
    """Payoffs when both plans run together; first to arrive is paid."""
    zone = zones(inst)
    arrive_a = evaluate(inst, plan_a, universe=()).arrivals
    arrive_b = evaluate(inst, plan_b, universe=()).arrivals
    credited_a, credited_b = set(arrive_a), set(arrive_b)
    for rid in credited_a & credited_b:
        a_first = arrive_a[rid] < arrive_b[rid] or (arrive_a[rid] == arrive_b[rid] and rid % 2 == 1)
        (credited_b if a_first else credited_a).discard(rid)
    pay_a = evaluate(inst, plan_a, credited=credited_a, universe=zone["A"]).payoff
    pay_b = evaluate(inst, plan_b, credited=credited_b, universe=zone["B"]).payoff
    return pay_a, pay_b


@dataclass
class GameAnalysis:
    matrix: dict           # (strategy A, strategy B) -> (payoff A, payoff B)
    dominant: dict         # fleet -> its dominant strategy, or None
    nash: list             # pure-strategy Nash equilibria
    best_joint: tuple      # profile with the highest total payoff


def payoff_matrix(inst: Instance, drones: int = 2, seed: int = 0,
                  config: ACOConfig = ACOConfig(ants=20, iterations=60)) -> dict:
    plans = {(fleet, s): plan_for(inst, fleet, s, drones, seed, config)
             for fleet in "AB" for s in STRATEGIES}
    return {(sa, sb): resolve(inst, plans[("A", sa)], plans[("B", sb)])
            for sa, sb in product(STRATEGIES, STRATEGIES)}


def analyse(matrix: dict) -> GameAnalysis:
    def best_responses(fleet: str, other: str) -> set:
        idx = 0 if fleet == "A" else 1
        profile = (lambda mine: (mine, other)) if fleet == "A" else (lambda mine: (other, mine))
        top = max(matrix[profile(m)][idx] for m in STRATEGIES)
        return {m for m in STRATEGIES if matrix[profile(m)][idx] == top}

    dominant = {}
    for fleet in "AB":
        always_best = set(STRATEGIES)
        for other in STRATEGIES:
            always_best &= best_responses(fleet, other)
        dominant[fleet] = next(iter(always_best)) if len(always_best) == 1 else None

    nash = [(sa, sb) for sa, sb in product(STRATEGIES, STRATEGIES)
            if sa in best_responses("A", sb) and sb in best_responses("B", sa)]
    best_joint = max(matrix, key=lambda p: sum(matrix[p]))
    return GameAnalysis(matrix, dominant, nash, best_joint)


# ------------------------------------------------------------------ repeated game

def always_share(mine, theirs):
    return "Share"


def always_compete(mine, theirs):
    return "Compete"


def tit_for_tat(mine, theirs):
    """Share first, then copy what the rival did last round."""
    return theirs[-1] if theirs else "Share"


def grim(mine, theirs):
    """Share until the rival competes once, then compete forever."""
    return "Compete" if "Compete" in theirs else "Share"


REPEATED_STRATEGIES = {
    "Always share": always_share,
    "Always compete": always_compete,
    "Tit-for-tat": tit_for_tat,
    "Grim trigger": grim,
}


def repeated(matrix: dict, a, b, rounds: int = 20) -> tuple[float, float]:
    hist_a, hist_b, total_a, total_b = [], [], 0.0, 0.0
    for _ in range(rounds):
        sa, sb = a(hist_a, hist_b), b(hist_b, hist_a)
        pa, pb = matrix[(sa, sb)]
        total_a, total_b = total_a + pa, total_b + pb
        hist_a.append(sa)
        hist_b.append(sb)
    return total_a, total_b


def tournament(matrix: dict, rounds: int = 20) -> dict:
    """Every strategy plays every strategy (including itself) as fleet A; average score per round."""
    scores = {}
    for name_a, a in REPEATED_STRATEGIES.items():
        results = [repeated(matrix, a, b, rounds)[0] / rounds for b in REPEATED_STRATEGIES.values()]
        scores[name_a] = sum(results) / len(results)
    return scores
