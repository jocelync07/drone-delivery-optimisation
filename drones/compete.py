"""Two rival fleets claiming deliveries one at a time: adversarial search.

Fleets A and B take turns. A turn either claims one unclaimed request, adding
it to the end of one of that fleet's drone routes (if the drone can still fit
it), or passes, because a far-away or late request can cost more than it pays.
The game ends after two passes in a row.

Each fleet's payoff is its revenue minus its flying and lateness costs (from
evaluate.py). The search value is payoff(A) - payoff(B), a zero-sum view in
which A maximises and B minimises. Negamax explores it with alpha-beta pruning;
plain negamax (no pruning) is kept to check the pruning never changes the answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .evaluate import can_append, evaluate
from .instance import Instance

Plan = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class State:
    a: Plan
    b: Plan
    unclaimed: frozenset
    to_move: str = "A"
    passes: int = 0  # passes in a row; two ends the game

    def plan(self, fleet: str) -> Plan:
        return self.a if fleet == "A" else self.b


@dataclass(frozen=True)
class Claim:
    drone: int
    request: int


def new_game(inst: Instance, drones_per_fleet: int = 2) -> State:
    empty = tuple(() for _ in range(drones_per_fleet))
    return State(empty, empty, frozenset(inst.ids), "A")


def fleet_payoff(inst: Instance, plan: Plan) -> float:
    # universe=() : a fleet is not penalised for requests the rival delivered.
    return evaluate(inst, plan, universe=()).payoff


def value_for_a(inst: Instance, s: State) -> float:
    return fleet_payoff(inst, s.a) - fleet_payoff(inst, s.b)


def legal_claims(inst: Instance, s: State, fleet: Optional[str] = None) -> list[Claim]:
    """Claims ordered nearest-first, which helps alpha-beta prune early.

    Empty drones are interchangeable, so only the first empty one is offered.
    """
    fleet = fleet or s.to_move
    plan = s.plan(fleet)
    claims = []
    seen_empty = False
    for d, route in enumerate(plan):
        if not route:
            if seen_empty:
                continue
            seen_empty = True
        here = route[-1] if route else 0
        for rid in s.unclaimed:
            if can_append(inst, route, rid):
                claims.append((inst.dist(here, rid), rid, d))
    return [Claim(d, rid) for _, rid, d in sorted(claims)]


def play(s: State, claim: Optional[Claim]) -> State:
    """Apply a claim (or a pass when claim is None) and hand the turn over."""
    nxt = "B" if s.to_move == "A" else "A"
    if claim is None:
        return State(s.a, s.b, s.unclaimed, nxt, s.passes + 1)
    plan = list(s.plan(s.to_move))
    plan[claim.drone] = plan[claim.drone] + (claim.request,)
    plan = tuple(plan)
    unclaimed = s.unclaimed - {claim.request}
    return State(plan, s.b, unclaimed, nxt) if s.to_move == "A" else State(s.a, plan, unclaimed, nxt)


def moves(inst: Instance, s: State) -> list[Optional[Claim]]:
    """Legal claims plus a pass (None); nothing once the game is over."""
    if s.passes >= 2:
        return []
    return legal_claims(inst, s) + [None]


@dataclass
class SearchStats:
    nodes: int = 0
    cutoffs: int = 0


def negamax(inst: Instance, s: State, depth: int, alpha: float = float("-inf"),
            beta: float = float("inf"), prune: bool = True,
            stats: Optional[SearchStats] = None) -> tuple[float, Optional[Claim]]:
    """Best value for the player to move (from their side), and the claim that gets it."""
    stats = stats if stats is not None else SearchStats()
    stats.nodes += 1
    sign = 1 if s.to_move == "A" else -1
    options = moves(inst, s)
    if depth == 0 or not options:
        return sign * value_for_a(inst, s), None

    best_value, best_move = float("-inf"), options[0]
    for move in options:
        value = -negamax(inst, play(s, move), depth - 1, -beta, -alpha, prune, stats)[0]
        if value > best_value:
            best_value, best_move = value, move
        alpha = max(alpha, value)
        if prune and alpha >= beta:
            stats.cutoffs += 1
            break
    return best_value, best_move


def greedy_move(inst: Instance, s: State) -> Optional[Claim]:
    """One-step lookahead: the claim that most improves this fleet's own payoff,
    or a pass if every claim would lose money."""
    claims = [c for c in moves(inst, s) if c is not None]
    if not claims:
        return None
    before = fleet_payoff(inst, s.plan(s.to_move))
    gain = {c: fleet_payoff(inst, play(s, c).plan(s.to_move)) - before for c in claims}
    best = max(claims, key=gain.get)
    return best if gain[best] > 0 else None


@dataclass
class MatchResult:
    a_player: str
    b_player: str
    payoff_a: float
    payoff_b: float
    served_a: int
    served_b: int
    final: State = field(repr=False, default=None)


def match(inst: Instance, a_player: str, b_player: str, depth: int = 3, drones_per_fleet: int = 2) -> MatchResult:
    """Play a whole game. Players are 'greedy' or 'negamax'."""
    s = new_game(inst, drones_per_fleet)
    while moves(inst, s):
        player = a_player if s.to_move == "A" else b_player
        claim = negamax(inst, s, depth)[1] if player == "negamax" else greedy_move(inst, s)
        s = play(s, claim)
    a, b = evaluate(inst, s.a, universe=()), evaluate(inst, s.b, universe=())
    return MatchResult(a_player, b_player, a.payoff, b.payoff, a.served, b.served, s)
