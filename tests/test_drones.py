import importlib
import random

import pytest

from drones import ACOConfig, GAConfig, Instance, Request, evaluate, generate, run_aco, run_ga, split_into_routes
from drones.compete import SearchStats, match, moves, negamax, new_game, play
from drones.ga import mutate, order_crossover
from drones.game import always_compete, analyse, grim, repeated, tit_for_tat

ev = importlib.import_module("drones.evaluate")  # the module, not the evaluate() function


@pytest.fixture
def line():
    """Three requests on a line east of the depot, easy to check by hand."""
    reqs = (Request(1, 60, 50, 8 * 60, 9 * 60),    # 10 from the depot
            Request(2, 70, 50, 8 * 60, 8 * 60 + 5),  # closes almost at once
            Request(3, 90, 50, 12 * 60, 13 * 60))  # opens at noon
    return Instance(reqs, n_drones=2, battery=100.0)


# ------------------------------------------------------------------ scoring

def test_simulate_times_and_distance(line):
    stats = ev.simulate(line, [1, 2])
    assert stats.distance == pytest.approx(10 + 10 + 20)
    assert stats.arrivals[1] == pytest.approx(8 * 60 + 10)
    assert stats.arrivals[2] == pytest.approx(8 * 60 + 10 + 5 + 10)  # service 5 min, then 10 units
    assert stats.battery_ok and stats.horizon_ok


def test_waiting_for_a_window(line):
    stats = ev.simulate(line, [3])
    assert stats.arrivals[3] == 12 * 60  # arrives 08:40, waits until noon


def test_payoff_formula(line):
    score = evaluate(line, [[1, 2]])
    late = (8 * 60 + 25) - (8 * 60 + 5)
    expected = 2 * ev.REVENUE - ev.ENERGY_PER_UNIT * 40 - ev.LATE_PER_MINUTE * late - ev.UNSERVED_PENALTY * 1
    assert score.payoff == pytest.approx(expected)
    assert (score.served, score.on_time, score.unserved) == (2, 1, 1)


def test_battery_violation_is_penalised(line):
    tight = Instance(line.requests, n_drones=1, battery=50.0)
    score = evaluate(tight, [[1, 3]])  # 10 + 30 + 40 = 80 > 50
    assert score.violations == 1
    assert score.payoff < -500


def test_duplicate_request_rejected(line):
    with pytest.raises(ValueError):
        evaluate(line, [[1], [1]])


def test_first_fit_decoder_respects_limits():
    inst = generate(0)
    routes = split_into_routes(inst, inst.ids)
    assert len(routes) == inst.n_drones
    for route in routes:
        stats = ev.simulate(inst, route)
        assert stats.battery_ok and stats.horizon_ok


# ------------------------------------------------------------------ GA and ACO

def test_order_crossover_and_mutation_keep_a_permutation():
    rng = random.Random(0)
    a, b = list(range(1, 11)), list(range(10, 0, -1))
    for _ in range(50):
        child = order_crossover(a, b, rng)
        assert sorted(child) == a
        assert sorted(mutate(child, rng)) == a


@pytest.mark.parametrize("runner", ["ga", "aco"])
def test_methods_return_valid_plans(runner):
    inst = generate(2)
    result = (run_ga(inst, 0, GAConfig(population=20, max_evaluations=300)) if runner == "ga"
              else run_aco(inst, 0, ACOConfig(ants=10, iterations=10)))
    score = evaluate(inst, result.plan)
    assert score.violations == 0
    assert score.payoff == pytest.approx(result.payoff)
    assert len(result.plan) == inst.n_drones
    # history is the best-so-far curve: it never goes down
    pays = [p for _, p in result.history]
    assert all(x <= y for x, y in zip(pays, pays[1:]))


def test_same_seed_same_result():
    inst = generate(4)
    assert run_aco(inst, 3, ACOConfig(ants=5, iterations=5)).plan == run_aco(inst, 3, ACOConfig(ants=5, iterations=5)).plan
    assert run_ga(inst, 3, GAConfig(population=10, max_evaluations=100)).plan == \
        run_ga(inst, 3, GAConfig(population=10, max_evaluations=100)).plan


def test_aco_respects_allowed_zone():
    inst = generate(1)
    zone = set(inst.ids[:8])
    plan = run_aco(inst, 0, ACOConfig(ants=5, iterations=5), allowed=zone).plan
    assert {rid for route in plan for rid in route} <= zone


# ------------------------------------------------------------------ adversarial search

@pytest.fixture
def small():
    return generate(7, n_requests=6, n_drones=2, battery=180)


def test_alpha_beta_gives_the_same_answer_with_fewer_nodes(small):
    start = new_game(small)
    for depth in (1, 2, 3):
        plain, pruned = SearchStats(), SearchStats()
        v1, _ = negamax(small, start, depth, prune=False, stats=plain)
        v2, _ = negamax(small, start, depth, prune=True, stats=pruned)
        assert v1 == pytest.approx(v2)
        assert pruned.nodes <= plain.nodes


def test_game_ends_after_two_passes(small):
    s = play(play(new_game(small), None), None)
    assert moves(small, s) == []


def test_match_claims_each_request_at_most_once(small):
    m = match(small, "negamax", "greedy", depth=2)
    a = [r for route in m.final.a for r in route]
    b = [r for route in m.final.b for r in route]
    assert len(a + b) == len(set(a + b))


# ------------------------------------------------------------------ game theory

PRISONERS = {("Share", "Share"): (3, 3), ("Share", "Compete"): (0, 5),
             ("Compete", "Share"): (5, 0), ("Compete", "Compete"): (1, 1)}


def test_prisoners_dilemma_analysis():
    result = analyse(PRISONERS)
    assert result.dominant == {"A": "Compete", "B": "Compete"}
    assert result.nash == [("Compete", "Compete")]
    assert result.best_joint == ("Share", "Share")


def test_no_pure_equilibrium_detected_for_matching_pennies():
    pennies = {("Share", "Share"): (1, -1), ("Share", "Compete"): (-1, 1),
               ("Compete", "Share"): (-1, 1), ("Compete", "Compete"): (1, -1)}
    assert analyse(pennies).nash == []


def test_repeated_strategies():
    # Tit-for-tat loses only the first round to an always-compete rival.
    a, b = repeated(PRISONERS, tit_for_tat, always_compete, rounds=10)
    assert (a, b) == (0 + 9 * 1, 5 + 9 * 1)
    # Two cooperative strategies share every round.
    assert repeated(PRISONERS, tit_for_tat, grim, rounds=10) == (30, 30)
