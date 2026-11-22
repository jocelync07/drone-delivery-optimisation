"""Run every experiment and write the tables and charts in results/.

    python experiments.py            # everything, about 2-3 minutes
    python experiments.py --quick    # fewer seeds and smaller searches, for a fast check

1. GA vs ACO on 5 random problems (same budget of 6,000 scored plans each).
2. Adversarial search: how much alpha-beta pruning saves, and how a searching
   fleet does against a greedy one.
3. Game theory: Share vs Compete payoff matrix, equilibrium, repeated game.
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from drones import evaluate, generate, run_aco, run_ga
from drones.aco import ACOConfig
from drones.compete import SearchStats, match, negamax, new_game
from drones.ga import GAConfig
from drones.game import STRATEGIES, analyse, payoff_matrix, tournament
from drones.instance import fmt_time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot
    return pyplot


# ------------------------------------------------------------------ 1. GA vs ACO

def ga_vs_aco(seeds: list[int], budget: int) -> list[str]:
    rows, histories = [], {"GA": [], "ACO": []}
    best_overall = None
    for seed in seeds:
        inst = generate(seed)
        for method, run in (("GA", lambda: run_ga(inst, seed, GAConfig(max_evaluations=budget))),
                            ("ACO", lambda: run_aco(inst, seed, ACOConfig(iterations=budget // 30)))):
            began = time.perf_counter()
            result = run()
            seconds = time.perf_counter() - began
            score = evaluate(inst, result.plan)
            rows.append({"seed": seed, "method": method, "payoff": score.payoff, "served": score.served,
                         "on_time": score.on_time, "distance": score.distance, "late": score.late_minutes,
                         "violations": score.violations, "seconds": seconds})
            histories[method].append(result.history)
            if seed == seeds[0] and method == "ACO":
                best_overall = (inst, result.plan, score)
            print(f"  seed {seed} {method:<3} payoff {score.payoff:7.1f}  served {score.served}/20  ({seconds:.1f}s)")

    lines = ["| Seed | Method | Payoff | Delivered (of 20) | On time | Distance | Late (min) | Limit breaks | Time (s) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['seed']} | {r['method']} | {r['payoff']:.1f} | {r['served']} | {r['on_time']} | "
                     f"{r['distance']:.0f} | {r['late']:.0f} | {r['violations']} | {r['seconds']:.1f} |")
    lines += ["", "| Method | Mean payoff | Std | Best | Mean delivered | Wins |", "|---|---|---|---|---|---|"]
    for method in ("GA", "ACO"):
        mine = [r for r in rows if r["method"] == method]
        other = {r["seed"]: r["payoff"] for r in rows if r["method"] != method}
        pays = [r["payoff"] for r in mine]
        wins = sum(r["payoff"] > other[r["seed"]] for r in mine)
        lines.append(f"| {method} | {statistics.mean(pays):.1f} | {statistics.pstdev(pays):.1f} | {max(pays):.1f} | "
                     f"{statistics.mean(r['served'] for r in mine):.1f} | {wins}/{len(seeds)} |")

    convergence_chart(histories, budget)
    route_map(*best_overall)
    return lines


def convergence_chart(histories: dict, budget: int) -> None:
    pyplot = plt()
    fig, ax = pyplot.subplots(figsize=(8, 4.5))
    checkpoints = list(range(1, budget + 1, max(1, budget // 100)))
    for method, runs in histories.items():
        curves = []
        for history in runs:
            best, i, curve = float("-inf"), 0, []
            for c in checkpoints:
                while i < len(history) and history[i][0] <= c:
                    best = history[i][1]
                    i += 1
                curve.append(best)
            curves.append(curve)
        mean = [statistics.mean(v) for v in zip(*curves)]
        ax.plot(checkpoints, mean, label=f"{method} (mean of {len(runs)} problems)")
    ax.set_xlabel("Plans scored")
    ax.set_ylabel("Best payoff so far")
    ax.set_ylim(bottom=0)
    ax.set_title("Convergence: GA vs ACO on the same budget")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "convergence.png", dpi=150)
    pyplot.close(fig)


def route_map(inst, plan, score) -> None:
    pyplot = plt()
    fig, ax = pyplot.subplots(figsize=(6.5, 6.5))
    served = {rid for route in plan for rid in route}
    for k, route in enumerate(plan):
        if not route:
            continue
        xs = [inst.depot[0]] + [inst.request(r).x for r in route] + [inst.depot[0]]
        ys = [inst.depot[1]] + [inst.request(r).y for r in route] + [inst.depot[1]]
        ax.plot(xs, ys, "-o", label=f"Drone {k + 1}: {len(route)} stops", markersize=5)
    for r in inst.requests:
        ax.annotate(f"{r.id}\n{fmt_time(r.tw_start)}-{fmt_time(r.tw_end)}", (r.x, r.y), fontsize=6,
                    xytext=(3, 3), textcoords="offset points", color="black" if r.id in served else "red")
        if r.id not in served:
            ax.plot(r.x, r.y, "x", color="red", markersize=9)
    ax.plot(*inst.depot, "ks", markersize=10, label="Depot")
    ax.set_title(f"Best ACO plan, problem seed 0: payoff {score.payoff:.0f}, "
                 f"{score.served}/20 delivered (red x = not delivered)", fontsize=9)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "routes.png", dpi=150)
    pyplot.close(fig)


# ------------------------------------------------------------------ 2. adversarial search

def adversarial(depth: int) -> list[str]:
    inst = generate(7, n_requests=10, n_drones=2, battery=180)
    start = new_game(inst)

    lines = [f"Problem: 10 requests, 2 drones per fleet; search depth {depth} plies from the opening position.", "",
             "| Search | Nodes visited | Cut-offs | Value for A | First claim | Time (s) |", "|---|---|---|---|---|---|"]
    for prune in (False, True):
        stats = SearchStats()
        began = time.perf_counter()
        value, claim = negamax(inst, start, depth, prune=prune, stats=stats)
        lines.append(f"| {'Alpha-beta' if prune else 'Plain negamax'} | {stats.nodes:,} | {stats.cutoffs:,} | "
                     f"{value:.2f} | request {claim.request} on drone {claim.drone + 1} | "
                     f"{time.perf_counter() - began:.1f} |")
        print(f"  {'alpha-beta' if prune else 'plain     '} nodes {stats.nodes:,}")

    lines += ["", "Whole games (A moves first):", "",
              "| Fleet A | Fleet B | A payoff | B payoff | A delivered | B delivered |", "|---|---|---|---|---|---|"]
    for a, b in (("greedy", "greedy"), ("negamax", "greedy"), ("greedy", "negamax"), ("negamax", "negamax")):
        m = match(inst, a, b, depth=depth)
        lines.append(f"| {a} | {b} | {m.payoff_a:.1f} | {m.payoff_b:.1f} | {m.served_a} | {m.served_b} |")
        print(f"  {a} vs {b}: {m.payoff_a:.1f} / {m.payoff_b:.1f}")
    return lines


# ------------------------------------------------------------------ 3. game theory

def game_theory(iterations: int) -> list[str]:
    inst = generate(3)
    matrix = payoff_matrix(inst, drones=2, seed=0, config=ACOConfig(ants=20, iterations=iterations))
    result = analyse(matrix)

    lines = ["Payoffs (fleet A, fleet B). A owns the west half of the map, B the east; each has 2 drones.", "",
             "| A \\ B | " + " | ".join(STRATEGIES) + " |", "|---|---|---|"]
    for sa in STRATEGIES:
        lines.append(f"| **{sa}** | " + " | ".join(f"{matrix[(sa, sb)][0]:.1f}, {matrix[(sa, sb)][1]:.1f}"
                                                  for sb in STRATEGIES) + " |")
    lines += ["",
              f"- Dominant strategy for A: **{result.dominant['A'] or 'none'}**; for B: **{result.dominant['B'] or 'none'}**",
              f"- Pure-strategy Nash equilibria: **{', '.join(f'({a}, {b})' for a, b in result.nash) or 'none'}**",
              f"- Highest total payoff: **({result.best_joint[0]}, {result.best_joint[1]})**, "
              f"{sum(matrix[result.best_joint]):.1f}", ""]

    scores = tournament(matrix, rounds=20)
    lines += ["Repeated game, 20 rounds; each strategy plays all four (itself included) as fleet A:", "",
              "| Strategy | Average payoff per round |", "|---|---|"]
    for name, score in sorted(scores.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {name} | {score:.1f} |")
    print("  matrix:", {k: tuple(round(x, 1) for x in v) for k, v in matrix.items()})
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)

    seeds, budget, depth, iterations = ([0, 1], 1500, 2, 20) if args.quick else ([0, 1, 2, 3, 4], 6000, 4, 60)
    print("1. GA vs ACO")
    part1 = ga_vs_aco(seeds, budget)
    print("2. Adversarial search")
    part2 = adversarial(depth)
    print("3. Game theory")
    part3 = game_theory(iterations)

    report = ["# Results", "", "Generated by `python experiments.py`.", "",
              f"## 1. GA vs ACO ({budget:,} scored plans each, {len(seeds)} problems)", "", *part1, "",
              "## 2. Adversarial search", "", *part2, "", "## 3. Share or compete?", "", *part3, ""]
    (OUT / "results.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {OUT / 'results.md'}, convergence.png and routes.png")


if __name__ == "__main__":
    main()
