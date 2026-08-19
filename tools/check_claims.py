#!/usr/bin/env python3
"""Guard against prose drifting away from results.

Numbers in this repo live in three places:

  1. inside `assert`s        -- self-enforcing, the notebook fails if they drift
  2. inside printed output   -- regenerated on every run, so never stale
  3. inside markdown prose   -- CANNOT self-update, and this is where rot starts

This script covers (3). Every load-bearing number written in the README or in a
notebook's narrative is pinned here next to the notebook output it came from. If
someone retunes a run and the printed number moves, the prose that quotes it now
fails CI instead of quietly becoming a lie.

It reads committed outputs only -- it does not execute anything, so it runs in
about a second.

Adding a claim is deliberately cheap: one Claim(...) entry. If you change a
number in the README or in notebook prose, change it here too. That coupling is
the entire point.

Usage:  python tools/check_claims.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {'.git', '__pycache__', '.ipynb_checkpoints', 'data', '.github', 'tools'}


@dataclass
class Claim:
    """A statement made in prose, and the run output that has to back it up."""
    what: str
    output: list[str] = field(default_factory=list)   # must appear in some notebook's outputs
    readme: list[str] = field(default_factory=list)   # must appear in README.md
    prose: list[tuple[str, str]] = field(default_factory=list)  # (notebook substring, markdown substring)


CLAIMS = [
    Claim("from-scratch GPT reaches 1.563 nats/char vs a 4.174 random baseline",
          output=["we reached 1.563", "ln(vocab) = 4.174"],
          readme=["**1.56 nats/char**", "random baseline: 4.17"]),

    Claim("hand-derived layer backward passes match PyTorch to ~1e-14",
          output=["2.13e-14"],
          readme=["match to ~1e-14"]),

    Claim("scalar autograd matches PyTorch to ~1e-9",
          output=["matches PyTorch to ~1e-9"],
          readme=["~1e-9"]),

    Claim("all six optimizers match torch.optim to < 1e-6",
          output=["match torch.optim to < 1e-6"],
          readme=["< 1e-6 across their full 40-step trajectories"]),

    Claim("FlashAttention is numerically identical to standard attention",
          output=["4.440892098500626e-16"],
          readme=["4e-16"]),

    Claim("grokking: validation generalizes 29x after memorization, and weight decay is the cause",
          output=["29x delay", "1.000  |  without: 0.001"],
          readme=["~29× after memorization", "1.000 vs 0.001"]),

    Claim("double descent peaks at the interpolation threshold",
          output=["test error peaks at h=500 (interpolation threshold N=500)"],
          readme=["peak exactly at the interpolation threshold"]),

    Claim("distillation lifts the small student from 0.882 to 0.972, seed-averaged",
          output=["0.9719", "0.8817"],
          readme=["0.972 vs 0.882"]),

    Claim("lottery ticket only partially reproduces, and the repo says so",
          output=["0.895 +/- 0.043", "+/-0.076", "+0.053"],
          readme=["0.895 ± 0.043", "**+0.053**", "±0.077"],
          prose=[("29_reproduce_lottery_ticket", "0.895 ± 0.043"),
                 ("29_reproduce_lottery_ticket", "+0.053")]),

    Claim("DQN learns a real policy but does not solve CartPole",
          output=["best episode: 407", "peak 50-episode average: 168", "survives ~25.4 steps"],
          readme=["best episode 407 of 500 steps", "peak 50-episode average 168"]),

    Claim("PPO's clipped objective peaks at 408",
          output=["peak-20 408"],
          readme=["408 peak"]),

    Claim("the scaling-law fit has a real irreducible-loss floor",
          output=["L∞=1.476", "N^(-0.278)"],
          readme=["L∞ ≈ 1.47", "α ≈ 0.27"]),

    Claim("the capstone goes from 0.000 instruction-following to 0.997 after SFT",
          output=["PRETRAIN only: 0.000", "after SFT: 0.997"],
          readme=["0% instruction-following", "~100% after SFT"]),

    Claim("ViT reaches ~96% on MNIST -- prose must not round it up to 98%",
          output=["epoch 7  test acc 0.9625"],
          prose=[("31_reproduce_vision_transformer", "~96% on MNIST after 8 epochs")]),

    Claim("NB36 prints the REINFORCE/actor-critic comparison instead of asserting an ordering in prose",
          # Both figures move whenever anything upstream touches the RNG stream -- raising the
          # variance-estimate sample count once flipped actor-critic from 83 to 164. The fix was to
          # take the numbers out of the narrative entirely, so this claim guards that they stay out.
          output=["REINFORCE: final-100 avg length", "(REINFORCE reached"],
          prose=[("36_policy_gradients_actor_critic", "Compare the two printed final-100 averages")]),

    Claim("NB36 claims no ordering for the baseline step, because the data does not support one",
          output=["that step is large and holds on every run", "claims no ordering for that step"]),

    Claim("NB07's reference to NB06's MLP matches what NB06 actually scores",
          output=["final: train acc 1.000, val acc 0.925"],
          prose=[("07_cnn_from_scratch", "hit ~93% on a 2k-example subset")]),

    Claim("LeNet beats the MLP with fewer parameters",
          output=["MLP: 0.9716 acc, 235,146 params  |  CNN: 0.9882 acc, 206,922 params"]),
]


def load():
    outputs, markdown, n_assert, problems = [], {}, 0, []
    for dp, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in sorted(files):
            if not f.endswith('.ipynb'):
                continue
            path = os.path.join(dp, f)
            rel = os.path.relpath(path, ROOT).replace('\\', '/')
            nb = json.load(open(path, encoding='utf-8'))
            code = [c for c in nb['cells'] if c['cell_type'] == 'code']
            src = '\n'.join(''.join(c.get('source', [])) for c in code)
            n = len(re.findall(r'^\s*assert\b', src, re.M))
            n_assert += n
            if n == 0:
                problems.append(f"{rel}: no asserts")
            markdown[rel] = '\n'.join(''.join(c['source']) for c in nb['cells']
                                      if c['cell_type'] == 'markdown')
            for i, c in enumerate(code):
                for o in c.get('outputs', []):
                    if o.get('output_type') == 'error':
                        problems.append(f"{rel} cell#{i}: error output {o.get('ename')}")
                    if o.get('output_type') == 'stream' and o.get('name') == 'stderr' \
                            and 'Traceback' in ''.join(o.get('text', [])):
                        problems.append(f"{rel} cell#{i}: stderr traceback")
                    t = ''.join(o.get('text', [])) or ''.join(o.get('data', {}).get('text/plain', []))
                    if t:
                        outputs.append(t)
    readme = open(os.path.join(ROOT, 'README.md'), encoding='utf-8').read()
    return '\n'.join(outputs), markdown, readme, n_assert, problems


def main():
    all_out, markdown, readme, n_assert, problems = load()
    failures = list(problems)

    # The README quotes the assert count in more than one phrasing ("113 asserts",
    # "113 of them"). Checking only the first phrasing let a stale second one through
    # once already, so match every construction that states a count.
    stated = {int(m) for pat in (r'(\d+)\s+asserts?', r'asserts?[^.]{0,40}?(\d+)\s+of them')
              for m in re.findall(pat, readme)}
    if not stated:
        failures.append("README no longer states an assert count at all")
    for n in sorted(stated - {n_assert}):
        failures.append(f"README says {n} asserts somewhere; the notebooks contain {n_assert}")

    for c in CLAIMS:
        for s in c.output:
            if s not in all_out:
                failures.append(f"{c.what}\n      no notebook output contains {s!r}")
        for s in c.readme:
            if s not in readme:
                failures.append(f"{c.what}\n      README no longer contains {s!r}")
        for nb_key, s in c.prose:
            hit = [k for k in markdown if nb_key in k]
            if not hit:
                failures.append(f"{c.what}\n      no notebook matching {nb_key!r}")
            elif s not in markdown[hit[0]]:
                failures.append(f"{c.what}\n      {hit[0]} prose no longer contains {s!r}")

    print(f"{len(CLAIMS)} pinned claims | {n_assert} asserts across the notebooks")
    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures:
            print("  -", f)
        print("\nEither the result moved and the prose needs updating, or a claim in this file "
              "is stale. Do not 'fix' this by deleting the claim.")
        return 1
    print("all claims backed by committed notebook output; no error outputs; every notebook asserts")
    return 0


if __name__ == '__main__':
    sys.exit(main())
