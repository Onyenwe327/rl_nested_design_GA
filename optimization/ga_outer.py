# optimization/ga_outer.py
# -*- coding: utf-8 -*-
"""
GA outer-loop (DEAP) to search morphology + actuation design variables.
"""

from __future__ import annotations
import os
import time
import random
import pickle
from typing import Tuple, List, Dict, Sequence, Optional
import copy
from pathlib import Path
from deap import base, creator, tools
# Project imports
from pipeline.param_schema import (
    MOTOR_TYPES, GEAR_RATIOS,
    LINK_MIN, LINK_COUNT, LINK_SUM_MAX,
    validate_params,
)
from pipeline.train_and_eval import train_and_eval
from pipeline.post_training import run_post_training_bundle
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.spaces.box")

# ----------------------------
# GA configuration
# ----------------------------
POP_SIZE = 10             # increase for bigger sweeps
N_GEN = 10                # 10–30 for serious runs
CXPB = 0.6                # crossover probability
MUTPB = 0.4               # mutation probability (per-individual)
INDPB = 0.3               # mutation probability (per-gene inside individual)
SEED = 0
CHECKPOINT_PATH = "ga_ckpt.pkl"
HOF_SIZE = 5
TOTAL_TIME_SEC = 0.0

RUN_ROOT = Path("runs") / "ga_session"

RL_KW = dict(
    algo="PPO",
    total_timesteps=2_000_000,   # keep modest for faster iterations; raise for fidelity
    eval_episodes=5,
    log_dir=str(RUN_ROOT),
    log_file="ga_eval.log",
    num_envs=18,                # parallel envs; adjust per CPU cores
    device="cpu",
)

# ----------------------------
# Genome spec helper
# ----------------------------
GENOME = {
    "L_idx": [0, 1, 2],
    "motor_idx": 3,
    "gear_idx": 4,
}

# ----------------------------
# Tracking: evaluated artifacts
# ----------------------------
EVALUATED_ARTIFACTS: Dict[Tuple[float, float, float, int, int], Path] = {}
EVALUATED_POOL_RAW: List[Tuple[List[float], Path]] = []  # exact genomes list

# ----------------------------
# Utilities
# ----------------------------
def _repair_lengths(ind: List[float]) -> None:
    Ls = [max(LINK_MIN, float(ind[i])) for i in GENOME["L_idx"]]
    s = sum(Ls)
    if s > LINK_SUM_MAX:
        slack = LINK_SUM_MAX - LINK_COUNT * LINK_MIN
        curr_slack = s - LINK_COUNT * LINK_MIN
        scale = slack / max(curr_slack, 1e-9)
        Ls = [LINK_MIN + (Li - LINK_MIN) * scale for Li in Ls]
    for k, i in enumerate(GENOME["L_idx"]):
        ind[i] = float(Ls[k])

def _clip_categorical(ind: List[float]) -> None:
    mN = len(MOTOR_TYPES)
    gN = len(GEAR_RATIOS)
    ind[GENOME["motor_idx"]] = int(max(0, min(mN - 1, round(ind[GENOME["motor_idx"]]))))
    ind[GENOME["gear_idx"]] = int(max(0, min(gN - 1, round(ind[GENOME["gear_idx"]]))))

def _decode(ind: Sequence[float]) -> Dict:
    Ls = [float(ind[i]) for i in GENOME["L_idx"]]
    m_id = int(ind[GENOME["motor_idx"]])
    g_id = int(ind[GENOME["gear_idx"]])
    return {
        "dof_per_leg": 3,
        "link_lengths": Ls,
        "motor_type": MOTOR_TYPES[m_id],
        "gear_ratio": GEAR_RATIOS[g_id],
    }

def ind_to_str(ind: Sequence[float]) -> str:
    Ls = [float(ind[i]) for i in GENOME["L_idx"]]
    m_id = int(ind[GENOME["motor_idx"]])
    g_id = int(ind[GENOME["gear_idx"]])
    m_name = MOTOR_TYPES[m_id] if 0 <= m_id < len(MOTOR_TYPES) else f"id{m_id}"
    g_val = GEAR_RATIOS[g_id] if 0 <= g_id < len(GEAR_RATIOS) else f"id{g_id}"
    return f"L=[{Ls[0]:.3f},{Ls[1]:.3f},{Ls[2]:.3f}]  motor={m_name}  gear={g_val}"

def _genome_key(ind: Sequence[float]) -> Tuple[float, float, float, int, int]:
    Ls = [round(float(ind[i]), 6) for i in GENOME["L_idx"]]
    return (Ls[0], Ls[1], Ls[2], int(ind[GENOME["motor_idx"]]), int(ind[GENOME["gear_idx"]]))

# ----------------------------
# DEAP Operators
# ----------------------------
def clip_and_repair(ind: List[float]) -> List[float]:
    _repair_lengths(ind)
    _clip_categorical(ind)
    return ind

def evaluate_individual(ind) -> Tuple[float]:
    clip_and_repair(ind)
    params = _decode(ind)

    if not validate_params(params):
        return (-1e9,)

    global TOTAL_TIME_SEC, EVALUATED_ARTIFACTS, EVALUATED_POOL_RAW

    try:
        t0 = time.time()
        result = train_and_eval(
            params,
            seed=random.randint(0, 10_000),
            return_info=True,
            **RL_KW,
        )
        elapsed = time.time() - t0

        if isinstance(result, tuple):
            score, info = result
            score = float(score)
            add_sec = float(info.get("elapsed_sec", elapsed)) if isinstance(info, dict) else elapsed
            TOTAL_TIME_SEC += add_sec
            print(f"  ⏱ elapsed={add_sec:.3f}s  total={TOTAL_TIME_SEC:.3f}s")

            artifact_dir = None
            if isinstance(info, dict):
                for k in ("artifact_dir", "out_dir", "model_dir", "save_dir"):
                    v = info.get(k, None)
                    if v:
                        artifact_dir = Path(v)
                        break
                if artifact_dir is None and "log_dir" in info:
                    candidate = Path(info["log_dir"])
                    if candidate.exists():
                        artifact_dir = candidate

            if artifact_dir is not None:
                EVALUATED_ARTIFACTS[_genome_key(ind)] = Path(artifact_dir)
                EVALUATED_POOL_RAW.append((list(ind), Path(artifact_dir)))

        else:
            score = float(result)
            TOTAL_TIME_SEC += elapsed
            print(f"  ⏱ elapsed={elapsed:.3f}s  total={TOTAL_TIME_SEC:.3f}s")

    except Exception as e:
        print(f"[WARN] train_and_eval failed for {ind_to_str(ind)}: {e}")
        score = -1e6

    return (score,)

def mutate_individual(ind, indpb=INDPB):
    for i in GENOME["L_idx"]:
        if random.random() < indpb:
            ind[i] = float(ind[i] + random.gauss(0.0, 0.05))
    if random.random() < 0.2:
        if random.random() < 0.5:
            ind[GENOME["motor_idx"]] = random.randint(0, len(MOTOR_TYPES) - 1)
        else:
            ind[GENOME["motor_idx"]] = int(ind[GENOME["motor_idx"]] + random.choice([-1, 1]))
    if random.random() < 0.2:
        if random.random() < 0.5:
            ind[GENOME["gear_idx"]] = random.randint(0, len(GEAR_RATIOS) - 1)
        else:
            ind[GENOME["gear_idx"]] = int(ind[GENOME["gear_idx"]] + random.choice([-1, 1]))
    clip_and_repair(ind)
    return (ind,)

# ----------------------------
# Checkpoint helpers
# ----------------------------
def maybe_resume(checkpoint_path: str):
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "rb") as f:
                data = pickle.load(f)
            print(f"[INFO] Resuming from checkpoint: gen={data.get('gen','?')}")
            return data
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e}")
    return None

# ----------------------------
# Eval env factory (for post-training bundle)
# ----------------------------
def eval_env_maker_for(genome, artifact_dir):
    from pipeline.env_factory import make_env_from_xml
    artifact_dir = Path(artifact_dir)
    xmls = list(artifact_dir.glob("*.xml"))
    if not xmls:
        raise FileNotFoundError(f"No XML found under {artifact_dir}.")
    xml_path = str(xmls[0])
    link_lengths = [float(genome[0]), float(genome[1]), float(genome[2])]
    params = {"link_lengths": link_lengths}
    env = make_env_from_xml(xml_path=xml_path, seed=404, params=params)
    return env
# from pipeline.param_schema import MOTOR_TYPES, GEAR_RATIOS, ...
from pipeline.xml_builder import build_xml_from_params

# def eval_env_maker_for(genome, artifact_dir):
#     from pipeline.env_factory import make_env_from_xml
#     artifact_dir = Path(artifact_dir)
#     artifact_dir.mkdir(parents=True, exist_ok=True)

#     params = _decode(genome)

#     xml_path = build_xml_from_params(
#         params=params,
#         out_dir=artifact_dir,
#         name_hint="eval_model",
#     )

#     env = make_env_from_xml(
#         xml_path=str(xml_path),
#         seed=404,
#         params=params,
#     )
#     return env

# ----------------------------
# Main
# ----------------------------
def main(resume: bool = False):
    random.seed(SEED)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    L_guess = LINK_SUM_MAX / LINK_COUNT
    def attr_L():
        return max(LINK_MIN, random.uniform(LINK_MIN, max(LINK_MIN + L_guess, LINK_MIN + 1e-6)))

    toolbox.register("attr_L1", attr_L)
    toolbox.register("attr_L2", attr_L)
    toolbox.register("attr_L3", attr_L)
    toolbox.register("attr_motor", random.randint, 0, len(MOTOR_TYPES) - 1)
    toolbox.register("attr_gear", random.randint, 0, len(GEAR_RATIOS) - 1)
    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        (toolbox.attr_L1, toolbox.attr_L2, toolbox.attr_L3, toolbox.attr_motor, toolbox.attr_gear),
        n=1,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.3)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=3)

    ckpt = maybe_resume(CHECKPOINT_PATH) if resume else None
    if ckpt:
        pop = ckpt["pop"]
        hof = ckpt["hof"]
        random.setstate(ckpt["rndstate"])
        start_gen = ckpt["gen"] + 1
        global TOTAL_TIME_SEC, EVALUATED_ARTIFACTS, EVALUATED_POOL_RAW
        TOTAL_TIME_SEC = float(ckpt.get("total_time_sec", 0.0))
        EVALUATED_ARTIFACTS = ckpt.get("evaluated_artifacts", {}) or {}
        EVALUATED_POOL_RAW = ckpt.get("evaluated_pool_raw", []) or []
        EVALUATED_ARTIFACTS = {tuple(k): Path(v) for k, v in EVALUATED_ARTIFACTS.items()}
        print(f"[INFO] Resumed gen {start_gen} | total_time_sec={TOTAL_TIME_SEC:.3f}")
    else:
        pop = toolbox.population(n=POP_SIZE)
        for ind in pop:
            clip_and_repair(ind)
        hof = tools.HallOfFame(HOF_SIZE)
        start_gen = 1

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda fits: sum(fits) / len(fits))
    stats.register("max", max)
    stats.register("min", min)

    unevaluated = [ind for ind in pop if not ind.fitness.valid]
    if unevaluated:
        print("Evaluating initial population...")
        for ind in unevaluated:
            ind.fitness.values = toolbox.evaluate(ind)
            print(f"  {ind_to_str(ind)} -> score={ind.fitness.values[0]:.3f}")

    print("Starting evolution...")
    for gen in range(start_gen, N_GEN + 1):
        print(f"\n-- Generation {gen} --")
        offspring = [copy.deepcopy(ind) for ind in toolbox.select(pop, len(pop))]

        # Crossover
        for i in range(1, len(offspring), 2):
            if random.random() < CXPB:
                tools.cxBlend(offspring[i - 1], offspring[i], alpha=0.3)
                clip_and_repair(offspring[i - 1])
                clip_and_repair(offspring[i])
                for j in (i - 1, i):
                    # Invalidate fitness by deleting the attribute (NOT assigning an empty tuple)
                    try:
                        del offspring[j].fitness.values
                    except AttributeError:
                        pass

        # Mutation
        for ind in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(ind)
                try:
                    del ind.fitness.values  # Invalidate properly
                except AttributeError:
                    pass

        # Evaluate only invalid fitness
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        print(f"Gen {gen}: evaluating {len(invalid)} new individuals")
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        # Replacement
        pop[:] = offspring
        hof.update(pop)

        record = stats.compile(pop)
        print(f"Stats: avg={record['avg']:.3f}  max={record['max']:.3f}")

        # Checkpoint
        with open(CHECKPOINT_PATH, "wb") as f:
            pickle.dump(dict(
                pop=pop, gen=gen, hof=hof, rndstate=random.getstate(),
                total_time_sec=TOTAL_TIME_SEC,
                evaluated_artifacts=EVALUATED_ARTIFACTS,
                evaluated_pool_raw=EVALUATED_POOL_RAW,
            ), f, protocol=pickle.HIGHEST_PROTOCOL)

    print("\nTop solutions (HOF):")
    for i, ind in enumerate(hof):
        print(f"[{i}] {ind_to_str(ind)}  fitness={ind.fitness.values[0]:.3f}")
    print(f"\n🏁 Total GA training time: {TOTAL_TIME_SEC/60:.2f} min ({TOTAL_TIME_SEC:.1f} s)")

    # -----------------------------
    # POST-TRAINING AUTOMATION
    # -----------------------------
    evaluated_pool: List[Tuple[Sequence[float], Path]] = list(EVALUATED_POOL_RAW)
    seen = {tuple(g) for (g, _) in evaluated_pool}
    for ind in list(pop) + list(hof):
        key = _genome_key(ind)
        if key in EVALUATED_ARTIFACTS:
            g_exact = [float(ind[i]) for i in GENOME["L_idx"]] + [
                int(ind[GENOME["motor_idx"]]), int(ind[GENOME["gear_idx"]])]
            if tuple(g_exact) not in seen:
                evaluated_pool.append((g_exact, EVALUATED_ARTIFACTS[key]))
                seen.add(tuple(g_exact))

    if not evaluated_pool:
        print("[POST] No evaluated artifacts found; skip post-training bundle.")
        return pop, hof

    post_dir = RUN_ROOT / "post_training"
    post_dir.mkdir(parents=True, exist_ok=True)

    print(f"[POST] Running post-training bundle on {len(evaluated_pool)} models...")
    artifacts = run_post_training_bundle(
        hof=hof,
        evaluated_pool=evaluated_pool,
        env_maker_for=eval_env_maker_for,
        out_dir=post_dir,
        rng_seed=404,
        max_eval_steps=1500,
        eval_fps=30,
        overlay_velocity_on_video=True,
    )

    print("[POST] Velocity plot:", artifacts["velocity_plot"])
    print("[POST] Torque plot  :", artifacts["torque_plot"])
    print("[POST] Videos dir   :", artifacts["videos_dir"])
    return pop, hof


if __name__ == "__main__":
    os.makedirs(".", exist_ok=True)
    main(resume=False)
