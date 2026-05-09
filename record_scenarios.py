"""
record_scenarios.py — Long-tail demo scenario recorder for SoTA Commission I.

Records 5 distinct driving scenarios in MetaDrive 0.4.3 using the built-in
ExpertPolicy. Each scenario opens a fresh window; you press Win+Alt+R (Xbox
Game Bar) when the window appears to start recording, and again when it closes
to stop. Game Bar saves to C:\\Users\\<you>\\Videos\\Captures\\.

Usage:
    cd D:\\affordance-driver
    .\\carla-env\\Scripts\\activate          # if the venv isn't already active
    python record_scenarios.py

Output target: D:\\affordance-driver\\sim\\<scenario_name>.mp4 (you move/rename
the Game Bar recording into place between scenarios).
"""

import time
import sys
import traceback
from pathlib import Path

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.expert_policy import ExpertPolicy

# Optional — used only by the "construction_longtail" scenario.
# If it's not in this MetaDrive build, that one scenario is skipped gracefully.
try:
    from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
    HAS_SAFE_ENV = True
except Exception:
    HAS_SAFE_ENV = False

SIM_DIR = Path(r"D:\affordance-driver\sim")
SIM_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Scenarios — each entry tunes MetaDrive to elicit a distinct long-tail visual.
# ---------------------------------------------------------------------------
SCENARIOS = [
    {
        "name": "highway_traffic",
        "blurb": "Multi-segment straight highway with moderate traffic. Expert overtakes.",
        "env_cls": "base",
        "config": {
            "map": "SSSSS",
            "traffic_density": 0.15,
            "start_seed": 7,
            "random_traffic": True,
        },
        "duration_steps": 400,
    },
    {
        "name": "curvy_road",
        "blurb": "Curved highway segments. Expert tracks lane through turns.",
        "env_cls": "base",
        "config": {
            "map": "SCCC",
            "traffic_density": 0.1,
            "start_seed": 13,
            "random_traffic": True,
        },
        "duration_steps": 400,
    },
    {
        "name": "dense_traffic_cutin",
        "blurb": "Dense traffic — vehicles change lanes around ego, expert reacts.",
        "env_cls": "base",
        "config": {
            "map": "SSSS",
            "traffic_density": 0.35,
            "start_seed": 22,
            "random_traffic": True,
        },
        "duration_steps": 400,
    },
    {
        "name": "intersection",
        "blurb": "Crossing intersection with cross-traffic. Expert navigates turn.",
        "env_cls": "base",
        "config": {
            "map": "SXS",
            "traffic_density": 0.15,
            "start_seed": 5,
            "random_traffic": True,
        },
        "duration_steps": 450,
    },
    {
        "name": "construction_longtail",
        "blurb": "* THE WOD-E2E LONG-TAIL CASE: cone obstacles in lane (SafeMetaDriveEnv).",
        "env_cls": "safe",
        "config": {
            "map": "SSSS",
            "traffic_density": 0.1,
            "accident_prob": 1.0,
            "start_seed": 33,
            "random_traffic": True,
        },
        "duration_steps": 500,
    },
]


def make_env(scenario):
    """Construct a MetaDrive env for the given scenario spec."""
    cfg = {
        "use_render": True,
        "agent_policy": ExpertPolicy,
        "num_scenarios": 1,
        "window_size": (1000, 600),
        "horizon": scenario["duration_steps"] + 100,
        **scenario["config"],
    }

    if scenario["env_cls"] == "safe":
        if not HAS_SAFE_ENV:
            raise RuntimeError(
                "SafeMetaDriveEnv not available in this MetaDrive build."
            )
        return SafeMetaDriveEnv(cfg)
    return MetaDriveEnv(cfg)


def run_scenario(scenario, max_resets=1):
    """Open the MetaDrive window, drive for duration_steps, auto-reset on early term."""
    env = None
    try:
        env = make_env(scenario)
        obs, info = env.reset()

        steps_done = 0
        resets_used = 0
        target = scenario["duration_steps"]
        progress_every = max(target // 5, 50)

        print(f"  [running] target {target} steps "
              f"(~{target / 30:.0f} sec at 30 fps)")

        while steps_done < target:
            obs, reward, terminated, truncated, info = env.step([0, 0])
            steps_done += 1

            if steps_done % progress_every == 0:
                pct = 100 * steps_done / target
                print(f"  [running] {steps_done}/{target} steps  ({pct:.0f}%)")

            if terminated or truncated:
                if resets_used >= max_resets:
                    print(f"  [ending] hit {max_resets+1} terminations, stopping early "
                          f"at step {steps_done}")
                    break
                resets_used += 1
                print(f"  [reset] expert terminated; reset {resets_used}/{max_resets}")
                obs, info = env.reset()

        print(f"  [done] {steps_done} steps total")
        return True, None
    except Exception as e:
        return False, e
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def record_one(scenario, idx, total):
    print("\n" + "=" * 72)
    print(f"  SCENARIO {idx}/{total}: {scenario['name']}")
    print("=" * 72)
    print(f"  {scenario['blurb']}")
    print(f"  Target file: {SIM_DIR / (scenario['name'] + '.mp4')}")
    print()
    print("  STEPS:")
    print("    1. START your screen recorder NOW (before pressing Enter):")
    print("         - Snipping Tool video mode (Win+Shift+S, switch to video):")
    print("           drag a generous rectangle covering the center of the")
    print("           screen, then click 'Start'.")
    print("         - OR Game Bar: focus a window, then Win+Alt+R.")
    print("    2. Press Enter here to launch (3-sec countdown).")
    print("    3. MetaDrive window opens; expert drives ~20-25 sec; window closes.")
    print("    4. Stop your recorder.")
    print("    5. Save the .mp4, then rename/move it to:")
    print(f"         {SIM_DIR / (scenario['name'] + '.mp4')}")
    print()
    input("  Press Enter to launch (recorder running already) ...")
    print("  Launching in 3..."); time.sleep(1)
    print("  Launching in 2..."); time.sleep(1)
    print("  Launching in 1..."); time.sleep(1)
    print("  GO.\n")

    ok, err = run_scenario(scenario)

    if not ok:
        print("\n  [SCENARIO ERRORED]")
        print(f"  {type(err).__name__}: {err}")
        print("  This scenario will be skipped. Check error and continue.")
        traceback.print_exception(type(err), err, err.__traceback__, limit=3)
    else:
        print("\n  Window closed cleanly.")

    print("\n  STOP recording. Save the .mp4 and rename/move it to:")
    print(f"     {SIM_DIR / (scenario['name'] + '.mp4')}")
    input("  Press Enter when done to continue (or Ctrl+C to stop here) ... ")


def main():
    print("=" * 72)
    print("  SoTA Commission I — Long-Tail Scenario Recorder")
    print("=" * 72)
    print(f"  {len(SCENARIOS)} scenarios. ~13-17 sec of footage each.")
    print(f"  Output dir: {SIM_DIR}")
    print()

    # Warn if torch missing — MetaDrive falls back to a weaker numpy expert.
    try:
        import torch  # noqa: F401
        torch_available = True
    except Exception:
        torch_available = False
    if not torch_available:
        print("  [warning] PyTorch not detected. MetaDrive will use the numpy")
        print("            expert — drives okay on straight roads, often")
        print("            wanders off-road on curves and intersections.")
        print("            For better recordings, install torch (CPU build):")
        print("              pip install torch --index-url "
              "https://download.pytorch.org/whl/cpu")
        print("            ~3 min, ~200 MB. Then re-run this script.")
        print()

    print("  PRE-FLIGHT:")
    print("    [ ] Screen recorder ready (Snipping Tool video mode OR Game Bar)")
    print("    [ ] Test recording produced a playable .mp4 already")
    print("    [ ] Decide where MetaDrive window will land on screen")
    print("        (so you can pre-draw the Snipping Tool region around it)")
    print(f"    [ ] {SIM_DIR} exists  ({'YES' if SIM_DIR.exists() else 'NO'})")
    print(f"    [ ] SafeMetaDriveEnv available  ({'YES' if HAS_SAFE_ENV else 'NO — last scenario will skip'})")
    print()
    input("  Press Enter to begin ...")

    for i, scenario in enumerate(SCENARIOS, 1):
        record_one(scenario, i, len(SCENARIOS))

    print("\n" + "=" * 72)
    print(f"  DONE. Check  {SIM_DIR}  for the .mp4s.")
    print("=" * 72)


if __name__ == "__main__":
    main()