"""Robomimic conversion with an audit of each actually loaded physical XML."""
from __future__ import annotations

import json
import os
from pathlib import Path
import runpy


def main():
    import mujoco
    import numpy as np
    from robomimic.envs.env_robosuite import EnvRobosuite
    from course_tools.backlash_physics import model_state_map

    original = EnvRobosuite.reset_to
    audits = []

    def reset_to(env, state):
        observation = original(env, state)
        if "model" in state:
            expected = mujoco.MjModel.from_xml_string(state["model"])
            actual = env.env.sim.model._model
            if model_state_map(expected) != model_state_map(actual):
                raise ValueError("live rendering model joint map differs from recorded XML")
            for name in ("jnt_range", "jnt_stiffness", "dof_armature", "dof_damping", "eq_active0"):
                if not np.allclose(getattr(actual, name), getattr(expected, name), rtol=0, atol=1e-12):
                    raise ValueError(f"live rendering model changed {name}")
            audits.append({"state_map": model_state_map(actual), "physical_parameters_match": True})
        return observation

    EnvRobosuite.reset_to = reset_to
    EnvRobosuite.get_reward = lambda env: 0.0  # raw rewards/dones are copied
    runpy.run_module("robomimic.scripts.dataset_states_to_obs", run_name="__main__")
    if not audits:
        raise ValueError("converter never loaded an episode XML")
    Path(os.environ["COURSE_REPLAY_AUDIT_PATH"]).write_text(json.dumps(audits, indent=2))


if __name__ == "__main__":
    main()
