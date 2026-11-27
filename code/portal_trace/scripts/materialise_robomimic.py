"""Compatibility entry point for robomimic 0.5.0 state replay.

Robomimic calls ``EnvRobosuite.get_reward()`` while replaying every state even
when ``--copy_rewards`` will replace that value. Robosuite 1.5.2's Stack task
made its unused ``action`` parameter mandatory, so that probe raises before the
copy can happen. PORTAL-Trace always copies both rewards and dones from the raw
trajectory; returning a placeholder here therefore cannot enter the published
dataset.
"""

from __future__ import annotations

import runpy


def main() -> None:
    from robomimic.envs.env_robosuite import EnvRobosuite

    def unused_replay_reward(_env: EnvRobosuite) -> float:
        return 0.0

    EnvRobosuite.get_reward = unused_replay_reward
    runpy.run_module("robomimic.scripts.dataset_states_to_obs", run_name="__main__")


if __name__ == "__main__":
    main()
