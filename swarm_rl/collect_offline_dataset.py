import argparse
import json
import os
from collections import defaultdict

import h5py
import numpy as np

from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env_multi
from swarm_rl.train import parse_swarm_cfg


class RandomBehaviorPolicy:
    def __init__(self, env):
        self.env = env

    def act(self, obs):
        return [self.env.action_space.sample() for _ in range(self.env.num_agents)]


def _infer_cost(info):
    rewards = info.get("rewards", {}) if isinstance(info, dict) else {}
    # Treat any hard collision or crash as cost=1.0, else 0.0.
    collision_signals = [
        rewards.get("rewraw_quadcol", 0.0),
        rewards.get("rewraw_quadcol_obstacle", 0.0),
        rewards.get("rew_crash", 0.0),
    ]
    return float(any(v < 0 for v in collision_signals))


def write_h5(output_dir, buffers):
    os.makedirs(output_dir, exist_ok=True)
    for agent_id, data in enumerate(buffers):
        file_path = os.path.join(output_dir, f"agent_{agent_id}.h5")
        with h5py.File(file_path, "w") as f:
            f.create_dataset("observations", data=np.asarray(data["observations"], dtype=np.float32))
            f.create_dataset("actions", data=np.asarray(data["actions"], dtype=np.float32))
            f.create_dataset("rewards", data=np.asarray(data["rewards"], dtype=np.float32))
            f.create_dataset("costs", data=np.asarray(data["costs"], dtype=np.float32))
            f.create_dataset("next_observations", data=np.asarray(data["next_observations"], dtype=np.float32))
            f.create_dataset("terminals", data=np.asarray(data["terminals"], dtype=np.bool_))
            f.create_dataset("timeouts", data=np.asarray(data["timeouts"], dtype=np.bool_))


def write_metadata(output_dir, metadata):
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Collect MOSDB-style multi-agent offline dataset for quadrotor env")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--max_episode_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)

    args, sf_argv = parser.parse_known_args()

    cfg = parse_swarm_cfg(argv=sf_argv, evaluation=True)
    env = make_quadrotor_env_multi(cfg)
    policy = RandomBehaviorPolicy(env)

    agent_num = env.num_agents
    buffers = [defaultdict(list) for _ in range(agent_num)]

    max_episode_steps = args.max_episode_steps
    if max_episode_steps is None:
        max_episode_steps = int(env.ep_len)

    np.random.seed(args.seed)

    total_steps = 0
    episode_lengths = []

    obs = env.reset()
    for ep in range(args.num_episodes):
        if ep > 0:
            obs = env.reset()

        ep_steps = 0
        for _ in range(max_episode_steps):
            actions = policy.act(obs)
            next_obs, rewards, dones, infos = env.step(actions)

            terminal = bool(any(dones))
            timeout = bool(ep_steps + 1 >= max_episode_steps and not terminal)

            for i in range(agent_num):
                buffers[i]["observations"].append(np.asarray(obs[i], dtype=np.float32))
                buffers[i]["actions"].append(np.asarray(actions[i], dtype=np.float32))
                buffers[i]["rewards"].append(np.float32(rewards[i]))
                buffers[i]["costs"].append(np.float32(_infer_cost(infos[i])))
                buffers[i]["next_observations"].append(np.asarray(next_obs[i], dtype=np.float32))
                buffers[i]["terminals"].append(np.bool_(terminal))
                buffers[i]["timeouts"].append(np.bool_(timeout))

            obs = next_obs
            ep_steps += 1
            total_steps += 1

            if terminal or timeout:
                break

        episode_lengths.append(ep_steps)

    write_h5(args.output_dir, buffers)

    metadata = {
        "env": "quadrotor_multi",
        "num_agents": agent_num,
        "num_episodes": args.num_episodes,
        "max_episode_steps": max_episode_steps,
        "total_steps": total_steps,
        "seed": args.seed,
        "observation_dim": int(np.asarray(buffers[0]["observations"][0]).shape[-1]),
        "action_dim": int(np.asarray(buffers[0]["actions"][0]).shape[-1]),
        "episode_lengths": episode_lengths,
        "policy": "random_action_space_sample",
    }
    write_metadata(args.output_dir, metadata)
    env.close()


if __name__ == "__main__":
    main()
