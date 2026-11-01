"""sim2gym-vf: AccelGym as a Prime Intellect `verifiers` (v1) environment.

    vf-eval sim2gym-vf -m <model> --env.taskset.path tasks.jsonl
    vf-eval sim2gym-vf -m <model> --env.taskset.generate-seed 7 --env.taskset.generate-n 12

The episode is the same `Sim2GymEnv` the runner, baselines and live demo use: the task prompt opens the
exchange, each model reply is stepped through the env, and the env's observation (simulator result, error
or retry message) comes back as the next user turn. The final score in [0, 1] is the reward, graded
against the exhaustive SCALE-Sim optimum, so verifiers training and sim2gym evaluation share one scoring path.
"""

from pathlib import Path

import verifiers.v1 as vf

from sim2gym.env import Sim2GymEnv
from sim2gym.tasks import load_tasks, public_task


class Sim2GymConfig(vf.TasksetConfig):
    path: str = "tasks.jsonl"
    """Tasks file from `sim2gym build` or `sim2gym generate`; ignored when `generate_seed` is set."""
    generate_seed: int | None = None
    """Sample fresh layers with this seed instead of reading `path` (runs SCALE-Sim on cache misses)."""
    generate_n: int = 12
    """Number of generated layers; each gives one task per PE budget."""


class Sim2GymData(vf.TaskData):
    task: dict
    """The full sim2gym task, optimum included; only the env reads it, the model sees `prompt`."""


class Sim2GymTask(vf.Task[Sim2GymData, vf.State, vf.TaskConfig]):
    pass


class Sim2GymTaskset(vf.Taskset[Sim2GymTask, Sim2GymConfig]):
    def load(self):
        if self.config.generate_seed is not None:
            from sim2gym.generate import generate_tasks
            tasks = generate_tasks(self.config.generate_seed, self.config.generate_n)
        else:
            tasks = load_tasks(Path(self.config.path))
        for task in tasks:
            yield Sim2GymTask(Sim2GymData(id=task["id"], name=task["id"], prompt=task["prompt"], task=task),
                              self.config.task)


class Sim2GymEnvConfig(vf.EnvConfig):
    # Tool-less chat: the protocol is plain JSON in the reply. Sim2GymEnv enforces its own
    # max turns (budget + 2), so the harness limit only needs to sit above it.
    player: vf.AgentConfig = vf.AgentConfig(harness={"id": "null"}, max_turns=16)


class Sim2GymVfEnv(vf.Env[Sim2GymEnvConfig]):
    async def run(self, task, agents):
        sim = Sim2GymEnv([task.data.task], budget=None)  # one env per episode: rollouts run concurrently
        sim.reset(0)
        reward, info = 0.0, None
        async with agents.player.interaction(task) as interaction:
            segment = await interaction.turn()
            while not segment.terminated:
                obs, reward, done, info = sim.step(segment.last_reply)
                if done:
                    break
                segment = await interaction.turn(obs)
        trace = interaction.trace
        trace.record_reward("sim2gym_score", reward)
        grade = (info or {}).get("grade") or {}
        trace.record_metric("solved", float(bool(grade.get("solved"))))
        trace.record_metric("feasible", float(bool(grade.get("feasible"))))
        trace.record_metric("sim_calls_used", float((info or {}).get("sim_calls_used", 0)))
        trace.info["sim2gym"] = {"task": public_task(task.data.task)["id"], "answer": (info or {}).get("answer"),
                                 "error": (info or {}).get("error"), "sim_log": (info or {}).get("sim_log", [])}


__all__ = ["Sim2GymConfig", "Sim2GymData", "Sim2GymTask", "Sim2GymTaskset", "Sim2GymEnvConfig", "Sim2GymVfEnv"]
