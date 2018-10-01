from collections.__init__ import deque, Counter

import numpy as np
import tensorflow as tf

from environments.hierarchical_wrapper import Hierarchical, HierarchicalAgents
from environments.hindsight_wrapper import HSRHindsightWrapper
from sac.train import Trainer, HindsightTrainer, Agents, BossState, squash_to_space
from sac.utils import Step


class UnsupervisedTrainer(Trainer):
    # noinspection PyMissingConstructor
    def __init__(self,
                 env: HSRHindsightWrapper,
                 sess: tf.Session,
                 worker_kwargs: dict,
                 boss_kwargs: dict,
                 boss_freq: int = None,
                 update_worker: bool = True,
                 n_goals: int = None,
                 **kwargs):

        self.update_worker = update_worker
        self.boss_state = None
        self.worker_o0 = None

        self.boss_freq = boss_freq
        self.reward_queue = deque(maxlen=boss_freq)
        self.use_entropy_reward = boss_freq is None
        if boss_freq:
            X = np.ones((boss_freq, 2))
            X[:, 1] = np.arange(boss_freq)
            self.reward_operator = np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T)

        self.env = env
        self.action_space = env.action_space
        self.sess = sess
        self.count = Counter(reward=0, episode=0, time_steps=0)
        self.episode_count = None
        self.time_steps = 0

        worker_kwargs = dict(
            observation_space=env.observation_space,
            action_space=env.action_space,
            env=env,
            sess=sess,
            name='worker',
            **worker_kwargs,
            **kwargs)
        if n_goals is None:
            worker_trainer = Trainer(**worker_kwargs)
        else:
            worker_trainer = HindsightTrainer(n_goals=n_goals, **worker_kwargs)
        boss_trainer = Trainer(
            observation_space=env.goal_space,
            action_space=env.goal_space,
            preprocess_func=lambda obs, shape: obs,
            env=env,
            sess=sess,
            name='boss',
            **boss_kwargs,
            **kwargs,
        )
        self.trainers = Hierarchical(boss=boss_trainer, worker=worker_trainer)

        self.agents = Agents(
            act=HierarchicalAgents(
                boss=self.trainers.boss.agents.act,
                worker=self.trainers.worker.agents.act,
                initial_state=0),
            train=HierarchicalAgents(
                boss=self.trainers.boss.agents.train,
                worker=self.trainers.worker.agents.train,
                initial_state=0))

    def boss_turn(self, episodes=None):
        if self.boss_freq:
            if episodes is None:
                episodes = self.episode_count['episode']
            return episodes % self.boss_freq == 0
        return self.time_steps == 0

    def get_actions(self, o1, s):
        # boss
        if self.boss_turn():
            goal_delta = self.trainers.boss.get_actions(o1.achieved_goal, s).output
            goal = squash_to_space(o1.achieved_goal + goal_delta,
                                   space=self.env.goal_space)
            self.env.hsr_env.set_goal(goal)
            self.boss_state = BossState(
                goal=goal, action=goal_delta, o0=o1.achieved_goal, v0=None)

        # worker
        self.worker_o0 = o1.replace(desired_goal=self.boss_state.goal)

        return self.trainers.worker.get_actions(self.worker_o0, s)

    def perform_update(self, _=None):
        boss_update = self.trainers.boss.perform_update(
            train_values=self.agents.act.boss.default_train_values + ['v1'])
        if self.boss_turn():
            self.boss_state = self.boss_state._replace(v0=boss_update['boss/v1'])

        if self.update_worker:
            worker_update = {
                f'worker_{k}': v
                for k, v in (self.trainers.worker.perform_update() or {}).items()
            }
        else:
            worker_update = dict()

        self.time_steps += 1
        return {
            **{f'boss_{k}': v
               for k, v in (boss_update or {}).items()},
            **worker_update
        }

    def trajectory(self, time_steps: int, final_index=None):
        raise NotImplementedError

    def step(self, action: np.ndarray, render: bool):
        o2, r, t, info = super().step(action=action, render=render)
        self.reward_queue += [r]
        return o2, r, t, info

    def add_to_buffer(self, step: Step):
        # boss
        if self.use_entropy_reward:
            boss_turn = step.t
        else:
            boss_turn = step.t and self.boss_turn(self.count['episode'] + 1)
        if boss_turn:
            if self.use_entropy_reward:
                p = self.boss_state.v0 / .99**self.time_steps
                if step.r == 0:
                    p = 1 - p
                boss_reward = -np.log(p)
            else:
                rewards = np.array(self.reward_queue)
                boss_reward = np.matmul(self.reward_operator, rewards)[1]
            self.trainers.boss.buffer.append(
                step.replace(
                    o1=self.boss_state.o0, a=self.boss_state.action, r=boss_reward))

        # worker
        step = step.replace(o1=self.worker_o0, o2=step.o2)
        self.trainers.worker.buffer.append(step)

    def reset(self):
        self.time_steps = 0
        return super().reset()


def regression_slope1(Y):
    Y = np.array(Y)
    X = np.ones((Y.size, 2))
    X[:, 1] = np.arange(Y.size)
    return np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))


def regression_slope2(Y):
    Y = np.array(Y)
    X = np.arange(Y.size)
    normalized_X = X - X.mean()
    return np.sum(normalized_X * Y) / np.sum(normalized_X**2)