# stdlib
import itertools
import time
from collections import Counter, deque, namedtuple
from pathlib import Path
from typing import Optional, Tuple

# third party
import gym
import numpy as np
import tensorflow as tf
from gym import Wrapper, spaces
# first party
from gym.wrappers import TimeLimit
from tensorboard.plugins import projector
from tensorflow.contrib import summary

from environments.hindsight_wrapper import HindsightWrapper
from sac.agent import AbstractAgent
from sac.networks import MLPAgent
from sac.policies import CategoricalPolicy, GaussianPolicy
from sac.replay_buffer import ReplayBuffer
from sac.utils import Obs, Shape, Step, unwrap_env, \
    vectorize, space_to_size, get_space_attrs

Agents = namedtuple('Agents', 'train act')

SUCCESS_KWD = 'success'
LOG_COUNT_KWD = 'log count'


class Trainer:
    def __init__(self,
                 env: gym.Env,
                 seed: Optional[int],
                 buffer_size: int,
                 batch_size: int,
                 n_train_steps: int,
                 preprocess_func=None,
                 **kwargs):
        tf.enable_eager_execution()

        if seed is not None:
            np.random.seed(seed)
            tf.set_random_seed(seed)
            env.seed(seed)

        self.episodes = None
        self.episode_count = None
        self.n_train_steps = n_train_steps
        self.batch_size = batch_size
        self.env = env
        self.buffer = ReplayBuffer(buffer_size)
        self.preprocess_func = preprocess_func
        obs = env.reset()
        if preprocess_func is None and not isinstance(obs, np.ndarray):
            try:
                self.preprocess_func = unwrap_env(
                    env, lambda e: hasattr(e, 'preprocess_obs')).preprocess_obs
            except RuntimeError:
                self.preprocess_func = vectorize
        observation_space = spaces.Box(
            *[
                self.preprocess_obs(get_space_attrs(env.observation_space, attr))
                for attr in ['low', 'high']
            ],
            dtype=np.float32)

        self.action_space = env.action_space
        self.agent = self.build_agent(
            observation_space=observation_space, action_space=self.action_space, **kwargs)
        self.time_steps = tf.train.get_or_create_global_step()

    def train(self,
              load_path: Path,
              logdir: Path,
              render: bool = False,
              save_threshold: int = None):

        if logdir:
            writer = summary.create_file_writer(str(logdir))
            writer.set_as_default()

        # writer = None
        # if load_path:
        # raise NotImplementedError
        # # TODO
        # saver.restore(self.sess, load_path)
        # print("Model restored from", load_path)
        # if logdir:
        # writer = tf.summary.FileWriter(str(logdir), self.sess.graph) TODO
        # config = projector.ProjectorConfig()
        # projector.visualize_embeddings(writer, config)

        past_returns = deque(maxlen=save_threshold)
        best_average = -np.inf

        for episodes in itertools.count(1):
            self.episodes = episodes
            self.episode_count = self.run_episode(
                o1=self.reset(),
                render=render,
                eval_period=self.is_eval_period() and load_path is None)

            episode_return = self.episode_count['reward']

            # save model
            passes_save_threshold = True
            if save_threshold is not None:
                past_returns.append(episode_return)
                new_average = np.mean(past_returns)
                if new_average < best_average:
                    passes_save_threshold = False
                else:
                    best_average = new_average

            # TODO
            # if logdir and episodes % 10 == 1 and passes_save_threshold:
            # save_path = saver.save(self.sess, str(logdir.joinpath('model.ckpt')))
            # print("model saved in path:", saver.save(self.sess, save_path=save_path))
            # embed_save_path = embed_saver.save(
            #     self.sess, str(logdir.joinpath('embed', 'model.ckpt')))
            # print("embeddings saved in path:", embed_save_path)
            # saver.save(self.sess, str(save_path).replace('<episode>', str(episodes)))
            self.time_steps.assign_add(self.episode_count['time_steps'])

            print_statement = f'({"EVAL" if self.is_eval_period() else "TRAIN"}) ' \
                              f'Episode: {episodes}\t ' \
                              f'Time Steps: {int(self.time_steps)}\t ' \
                              f'Reward: {episode_return}\t ' \
                              f'Success: {self.episode_count[SUCCESS_KWD]}'
            print(print_statement)

            if logdir:
                with tf.contrib.summary.record_summaries_every_n_global_steps(1):
                    if self.is_eval_period():
                        summary.scalar('eval return', episode_return)
                    else:
                        for k, v in self.episode_count.items():
                            if np.isscalar(v):
                                summary.scalar(k.replace('_', ' '), v)

    def is_eval_period(self):
        return self.episodes % 100 == 0

    def trajectory(self, time_steps: int, final_index=None) -> Optional[Step]:
        if final_index is None:
            final_index = 0  # points to current time step
        else:
            final_index -= time_steps  # relative to start of episode
        if self.buffer.empty:
            return None
        return Step(*self.buffer[-time_steps:final_index])

    def run_episode(self, o1, eval_period, render):
        episode_count = Counter()
        episode_mean = Counter()
        tick = time.time()
        for time_steps in itertools.count(1):
            a = self.agent.get_actions(self.preprocess_obs(o1), sample=not eval_period)
            o2, r, t, info = self.step(a, render)
            if 'print' in info:
                print('Time step:', time_steps, info['print'])
            if not eval_period:
                episode_mean.update(self.perform_update())

            self.add_to_buffer(Step(o1=o1, a=a, r=r, o2=o2, t=t))
            o1 = o2
            # noinspection PyTypeChecker
            fps = 1 / float(time.time() - tick)
            episode_mean.update(Counter(fps=fps, **info.get('log mean', {})))
            # noinspection PyTypeChecker
            episode_count.update(
                Counter(reward=r, time_steps=1, **info.get('log count', {})))
            tick = time.time()
            if t:
                for k in episode_mean:
                    episode_count[k] = episode_mean[k] / float(time_steps)
                return episode_count

    def train_step(self, sample=None):
        sample = sample or self.sample_buffer()
        return self.agent.train_step(sample)

    def perform_update(self):
        counter = Counter()
        if self.buffer_ready():
            for i in range(self.n_train_steps):
                counter.update(
                    Counter({
                        k.replace('_', ' '): v / self.n_train_steps
                        for k, v in self.train_step().items() if np.isscalar(v)
                    }))
        return counter

    def build_agent(self, observation_space: gym.Space, action_space: gym.Space,
                    **kwargs) -> AbstractAgent:
        if isinstance(action_space, spaces.Discrete):
            policy_type = CategoricalPolicy
        else:
            policy_type = GaussianPolicy

        class Agent(policy_type, MLPAgent):
            def __init__(self):
                super(Agent, self).__init__(
                    o_size=observation_space.shape[0],
                    a_size=space_to_size(action_space),
                    **kwargs)

        return Agent()  # type: MLPAgent

    def reset(self) -> Obs:
        self.episode_count = None
        return self.env.reset()

    def step(self, action: np.ndarray, render: bool) -> Tuple[Obs, float, bool, dict]:
        """ Preprocess action before feeding to env """
        if render:
            self.env.render()
        if type(self.action_space) is spaces.Discrete:
            # noinspection PyTypeChecker
            s, r, t, i = self.env.step(np.argmax(action))
        else:
            s, r, t, i = self.env.step(fit_to_space(action, space=self.action_space))
        if isinstance(self.env, TimeLimit) and self.env._max_episode_steps and t:
            if LOG_COUNT_KWD not in i:
                i[LOG_COUNT_KWD] = dict()
            if SUCCESS_KWD not in i[LOG_COUNT_KWD]:
                i[LOG_COUNT_KWD][SUCCESS_KWD] = dict()
            i[LOG_COUNT_KWD][
                SUCCESS_KWD] = 0 < self.env._elapsed_steps < self.env._max_episode_steps
        return s, r, t, i

    def preprocess_obs(self, obs, shape: Shape = None):
        if self.preprocess_func is not None:
            obs = self.preprocess_func(obs, shape)
        return obs

    def add_to_buffer(self, step: Step) -> None:
        assert isinstance(step, Step)
        self.buffer.append(step)

    def buffer_ready(self):
        return len(self.buffer) >= self.batch_size

    def sample_buffer(self, batch_size=None) -> Step:
        batch_size = batch_size or self.batch_size
        sample = Step(*self.buffer.sample(batch_size))
        # leave state as dummy value for non-recurrent
        shape = [batch_size, -1]
        return Step(
            o1=self.preprocess_obs(sample.o1, shape=shape),
            o2=self.preprocess_obs(sample.o2, shape=shape),
            a=sample.a,
            r=sample.r,
            t=sample.t)


class HindsightTrainer(Trainer):
    def __init__(self, env: Wrapper, n_goals: int, **kwargs):
        self.n_goals = n_goals
        self.hindsight_env = unwrap_env(env, lambda e: isinstance(e, HindsightWrapper))
        assert isinstance(self.hindsight_env, HindsightWrapper)
        super().__init__(env=env, **kwargs)

    def add_hindsight_trajectories(self, time_steps: int) -> None:
        assert isinstance(self.hindsight_env, HindsightWrapper)
        if time_steps > 0:
            trajectory = self.trajectory(time_steps=time_steps)
            new_trajectory = self.hindsight_env.recompute_trajectory(trajectory)
            self.buffer.append(new_trajectory)
        if self.n_goals - 1 and time_steps > 1:
            final_indexes = np.random.randint(1, time_steps, size=self.n_goals - 1)
            assert isinstance(final_indexes, np.ndarray)

            for final_index in final_indexes:
                traj = self.trajectory(time_steps=time_steps, final_index=final_index)
                new_traj = self.hindsight_env.recompute_trajectory(traj)
                self.buffer.append(new_traj)

    def reset(self) -> Obs:
        time_steps = 0 if self.episode_count is None else self.episode_count['time_steps']
        self.add_hindsight_trajectories(time_steps)
        return super().reset()


def fit_to_space(x: np.ndarray, space: spaces.Box) -> np.ndarray:
    return (np.tanh(x) + 1) / 2 * (space.high - space.low) + space.low
