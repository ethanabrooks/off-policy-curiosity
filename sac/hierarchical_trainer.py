# stdlib
from collections import Counter, namedtuple

# third party
from gym.spaces import Box
import numpy as np
import tensorflow as tf

# first party
from sac.hindsight_wrapper import Observation
from sac.agent import NetworkOutput
from sac.train import Agents, Trainer


class HierarchicalTrainer(Trainer):
    # noinspection PyMissingConstructor
    def __init__(self, env, boss_act_freq: int, use_boss_oracle: bool,
                 use_worker_oracle: bool, sess: tf.Session, worker_kwargs, boss_kwargs,
                 correct_boss_action: bool, worker_gets_term_r: bool, **kwargs):

        self.worker_gets_term_r = worker_gets_term_r
        self.correct_boss_action = correct_boss_action
        self.use_worker_oracle = use_worker_oracle
        self.use_boss_oracle = use_boss_oracle
        self.boss_act_freq = boss_act_freq
        self.boss_state = None
        self.worker_o1 = None

        self.env = env
        self.sess = sess
        self.count = Counter(reward=0, episode=0, time_steps=0)
        self.episode_count = Counter()
        self.action_space = env.action_space.worker
        self.time_steps = 0

        def boss_preprocess_obs(obs, shape=None):
            obs = Observation(*obs)
            return vectorize([obs.achieved_goal, obs.desired_goal], shape)

        def worker_preprocess_obs(obs, shape=None):
            obs = Observation(*obs)
            return vectorize([obs.observation, obs.desired_goal], shape)

        self.trainers = Hierarchical(
            boss=Trainer(
                observation_space=env.observation_space.boss,
                action_space=env.action_space.boss,
                preprocess_func=boss_preprocess_obs,
                env=env,
                sess=sess,
                name='boss',
                **boss_kwargs,
                **kwargs,
            ),
            worker=Trainer(
                observation_space=env.observation_space.worker,
                action_space=env.action_space.worker,
                preprocess_func=worker_preprocess_obs,
                env=env,
                sess=sess,
                name='worker',
                **worker_kwargs,
                **kwargs,
            ))

        self.agents = Agents(
            act=HierarchicalAgents(
                boss=self.trainers.boss.agents.act,
                worker=self.trainers.worker.agents.act,
                initial_state=0),
            train=HierarchicalAgents(
                boss=self.trainers.boss.agents.train,
                worker=self.trainers.worker.agents.train,
                initial_state=0))

    def boss_turn(self):
        return self.time_steps % self.boss_act_freq == 0

    def get_actions(self, o1, s):
        # boss
        if self.boss_turn():
            if self.boss_oracle:
                action = boss_oracle(self.env)
            else:
                action = self.trainers.boss.get_actions(o1, s).output
            goal = o1.achieved_goal + self.env.boss_action_to_goal_space(action)
            self.boss_state = BossState(goal=goal, action=action, o1=o1)

        # worker
        direction = self.boss_state.goal - o1.achieved_goal
        self.worker_o1 = o1.replace(desired_goal=direction)

        if self.worker_oracle:
            oracle_action = worker_oracle(self.env, direction)
            return NetworkOutput(output=oracle_action, state=0)
        else:
            return self.trainers.worker.get_actions(self.worker_o1, s)

    def perform_update(self):
        return {
            **{
                f'boss_{k}': v
                for k, v in (self.trainers.boss.perform_update() or {}).items()
            },
            **{
                f'worker_{k}': v
                for k, v in (self.trainers.worker.perform_update() or {}).items()
            }
        }

    def trajectory(self, time_steps: int, final_index=None):
        raise NotImplementedError

    def add_to_buffer(self, step: Step):
        # boss
        if not self.boss_oracle and (self.boss_turn() or step.t):
            boss_action = self.boss_state.action
            if self.correct_boss_action:
                rel_step = step.o2.achieved_goal - self.boss_state.o1.achieved_goal
                boss_action = self.env.goal_to_boss_action_space(rel_step)
            self.trainers.boss.buffer.append(
                step.replace(o1=self.boss_state.o1, a=boss_action))

        # worker
        if not self.worker_oracle:
            direction = self.worker_o1.desired_goal
            if not (step.t and self.worker_gets_term_r):
                movement = vectorize(step.o2.achieved_goal) - vectorize(
                    step.o1.achieved_goal)
                step = step.replace(r=np.dot(direction, movement))

            step = step.replace(
                o1=self.worker_o1, o2=step.o2.replace(desired_goal=direction))
            self.trainers.worker.buffer.append(step)
            self.episode_count.update(Counter(worker_reward=step.r))

        self.time_steps += 1

    def reset(self):
        self.time_steps = 0
        return super().reset()


def boss_oracle(env: HierarchicalWrapper):
    def alignment(a):
        goal_dir = env._boss_action_to_goal_space(a)
        norm = np.linalg.norm(goal_dir)
        if not np.allclose(norm, 0):
            goal_dir /= norm
        return np.dot(goal_dir, direction)

    direction = env._desired_goal() - env._achieved_goal()
    if isinstance(env.action_space.boss, Box):
        return direction / max(np.linalg.norm(direction), 1e-6)
    actions = list(range(env.action_space.boss.n))
    # alignments = list(map(alignment, actions))
    return env._boss_action_to_goal_space(max(actions, key=alignment))


def worker_oracle(env: FrozenLakeHierarchicalWrapper, relative_goal: np.ndarray):
    fl = env.frozen_lake_env
    s = fl.from_s(fl.s)
    goal = s + relative_goal

    def in_bounds(new_s):
        return np.all(np.zeros(2) <= new_s) and np.all(
            new_s < np.array([fl.nrow, fl.ncol]))

    def distance_from_goal(i):
        new_s = s + DIRECTIONS[i]
        if in_bounds(new_s) and fl.desc[tuple(new_s)] == b'H':
            return np.inf
        return np.linalg.norm(new_s - goal)

    actions = list(range(env.action_space.worker.n))
    distances = list(map(distance_from_goal, actions))
    action = np.zeros(env.action_space.worker.n)
    best_distances = [i for i in actions if distances[i] == min(distances)]
    i = np.random.choice(best_distances)
    action[i] = 1
    return action


DIRECTIONS = np.array([
    [0, 0],  # stay
    [0, -1],  # left
    [1, 0],  # down
    [0, 1],  # right
    [-1, 0],  # up
])
BossState = namedtuple('BossState', 'goal action o0 v0')
