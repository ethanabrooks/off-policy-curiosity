from abc import abstractmethod
from collections import namedtuple

import gym
import numpy as np
from gym.spaces import Box

from environments.base import at_goal, distance_between
from environments.pick_and_place import Goal, PickAndPlaceEnv
from sac.utils import Step

State = namedtuple('State', 'obs goal')


class HindsightWrapper(gym.Wrapper):
    def __init__(self, env, default_reward=0):
        self._default_reward = default_reward
        super().__init__(env)
        vector_state = self.vectorize_state(self.reset())
        self.observation_space = Box(-1, 1, vector_state.shape)

    @abstractmethod
    def achieved_goal(self, obs):
        raise NotImplementedError

    @abstractmethod
    def at_goal(self, obs, goal):
        raise NotImplementedError

    @abstractmethod
    def desired_goal(self):
        raise NotImplementedError

    @staticmethod
    def vectorize_state(state):
        return np.concatenate(state)

    def step(self, action):
        s2, r, t, info = self.env.step(action)
        new_s2 = State(obs=s2, goal=self.desired_goal())
        return new_s2, r, t, info

    def reset(self):
        return State(obs=self.env.reset(), goal=self.desired_goal())

    def recompute_trajectory(self, trajectory, final_state=-1):
        if not trajectory:
            return ()
        achieved_goal = self.achieved_goal(trajectory[final_state].s2.obs)
        for step in trajectory[:final_state]:
            new_t = self.at_goal(step.s2.obs, achieved_goal) or step.t
            r = float(new_t)
            s = Step(s1=State(obs=step.s1.obs, goal=achieved_goal), a=step.a, r=r,
                     s2=State(obs=step.s2.obs, goal=achieved_goal), t=new_t)
            print('reward', r)
            yield s
            if new_t:
                break


class MountaincarHindsightWrapper(HindsightWrapper):
    """
    new obs is [pos, vel, goal_pos]
    """

    def achieved_goal(self, obs):
        return np.array([obs[0]])

    def at_goal(self, obs, goal):
        return obs[0] >= goal[0]

    def desired_goal(self):
        return np.array([0.45])


class PickAndPlaceHindsightWrapper(HindsightWrapper):
    def __init__(self, env, default_reward):
        if isinstance(env, gym.Wrapper):
            assert isinstance(env.unwrapped, PickAndPlaceEnv)
            self.unwrapped_env = env.unwrapped
        else:
            assert isinstance(env, PickAndPlaceEnv)
            self.unwrapped_env = env
        super().__init__(env, default_reward)

    def achieved_goal(self, obs):
        return Goal(
            gripper=self.unwrapped_env.gripper_pos(obs),
            block=self.unwrapped_env.block_pos(obs))

    def at_goal(self, obs, goal):
        gripper_at_goal = at_goal(self.unwrapped_env.gripper_pos(obs), goal.gripper,
                                  self.unwrapped_env.geofence)
        block_at_goal = at_goal(self.unwrapped_env.block_pos(obs), goal.block,
                                self.unwrapped_env.geofence)
        return gripper_at_goal and block_at_goal

    def desired_goal(self):
        return self.unwrapped_env.goal()

    @staticmethod
    def vectorize_state(state):
        state = State(*state)
        return np.concatenate([state.obs, (np.concatenate(state.goal))])
