from abc import abstractmethod
from collections import namedtuple
from copy import deepcopy
from typing import Optional

import gym
import numpy as np
from gym.spaces import Box

from environments.mujoco import distance_between
from environments.multi_task import MultiTaskEnv
from environments.pick_and_place import PickAndPlaceEnv
from sac.array_group import ArrayGroup
from sac.utils import Step, unwrap_env, vectorize

Goal = namedtuple('Goal', 'gripper block')


class Observation(namedtuple('Obs', 'observation achieved_goal desired_goal')):
    def replace(self, **kwargs):
        return super()._replace(**kwargs)


class HindsightWrapper(gym.Wrapper):
    def __init__(self, env):
        self.state_vectors = {}
        super().__init__(env)
        s = self.reset()
        vector_state = self.vectorize_state(s)
        self.observation_space = Box(-1, 1, vector_state.shape)

    @abstractmethod
    def _achieved_goal(self):
        raise NotImplementedError

    @abstractmethod
    def _is_success(self, achieved_goal, desired_goal):
        raise NotImplementedError

    @abstractmethod
    def _desired_goal(self):
        raise NotImplementedError

    def step(self, action):
        o2, r, t, info = self.env.step(action)
        new_o2 = Observation(
            observation=o2,
            desired_goal=self._desired_goal(),
            achieved_goal=self._achieved_goal())
        return new_o2, r, t, info

    def reset(self):
        return Observation(
            observation=self.env.reset(),
            desired_goal=self._desired_goal(),
            achieved_goal=self._achieved_goal())

    def recompute_trajectory(self, trajectory: Step):
        trajectory = deepcopy(trajectory)

        # get values
        o1 = Observation(*trajectory.o1)
        o2 = Observation(*trajectory.o2)
        achieved_goal = ArrayGroup(o2.achieved_goal)[-1]

        # perform assignment
        ArrayGroup(o1.desired_goal)[:] = achieved_goal
        ArrayGroup(o2.desired_goal)[:] = achieved_goal
        trajectory.r[:] = self._is_success(o2.achieved_goal, o2.desired_goal)
        trajectory.t[:] = np.logical_or(trajectory.t, trajectory.r)

        first_terminal = np.flatnonzero(trajectory.t)[0]
        return ArrayGroup(trajectory)[:first_terminal + 1]  # include first terminal

    @staticmethod
    def vectorize_state(obs, shape: Optional[tuple] = None):
        obs = Observation(*obs)
        obs = [obs.observation, obs.desired_goal]
        return vectorize(obs, shape=shape)


class MountaincarHindsightWrapper(HindsightWrapper):
    """
    new obs is [pos, vel, goal_pos]
    """

    def step(self, action):
        s2, r, t, info = super().step(action)
        return s2, max([0, r]), t, info

    def _achieved_goal(self):
        return self.env.unwrapped.state[0]

    def _is_success(self, achieved_goal, desired_goal):
        return self.env.unwrapped.state[0] >= self._desired_goal()

    def _desired_goal(self):
        return 0.45


class PickAndPlaceHindsightWrapper(HindsightWrapper):
    def _is_success(self, achieved_goal, desired_goal):
        achieved_goal = Goal(*achieved_goal)
        desired_goal = Goal(*desired_goal)

        geofence = self.env.unwrapped.geofence
        block_distance = distance_between(achieved_goal.block, desired_goal.block)
        goal_distance = distance_between(achieved_goal.gripper, desired_goal.gripper)
        return np.logical_and(block_distance < geofence, goal_distance < geofence)

    def _achieved_goal(self):
        return Goal(
            gripper=self.env.unwrapped.gripper_pos(),
            block=self.env.unwrapped.block_pos())

    def _desired_goal(self):
        return self.env.unwrapped.goal()

    @staticmethod
    def vectorize_state(state, shape=None):
        state = Observation(*state)
        return vectorize(
            [state.observation, state.desired_goal], shape=shape)
