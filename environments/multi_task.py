import itertools
from collections import namedtuple

import numpy as np
from gym import spaces
from mujoco import ObjType

from environments.mujoco import distance_between
from environments.pick_and_place import PickAndPlaceEnv

Observation = namedtuple('Obs', 'observation goal')


class MultiTaskEnv(PickAndPlaceEnv):
    def __init__(self, geofence: float, randomize_pose=False, fixed_block=None, fixed_goal=None, **kwargs):
        self.fixed_block = fixed_block
        self.fixed_goal = fixed_goal
        self.randomize_pose = randomize_pose
        self.geofence = geofence
        self.goal_space = spaces.Box(
            low=np.array([-.11, -.19, .40]), high=np.array([.09, .2, .4001]))
        self.goal = self.goal_space.sample() if fixed_goal is None else fixed_goal
        super().__init__(fixed_block=False, **kwargs)
        # low=np.array([-.14, -.22, .40]), high=np.array([.11, .22, .63]))
        # goal_size = np.array([.0317, .0635, .0234]) * geofence
        intervals = [2, 3, 1]
        x, y, z = [
            np.linspace(l, h, n)
            for l, h, n in zip(self.goal_space.low, self.goal_space.high, intervals)
        ]
        goal_corners = np.array(list(itertools.product(x, y, z)))
        self.labels = {tuple(g): '.' for g in goal_corners}

    def _is_successful(self):
        return distance_between(self.goal, self.block_pos()) < self.geofence

    def _get_obs(self):
        return Observation(observation=super()._get_obs(), goal=self.goal)

    def _reset_qpos(self):
        if self.randomize_pose:
            for joint in [
                'slide_x', 'slide_y', 'arm_lift_joint', 'arm_flex_joint',
                'wrist_roll_joint', 'hand_l_proximal_joint'
            ]:
                qpos_idx = self.sim.get_jnt_qposadr(joint)
                jnt_range_idx = self.sim.name2id(ObjType.JOINT, joint)
                self.init_qpos[qpos_idx] = np.random.uniform(
                    *self.sim.jnt_range[jnt_range_idx])

        r = self.sim.get_jnt_qposadr('hand_r_proximal_joint')
        l = self.sim.get_jnt_qposadr('hand_l_proximal_joint')
        self.init_qpos[r] = self.init_qpos[l]

        block_joint = self.sim.get_jnt_qposadr('block1joint')
        if self.fixed_block is None:
            self.init_qpos[[
                block_joint + 0, block_joint + 1, block_joint + 3, block_joint + 6
            ]] = np.random.uniform(
                low=list(self.goal_space.low)[:2] + [0, -1],
                high=list(self.goal_space.high)[:2] + [1, 1])
        else:
            self.init_qpos[[
                block_joint + 0, block_joint + 1, block_joint + 2
            ]] = self.fixed_block
        return self.init_qpos

    def reset(self):
        if self.fixed_goal is None:
            self.goal = self.goal_space.sample()
        return super().reset()

    def render(self, labels=None, **kwargs):
        if labels is None:
            labels = dict()
        for label in self.labels:
            labels[tuple(label)] = 'x'
        labels[tuple(self.goal)] = 'g'
        return super().render(labels=labels, **kwargs)
