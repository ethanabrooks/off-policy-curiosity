# stdlib
import sys
import time
from typing import Iterable, Tuple, Union

# third party
from gym import utils
import gym.envs.toy_text.frozen_lake
from gym.spaces import Discrete
import numpy as np
from six import StringIO

# first party
from hsr.hsr import Observation

MAPS = gym.envs.toy_text.frozen_lake.MAPS
MAPS["2x2"] = ["FF"] * 2
MAPS["3x3"] = [
    "SFF",
    "FHF",
    "FFG",
]

MAPS["3x4"] = ["SFFF", "FHFH", "HFFG"]
MAPS["5x5"] = ['FFFFH', 'HHFFF', 'HFFFF', 'FHFFF', 'FFFFF']
MAPS["10x10"] = [
    'HFFFFFFFFF', 'HFFFFFFFFH', 'FFFFFFHFHF', 'FFFFFFFHFF', 'FFFFFFFFHF', 'FFFHFFHFHF',
    'FFFFFFFFFF', 'FFFFFFFFFF', 'FHFFFFFFFF', 'FFFFHFFFFF'
]


class FrozenLakeEnv(gym.envs.toy_text.frozen_lake.FrozenLakeEnv):
    def __init__(self,
                 map_dims: Tuple[int, int] = (4, 4),
                 default_reward: float = 0.,
                 is_slippery: bool = False,
                 random_start: bool = False,
                 random_goal: bool = False,
                 random_map=False):

        self.default_reward = default_reward or 0
        self.is_slippery = is_slippery
        self.random_start = random_start
        self.random_goal = random_goal
        self.start = (0, 0)
        h, w = map_dims

        if random_map:
            while True:
                desc = np.random.choice([b'F', b'H'], [h, w], p=[.85, .15])
                if self.random_walk(self.start, 1) != self.start:
                    break  # start is not surrounded by holes
            self.original_desc = np.copy(desc)
        else:
            desc = np.array([list(r) for r in MAPS[f"{h}x{w}"]])
            self.original_desc = np.copy(desc)
            self.original_desc[0, 0] = b'F'
            self.original_desc[-1, -1] = b'F'

        self.goal = tuple(desc.shape - np.ones(2, dtype=int))
        desc[self.start] = b'S'
        desc[self.goal] = b'G'
        self.reverse = {
            1: 3,  # left -> right
            3: 1,  # right -> left
            2: 4,  # down -> up
            4: 2  # up -> down
        }
        super().__init__(desc=desc, is_slippery=is_slippery)
        for i in range(self.nrow):
            for j in range(self.ncol):
                self.set_transitions((i, j))

        self.observation_space = Discrete(4)
        self.action_space = Discrete(5)

    def inc(self, row, col, a):
        if a == 1:  # left
            col = max(col - 1, 0)
        elif a == 2:  # down
            row = min(row + 1, self.nrow - 1)
        elif a == 3:  # right
            col = min(col + 1, self.ncol - 1)
        elif a == 4:  # up
            row = max(row - 1, 0)
        return row, col

    def random_walk(self, pos: tuple, n_steps: int):
        explored = []
        while n_steps > 0:
            explored.append(pos)
            next_positions = [
                self.inc(*pos, d) for d in range(5) if not self.inc(*pos, d) in explored
                and not self.desc[self.inc(*pos, d)] == b'H'
            ]
            if not next_positions:
                return pos
            pos = next_positions[np.random.randint(len(next_positions))]
            n_steps -= 1
        return pos

    def set_transition(self, pos: Union[tuple, np.ndarray], actions: Iterable = None):
        pos = tuple(pos)
        s = self.to_s(*pos)
        actions = actions or range(5)  # type: Iterable
        for a in actions:
            letter = self.desc[pos]
            if letter in b'GH':
                self.P[s][a] = [(1.0, s, 0, True)]
            else:
                if self.is_slippery:
                    for b in [(a - 1) % 5, a, (a + 1) % 5]:
                        newrow, newcol = self.inc(*pos, b)
                        newstate = self.to_s(newrow, newcol)
                        newletter = self.desc[newrow, newcol]
                        done = bytes(newletter) in b'GH'
                        rew = float(newletter == b'G')
                        self.P[s][a] = [(1.0 / 3.0, newstate, rew, done)]
                else:
                    newrow, newcol = self.inc(*pos, a)
                    newstate = self.to_s(newrow, newcol)
                    newletter = self.desc[newrow, newcol]
                    done = bytes(newletter) in b'GH'
                    rew = float(newletter == b'G')
                    self.P[s][a] = [(1.0, newstate, rew, done)]

    def set_transitions(self, pos: Union[tuple, np.ndarray]):
        self.set_transition(pos)
        for a in range(5):
            adjacent = self.inc(*pos, a)
            if adjacent != pos:
                self.set_transition(adjacent, actions=[self.reverse[a]])
                assert self.inc(*adjacent, self.reverse[a]) == pos

    def mutate_desc(self, old_pos, new_pos):
        letter = self.desc[old_pos]
        self.desc[old_pos] = self.original_desc[old_pos]
        self.desc[new_pos] = letter

    def goal_vector(self):
        return np.array(self.goal)

    def reset(self):
        time.sleep(1)
        if self.random_start:
            assert self.desc[self.start] == b'S'
            old_start = self.start
            while True:
                new_start = np.random.randint(self.nrow), np.random.randint(self.ncol)
                if self.desc[new_start] not in b'GH':
                    break
            assert self.desc[new_start] not in b'GH'
            self.mutate_desc(old_start, new_start)
            assert self.desc[new_start] == b'S'
            if new_start != old_start:
                assert self.desc[old_start] != b'S'
            self.isd[self.to_s(*old_start)] = 0
            self.isd[self.to_s(*new_start)] = 1
            self.start = new_start

        if self.random_goal:
            old_goal = self.goal
            while True:
                new_goal = np.random.randint(self.nrow), np.random.randint(self.ncol)
                if self.desc[new_goal] not in b'SH':
                    break

            self.mutate_desc(old_goal, new_goal)
            self.set_transitions(new_goal)
            self.set_transitions(old_goal)
            assert self.desc[new_goal] == b'G'
            if old_goal != new_goal:
                assert self.desc[old_goal] != b'G'
            for d in range(5):
                pos = self.inc(*new_goal, d)
                if pos != new_goal:
                    transition = self.P[self.to_s(*pos)][self.reverse[d]][0]
                    if self.desc[pos] in b'SF':
                        # going backward on ice
                        assert transition[1:] == (self.to_s(*new_goal), 1, True)
                    elif self.desc[pos] == b'H':
                        # going backward in a hole
                        assert transition[1:] == (self.to_s(*pos), 0, True)

            for transition in self.P[self.to_s(*new_goal)].values():
                # transitions from goal
                assert transition[0][1:] == (self.to_s(*new_goal), 0, True)

            self.goal = new_goal

        return Observation(
            observation=self.preprocess(super().reset()), goal=self.goal_vector())

    def to_s(self, row, col):
        return row * self.ncol + col

    def from_s(self, s):
        return np.array([s // self.ncol, s % self.ncol])

    def step(self, a):
        s, r, t, i = super().step(a)
        s = self.preprocess(s)
        s = Observation(observation=s, goal=self.goal_vector())
        if r == 0:
            r = self.default_reward
        i['log count'] = {'successes': float(r > 0)}
        return s, r, t, i

    # def step(self, s: int):
    #     self.s = s
    #     s2d = self.from_s(s)
    #     self.lastaction = s2d
    #     newletter = self.desc[tuple(s2d)]
    #     o = Observation(observation=s2d, goal=self.goal_vector())
    #     r = float(newletter == b'G')
    #     t = bytes(newletter) in b'GH'
    #     i = {}
    #     if r == 0:
    #         r = self.default_reward
    #     i['log count'] = {'successes': float(r > 0)}
    #     return o, r, t, i

    def preprocess(self, s):
        return s // self.nrow, s % self.ncol

    def render(self, mode='human'):
        time.sleep(.5)
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Stay", "Left", "Down", "Right",
                                             "Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc) + "\n")

        if mode != 'human':
            return outfile
