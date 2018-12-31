# stdlib
from collections import Counter, namedtuple

# third party
# first party
from gym.spaces import Box
import numpy as np

from hsr.hsr import HSREnv, distance_between
from sac.replay_buffer import ReplayBuffer
from sac.train import Trainer
from sac.old_utils import Step, unwrap_env

Samples = namedtuple('Samples', 'train valid test')

REWARD_DELTA_KWD = 'return_delta'


class UnsupervisedTrainer(Trainer):
    def __init__(self, episodes_per_goal: int, **kwargs):
        super().__init__(**kwargs)
        self.episodes_per_goal = episodes_per_goal
        self.hsr_env = unwrap_env(self.env, lambda e: isinstance(e, HSREnv))
        self.return_history = []
        self.test_sample = None
        if episodes_per_goal == 1:
            self.lin_regress_op = None
        else:
            x = np.stack(
                [np.ones(episodes_per_goal),
                 np.arange(episodes_per_goal)], axis=1)
            self.lin_regress_op = np.linalg.inv(x.T @ x) @ x.T
        self.td_errors = None
        self.initial_obs = None
        self.initial_achieved = None
        self.prev_obs = [
            np.random.uniform(-1, 1, space.low.shape)
            for space in self.env.observation_space.spaces
        ]
        self.prev_obs = None
        self.prev_goal = None

        self.double_goal_space = Box(
            low=self.hsr_env.goal_space.low * 1.1,
            high=self.hsr_env.goal_space.high * 1.1,
        )

    def perform_update(self):
        counter = Counter()

        if self.buffer_ready():
            if self.test_sample is None:
                self.test_sample = ReplayBuffer(maxlen=self.batch_size)
                self.test_sample.extend(self.sample_buffer())
            agent = self.agents.act

            for i in range(self.n_train_steps):
                # get samples
                train_sample = self.sample_buffer()
                valid_sample = self.sample_buffer()
                self.test_sample.append(self.sample_buffer(batch_size=1))

                def td_errors():
                    return Samples(*map(
                        agent.td_error,
                        Samples(
                            train=train_sample,
                            valid=valid_sample,
                            test=Step(*self.test_sample.buffer.values))))

                pre = td_errors()
                train_result = self.train_step(sample=train_sample)
                post = td_errors()

                def get_delta(pre, post):
                    return np.mean(pre - post)

                delta = Samples(*[get_delta(*args) for args in zip(pre, post)])

                # noinspection PyTypeChecker
                counter.update(
                    pre_train_td_error=np.mean(pre.train),
                    pre_test_td_error=np.mean(pre.test),
                    pre_valid_td_error=np.mean(pre.valid),
                    post_train_td_error=np.mean(post.train),
                    post_test_td_error=np.mean(post.test),
                    post_valid_td_error=np.mean(post.valid),
                    train_td_error_delta=delta.train,
                    test_td_error_delta=delta.test,
                    valid_td_error_delta=delta.valid,
                )

                if self.td_errors is not None:
                    episodic = Samples(
                        *[get_delta(*args) for args in zip(self.td_errors, post)])

                    counter.update(
                        episodic_train_td_error_delta=episodic.train,
                        episodic_test_td_error_delta=episodic.test,
                        episodic_valid_td_error_delta=episodic.valid,
                    )
                self.td_errors = post

                for k, v in train_result.items():
                    if np.isscalar(v):
                        counter.update(**{k: v})

        return counter

    def trajectory(self, time_steps: int, final_index=None):
        raise NotImplementedError

    def reinforce(self, o1, goal_reward: float):
        agent = self.agents.act
        fetches = dict(goal=agent.new_goal)
        feed_dict = {agent.new_initial_obs: self.preprocess_func(o1)}
        additional_results = dict()
        if self.prev_obs is not None:
            fetches.update(
                goal_loss=agent.goal_loss,
                op=agent.train_goal,
            )
            feed_dict.update({
                agent.old_goal: self.prev_goal,
                agent.old_initial_obs: self.preprocess_func(self.prev_obs),
                agent.goal_reward: goal_reward,
            })
            additional_results = dict(goal_reward=goal_reward)

        train_result = {
            **self.sess.run(fetches=fetches, feed_dict=feed_dict),
            **additional_results
        }
        self.prev_goal = train_result['goal']
        self.prev_obs = o1
        return train_result

    def run_episode(self, o1, eval_period, render):
        episode_count = dict()
        achieved_goal = self.hsr_env.achieved_goal()

        if self.initial_obs is None:
            self.initial_obs = o1
        if self.initial_achieved is None:
            self.initial_achieved = achieved_goal

        if eval_period:
            self.hsr_env.set_goal(self.hsr_env.goal_space.sample())

        elif len(self.return_history) == self.episodes_per_goal:
            _, return_delta = self.lin_regress_op @ np.array(self.return_history)

            reinforce_result = self.reinforce(
                self.initial_obs, goal_reward=np.abs(return_delta))
            episode_count.update(reinforce_result)
            goal = reinforce_result['goal']
            print('goal', goal)
            self.hsr_env.set_goal(goal)

            # reset values
            self.return_history = []
            self.initial_obs = o1
            self.initial_achieved = achieved_goal

            episode_count.update(return_delta=return_delta)
            # **train_result)

        goal_distance = distance_between(achieved_goal, self.hsr_env.goal)
        print(f'Goal distance: {goal_distance}')
        episode_count.update(
            goal_distance=goal_distance,
            **super().run_episode(o1=o1, eval_period=eval_period, render=render))

        if not eval_period:
            self.return_history.append(episode_count['reward'])

        return episode_count


def sigmoid(x):
    return (np.tanh(x * 2 - 1) + 1) / 2


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
