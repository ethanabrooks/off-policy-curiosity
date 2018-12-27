# stdlib
from abc import abstractmethod
from collections import namedtuple
import enum

# third party
import numpy as np
import tensorflow as tf

# first party
from sac.utils import ArrayLike, Step, make_network, mlp

NetworkOutput = namedtuple('NetworkOutput', 'output state')


# noinspection PyArgumentList
class ModelType(enum.Enum):
    posterior = 'posterior'
    prior = 'prior'
    none = 'none'

    def __str__(self):
        return self.value


class AbstractAgent:
    def __init__(
            self,
            sess,
            a_size: int,
            o_size: int,
            reward_scale: float,
            entropy_scale: float,
            learning_rate: float,
            grad_clip: float,
            network_args: dict,
            embed_args: dict = None,
    ) -> None:
        self.o_size = o_size
        self.embed_args = embed_args
        self.embed = bool(embed_args)
        self.grad_clip = grad_clip
        self.learning_rate = learning_rate
        self.entropy_scale = entropy_scale
        self.a_size = a_size
        self.reward_scale = reward_scale
        self.sess = sess

        self.q_network = make_network(a_size + o_size, 1, **network_args)
        self.v1_network = make_network(o_size, 1, **network_args)
        self.v2_network = make_network(o_size, 1, **network_args)

        self.v2_network.set_weights(self.v1_network.get_weights())
        self.global_step = tf.Variable(0, name='global_step')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        if embed_args:
            self.embed_optimizer = tf.train.AdamOptimizer(
                learning_rate=embed_args.pop('learning_rate', learning_rate))

        self.o1 = tf.placeholder(tf.float32, [None, o_size], name='O1')
        self.o2 = tf.placeholder(tf.float32, [None, o_size], name='O2')
        self.a = tf.placeholder(tf.float32, [None, a_size], name='A')
        self.r = tf.placeholder(tf.float32, [None], name='R')
        self.t = tf.placeholder(tf.float32, [None], name='T')

        gamma = tf.constant(0.99)
        tau = 0.01

        def update(network: tf.keras.Model, loss: tf.Tensor):
            variables = network.trainable_variables
            gradients, variables = zip(*self.optimizer.compute_gradients(loss, variables))
            if self.grad_clip:
                gradients, norm = tf.clip_by_global_norm(gradients, self.grad_clip)
            else:
                norm = tf.global_norm(gradients)
            op = self.optimizer.apply_gradients(
                zip(gradients, variables), global_step=self.global_step)
            return op, norm

        step = self
        with tf.variable_scope('agent'):
            parameters = self.get_policy_params(step.o1)
            A_sampled1 = self.policy_parameters_to_sample(parameters)
            A_sampled2 = self.policy_parameters_to_sample(parameters)
            self.A_sampled1 = A_sampled1

            # generate actions:
            self.A_max_likelihood = tf.stop_gradient(
                self.policy_parameters_to_max_likelihood_action(parameters))

            # constructing V loss
            v1 = self.getV1(step.o1)
            self.v1 = v1
            q1 = self.getQ(step.o1, A_sampled1)
            log_pi_sampled1 = self.policy_parameters_to_log_prob(A_sampled1, parameters)
            log_pi_sampled1 *= self.entropy_scale  # type: tf.Tensor
            self.V_loss = V_loss = tf.reduce_mean(
                0.5 * tf.square(v1 - (q1 - log_pi_sampled1)))

            # constructing Q loss
            self.v2 = v2 = self.getV2(step.o2)
            self.q1 = q = self.getQ(step.o1, step.a)
            not_done = 1 - step.t  # type: tf.Tensor
            self.q_target = q_target = step.r + gamma * not_done * v2
            self.Q_error = tf.square(q - q_target)
            self.Q_loss = Q_loss = tf.reduce_mean(0.5 * self.Q_error)

            # constructing pi loss
            q2 = self.getQ(step.o1, A_sampled2)
            log_pi_sampled2 = self.policy_parameters_to_log_prob(A_sampled1, parameters)
            log_pi_sampled2 *= self.entropy_scale  # type: tf.Tensor
            self.pi_loss = pi_loss = tf.reduce_mean(
                log_pi_sampled2 * tf.stop_gradient(log_pi_sampled2 - q2 + v1))

            step.train_V, self.V_grad = update(network=self.v1_network, loss=V_loss)
            step.train_Q, self.Q_grad = update(network=self.q_network, loss=Q_loss)
            step.train_pi, self.pi_grad = update(network=self.pi_network, loss=pi_loss)

            # placeholders
            soft_update_xi_bar_ops = [
                tf.assign(xbar, tau * x + (1 - tau) * xbar)
                for (xbar, x) in zip(self.v2_network.trainable_variables,
                                     self.v1_network.trainable_variables)
            ]
            self.soft_update_xi_bar = tf.group(*soft_update_xi_bar_ops)
            # self.check = tf.add_check_numerics_ops()
            self.entropy = tf.reduce_mean(self.entropy_from_params(parameters))
            # ensure that xi and xi_bar are the same at initialization

            self.sess.run(tf.global_variables_initializer())

    def train_step(self, step: Step) -> dict:
        feed_dict = {
            self.o1: step.o1,
            self.a:  step.a,
            self.r:  np.array(step.r) * self.reward_scale,
            self.o2: step.o2,
            self.t:  step.t,
        }

        return self.sess.run(
            dict(
                entropy=self.entropy,
                soft_update_xi_bar=self.soft_update_xi_bar,
                Q_error=self.Q_error,
                V_loss=self.V_loss,
                Q_loss=self.Q_loss,
                pi_loss=self.pi_loss,
                V_grad=self.V_grad,
                Q_grad=self.Q_grad,
                pi_grad=self.pi_grad,
                train_V=self.train_V,
                train_Q=self.train_Q,
                train_pi=self.train_pi,
            ), feed_dict)

    def getV1(self, o):
        return tf.reshape(self.v1_network(o), [-1])

    def getV2(self, o):
        return tf.reshape(self.v2_network(o), [-1])

    def getQ(self, o, a):
        return tf.reshape(
            self.q_network(tf.concat([o, self.preprocess_action(a)], axis=1)), [-1])

    def get_actions(self, o: ArrayLike, sample: bool = True, state=None) -> NetworkOutput:
        A = self.A_sampled1 if sample else self.A_max_likelihood
        return NetworkOutput(output=self.sess.run(A, {self.o1: [o]})[0], state=0)

    def get_v1(self, o1: np.ndarray):
        return self.sess.run(self.v1, feed_dict={self.o1: [o1]})[0]

    def get_value(self, step: Step):
        return self.sess.run(
            self.v1, feed_dict={
                self.o1: step.o1,
            })

    def td_error(self, step: Step):
        return self.sess.run(
            self.Q_error,
            feed_dict={
                self.o1: step.o1,
                self.a:  step.a,
                self.r:  step.r,
                self.o2: step.o2,
                self.t:  step.t
            })

    @abstractmethod
    def pi_network(self, inputs: tf.Tensor) -> NetworkOutput:
        pass

    def get_policy_params(self, obs: tf.Tensor) -> tf.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def policy_parameters_to_log_prob(a: tf.Tensor, parameters: tf.Tensor) -> tf.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def policy_parameters_to_sample(parameters: tf.Tensor) -> tf.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def policy_parameters_to_max_likelihood_action(parameters) -> tf.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def preprocess_action(action_sample: tf.Tensor) -> tf.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def entropy_from_params(params: tf.Tensor) -> tf.Tensor:
        pass
