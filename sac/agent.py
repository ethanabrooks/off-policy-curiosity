# stdlib
from abc import abstractmethod
from collections import namedtuple
import enum
from typing import List

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

        self.O1 = tf.placeholder(tf.float32, [None, o_size], name='O1')
        self.O2 = tf.placeholder(tf.float32, [None, o_size], name='O2')
        self.A = A = tf.placeholder(tf.float32, [None, a_size], name='A')
        self.R = R = tf.placeholder(tf.float32, [None], name='R')
        self.T = T = tf.placeholder(tf.float32, [None], name='T')
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

        with tf.variable_scope('agent'):
            processed_s = self.pi_network(self.O1)
            parameters = self.parameters = self.produce_policy_parameters(a_size, processed_s)

            def pi_network_log_prob(a: tf.Tensor, name: str, _reuse: bool) \
                    -> tf.Tensor:
                with tf.variable_scope(name, reuse=_reuse):
                    return self.policy_parameters_to_log_prob(a, parameters)

            def sample_pi_network(name: str, _reuse: bool) -> tf.Tensor:
                with tf.variable_scope(name, reuse=_reuse):
                    return self.policy_parameters_to_sample(parameters)

            # generate actions:
            self.A_max_likelihood = tf.stop_gradient(
                self.policy_parameters_to_max_likelihood_action(parameters))
            self.A_sampled1 = A_sampled1 = tf.stop_gradient(
                sample_pi_network('pi', _reuse=True))

            # constructing V loss
            v1 = tf.reshape(self.v1_network(self.O1), [-1])
            self.v1 = v1
            q1 = tf.reshape(
                self.q_network(
                    tf.concat([self.O1, self.transform_action_sample(A_sampled1)], axis=1)),
                [-1])
            log_pi_sampled1 = pi_network_log_prob(A_sampled1, 'pi', _reuse=True)
            log_pi_sampled1 *= entropy_scale  # type: tf.Tensor
            self.V_loss = V_loss = tf.reduce_mean(
                0.5 * tf.square(v1 - (q1 - log_pi_sampled1)))

            # constructing Q loss
            self.v2 = v2 = tf.reshape(self.v2_network(self.O2), [-1])
            self.q1 = q = tf.reshape(
                self.q_network(tf.concat([self.O1, self.transform_action_sample(A)], axis=1)),
                [-1])
            not_done = 1 - T  # type: tf.Tensor
            self.q_target = q_target = R + gamma * not_done * v2
            self.Q_error = tf.square(q - q_target)
            self.Q_loss = Q_loss = tf.reduce_mean(0.5 * self.Q_error)

            # constructing pi loss
            self.A_sampled2 = A_sampled2 = tf.stop_gradient(
                sample_pi_network('pi', _reuse=True))
            q2 = tf.reshape(
                self.q_network(
                    tf.concat([self.O1, self.transform_action_sample(A_sampled2)], axis=1)),
                [-1])
            log_pi_sampled2 = pi_network_log_prob(A_sampled2, 'pi', _reuse=True)
            log_pi_sampled2 *= entropy_scale  # type: tf.Tensor
            self.pi_loss = pi_loss = tf.reduce_mean(
                log_pi_sampled2 * tf.stop_gradient(log_pi_sampled2 - q2 + v1))

            self.train_V, self.V_grad = update(network=self.v1_network, loss=V_loss)
            self.train_Q, self.Q_grad = update(network=self.q_network, loss=Q_loss)
            self.train_pi, self.pi_grad = update(network=self.pi_network, loss=pi_loss)

            # placeholders
            soft_update_xi_bar_ops = [
                tf.assign(xbar, tau * x + (1 - tau) * xbar) for (xbar, x)
                in zip(self.v2_network.trainable_variables, self.v1_network.trainable_variables)
            ]
            self.soft_update_xi_bar = tf.group(*soft_update_xi_bar_ops)
            # self.check = tf.add_check_numerics_ops()
            self.entropy = tf.reduce_mean(self.entropy_from_params(self.parameters))
            # ensure that xi and xi_bar are the same at initialization

            sess.run(tf.global_variables_initializer())

    def train_step(self, step: Step) -> dict:
        feed_dict = {
            self.O1: step.o1,
            self.A: step.a,
            self.R: np.array(step.r) * self.reward_scale,
            self.O2: step.o2,
            self.T: step.t,
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

    def get_actions(self, o: ArrayLike, sample: bool = True, state=None) -> NetworkOutput:
        A = self.A_sampled1 if sample else self.A_max_likelihood
        return NetworkOutput(output=self.sess.run(A, {self.O1: [o]})[0], state=0)

    def get_v1(self, o1: np.ndarray):
        return self.sess.run(self.v1, feed_dict={self.O1: [o1]})[0]

    def get_value(self, step: Step):
        return self.sess.run(
            self.v1, feed_dict={
                self.O1: step.o1,
            })

    def td_error(self, step: Step):
        return self.sess.run(
            self.Q_error,
            feed_dict={
                self.O1: step.o1,
                self.A: step.a,
                self.R: step.r,
                self.O2: step.o2,
                self.T: step.t
            })

    def network(self, inputs: tf.Tensor):
        return mlp(inputs=inputs, **self.network_args)

    def _print(self, tensor, name: str):
        return tf.Print(tensor, [tensor], message=name, summarize=1e5)

    @abstractmethod
    def pi_network(self, inputs: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def produce_policy_parameters(self, a_shape: int,
                                  processed_o: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def policy_parameters_to_log_prob(self, a: tf.Tensor,
                                      parameters: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def policy_parameters_to_sample(self, parameters: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def policy_parameters_to_max_likelihood_action(self, parameters) -> tf.Tensor:
        pass

    @abstractmethod
    def transform_action_sample(self, action_sample: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def entropy_from_params(self, params: tf.Tensor) -> tf.Tensor:
        pass
