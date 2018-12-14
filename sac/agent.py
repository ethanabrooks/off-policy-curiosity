# stdlib
import enum
from abc import abstractmethod
from collections import namedtuple
from typing import Callable, Collection

# third party
import numpy as np
import tensorflow as tf

# first party
from sac.utils import ArrayLike, Step

NetworkOutput = namedtuple('NetworkOutput', 'output state')


# noinspection PyArgumentList
class ModelType(enum.Enum):
    posterior = 'posterior'
    prior = 'prior'
    none = 'none'

    def __str__(self):
        return self.value


def make_network(input_size: int, sizes: Collection[int], activation, use_bias=True) -> \
        tf.keras.Sequential:
    assert len(sizes) >= 1
    return tf.keras.Sequential([
        tf.layers.Dense(
            input_shape=(in_size,),
            units=out_size, activation=activation, use_bias=use_bias)
        for in_size, out_size in zip([input_size] + sizes, sizes)
    ])


class AbstractAgent:
    def __init__(
            self,
            a_size: int,
            o_size: int,
            reward_scale: float,
            entropy_scale: float,
            activation: Callable,
            n_layers: int,
            layer_size: int,
            learning_rate: float,
            grad_clip: float,
            embed_args: dict = None,
    ) -> None:

        self.embed_args = embed_args
        self.embed = bool(embed_args)
        self.grad_clip = grad_clip
        self.learning_rate = learning_rate
        self.entropy_scale = entropy_scale
        self.a_size = a_size
        self.reward_scale = reward_scale
        self.activation = activation

        def _make_network(input_size, last_layer) -> tf.keras.Sequential:
            return make_network(input_size=input_size,
                                sizes=[layer_size] * n_layers + [last_layer],
                                activation=activation)

        self.pi_network = _make_network(o_size, a_size)
        self.q_network = _make_network(a_size + o_size, 1)
        self.v1_network = _make_network(o_size, 1)
        self.v2_network = _make_network(o_size, 1)

        self.v2_network.set_weights(self.v1_network.get_weights())
        self.global_step = tf.Variable(0, name='global_step')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        if embed_args:
            self.embed_optimizer = tf.train.AdamOptimizer(
                learning_rate=embed_args.pop('learning_rate', learning_rate))

    def get_parameters(self, o1):
        processed_s = self.pi_network(o1)
        return self.produce_policy_parameters(self.a_size, processed_s)

    def train_step(self, step: Step) -> dict:
        step = Step(*[tf.convert_to_tensor(x, dtype=tf.float32) for x in step])
        gamma = tf.constant(0.99)
        tau = 0.01

        def update(network: tf.keras.Model, loss: tf.Tensor, tape: tf.GradientTape):
            variables = network.trainable_variables
            gradients = tape.gradient(loss, variables)
            if self.grad_clip:
                gradients, norm = tf.clip_by_global_norm(gradients, self.grad_clip)
            else:
                norm = tf.global_norm(gradients)

            self.optimizer.apply_gradients(zip(gradients, variables))
            return norm

        with tf.GradientTape() as tape:
            parameters = self.get_parameters(step.o1)
            A_sampled1 = self.policy_parameters_to_sample(parameters)
            A_sampled2 = self.policy_parameters_to_sample(parameters)

            # constructing pi loss
            q2 = self.q_network(
                tf.concat([step.o1, self.preprocess_action(A_sampled2)], axis=1))
            v1 = self.v1_network(step.o1)
            log_pi_sampled2 = self.policy_parameters_to_log_prob(A_sampled2, parameters)
            log_pi_sampled2 *= self.entropy_scale  # type: tf.Tensor
            pi_loss = tf.reduce_mean(
                log_pi_sampled2 * tf.stop_gradient(log_pi_sampled2 - q2 + v1))

        pi_norm = update(self.pi_network, pi_loss, tape)

        with tf.GradientTape() as tape:
            # constructing V loss
            v1 = self.v1_network(step.o1)
            action = self.preprocess_action(A_sampled1)
            q1 = self.q_network(tf.concat([step.o1, action], axis=1))
            log_pi_sampled1 = self.policy_parameters_to_log_prob(A_sampled1, parameters)
            log_pi_sampled1 *= self.entropy_scale  # type: tf.Tensor
            V_loss = tf.reduce_mean(0.5 * tf.square(v1 - (q1 - log_pi_sampled1)))

        V_norm = update(self.v1_network, V_loss, tape)

        with tf.GradientTape() as tape:
            # constructing Q loss
            v2 = self.v2_network(step.o2)
            q1 = self.q_network(tf.concat([step.o1, self.preprocess_action(step.a)],
                                          axis=1))
            not_done = 1 - step.t  # type: tf.Tensor
            td_error = step.r + gamma * not_done * v2 - q1
            Q_loss = tf.reduce_mean(0.5 * td_error ** 2)

        Q_norm = update(self.q_network, Q_loss, tape)

        for var1, var2 in zip(self.v1_network.variables, self.v2_network.variables):
            tf.assign(var2, tau * var1 + (1 - tau) * var2)

        # self.check = tf.add_check_numerics_ops()
        entropy = tf.reduce_mean(self.entropy_from_params(parameters))

        return dict(
            entropy=entropy,
            V_loss=V_loss,
            Q_loss=Q_loss,
            pi_loss=pi_loss,
            V_grad=V_norm,
            Q_grad=Q_norm,
            pi_grad=pi_norm,
        )

    def get_actions(self, o: ArrayLike, sample: bool = True, state=None) -> np.array:
        parameters = self.get_parameters(tf.convert_to_tensor(o.reshape(1, -1),
                                                              dtype=tf.float32))
        if sample:
            func = self.policy_parameters_to_sample
        else:
            func = self.policy_parameters_to_max_likelihood_action
        return func(parameters).numpy().reshape(-1)

    @abstractmethod
    def network(self, inputs: tf.Tensor) -> NetworkOutput:
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
    def preprocess_action(self, action_sample: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def entropy_from_params(self, params: tf.Tensor) -> tf.Tensor:
        pass
