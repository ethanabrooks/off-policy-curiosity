# stdlib
from abc import abstractmethod
from collections import namedtuple
import enum

# third party
import numpy as np
import tensorflow as tf

# first party
from sac.utils import ArrayLike, Step, make_network

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

        self.q_network = make_network(a_size + o_size, 1, **network_args)
        self.v1_network = make_network(o_size, 1, **network_args)
        self.v2_network = make_network(o_size, 1, **network_args)

        self.v2_network.set_weights(self.v1_network.get_weights())
        self.global_step = tf.Variable(0, name='global_step')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        if embed_args:
            self.embed_optimizer = tf.train.AdamOptimizer(
                learning_rate=embed_args.pop('learning_rate', learning_rate))

    @tf.contrib.eager.defun
    def _get_actions(self, o1: tf.Tensor, sample: bool = True) -> np.array:
        parameters = self.get_policy_params(o1)
        self.A_sampled1 = self.policy_parameters_to_sample(parameters)
        self.A_max_likelihood = self.policy_parameters_to_max_likelihood_action(
            parameters)
        if sample:
            return self.A_sampled1
        else:
            return self.A_max_likelihood

    @tf.contrib.eager.defun
    def _train_step(self, step: Step):
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

        with tf.GradientTape(persistent=True) as tape:
            parameters = self.get_policy_params(step.o1)
            A_sampled1 = self.policy_parameters_to_sample(parameters)
            A_sampled2 = self.policy_parameters_to_sample(parameters)

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

        pi_norm = update(self.pi_network, pi_loss, tape)
        V_norm = update(self.v1_network, V_loss, tape)
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

    def getV1(self, o):
        return tf.reshape(self.v1_network(o), [-1])

    def getV2(self, o):
        return tf.reshape(self.v2_network(o), [-1])

    def getQ(self, o, a):
        return tf.reshape(
            self.q_network(tf.concat([o, self.preprocess_action(a)], axis=1)), [-1])

    def get_actions(self, o: ArrayLike, sample: bool = True) -> np.array:
        o = tf.convert_to_tensor(o.reshape(1, -1), dtype=tf.float32)
        return self._get_actions(o).numpy().reshape(-1)

    def train_step(self, step: Step) -> dict:
        step = Step(*[tf.convert_to_tensor(x, dtype=tf.float32) for x in step])
        return {k: v.numpy() for k, v in self._train_step(step).items()}

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
