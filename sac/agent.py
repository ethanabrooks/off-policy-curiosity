# stdlib
from abc import abstractmethod
from collections import namedtuple
import enum
from typing import Callable, Iterable, List, Sequence

# third party
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.nn_impl import l2_normalize

# first party
from sac.utils import ArrayLike, Step, mlp

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
            sess: tf.Session,
            o_shape: Iterable,
            a_shape: Sequence,
            batch_size: int,
            reward_scale: float,
            entropy_scale: float,
            activation: Callable,
            n_layers: int,
            layer_size: int,
            learning_rate: float,
            grad_clip: float,
            device_num: int = 1,
            embed_args: dict = None,
            reuse: bool = False,
            scope: str = 'agent',
    ) -> None:

        self.default_train_values = [
            'entropy',
            'soft_update_xi_bar',
            'Q_error',
            'V_loss',
            'Q_loss',
            'pi_loss',
            'V_grad',
            'Q_grad',
            'pi_grad',
            'train_V',
            'train_Q',
            'train_pi',
        ]
        embed = bool(embed_args)
        if embed:
            self.default_train_values.extend([
                'copy_o1_embed',
                'embed_loss',
                'embed_grad',
                'embed_baseline',
                'train_embed',
                'regularization'
            ])
        self.reward_scale = reward_scale
        self.activation = activation
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.initial_state = None
        self.sess = sess

        with tf.device('/cpu:0'), tf.variable_scope(scope, reuse=reuse):
            self.global_step = tf.Variable(0, name='global_step')

            self.O1 = tf.placeholder(tf.float32, [None] + list(o_shape), name='O1')
            self.O2 = tf.placeholder(tf.float32, [None] + list(o_shape), name='O2')
            self.A = A = tf.placeholder(tf.float32, [None] + list(a_shape), name='A')
            self.R = R = tf.placeholder(tf.float32, [None], name='R')
            self.T = T = tf.placeholder(tf.float32, [None], name='T')
            gamma = tf.constant(0.99)
            tau = 0.01

            processed_s, self.S_new = self.pi_network(self.O1)
            parameters = self.parameters = self.produce_policy_parameters(
                a_shape[0], processed_s)

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
            v1 = self.v_network(self.O1, 'V')
            self.v1 = v1
            q1 = self.q_network(self.O1, self.transform_action_sample(A_sampled1), 'Q')
            log_pi_sampled1 = pi_network_log_prob(A_sampled1, 'pi', _reuse=True)
            log_pi_sampled1 *= entropy_scale  # type: tf.Tensor
            self.V_loss = V_loss = tf.reduce_mean(
                0.5 * tf.square(v1 - (q1 - log_pi_sampled1)))

            # constructing Q loss
            self.v2 = v2 = self.v_network(self.O2, 'V_bar')
            self.q1 = q = self.q_network(
                self.O1, self.transform_action_sample(A), 'Q', reuse=True)
            not_done = 1 - T  # type: tf.Tensor
            self.q_target = q_target = R + gamma * not_done * v2
            self.Q_error = tf.square(q - q_target)
            self.Q_loss = Q_loss = tf.reduce_mean(0.5 * self.Q_error)

            # constructing pi loss
            self.A_sampled2 = A_sampled2 = tf.stop_gradient(
                sample_pi_network('pi', _reuse=True))
            q2 = self.q_network(
                self.O1, self.transform_action_sample(A_sampled2), 'Q', reuse=True)
            log_pi_sampled2 = pi_network_log_prob(A_sampled2, 'pi', _reuse=True)
            log_pi_sampled2 *= entropy_scale  # type: tf.Tensor
            self.pi_loss = pi_loss = tf.reduce_mean(
                log_pi_sampled2 * tf.stop_gradient(log_pi_sampled2 - q2 + v1))

            # grabbing all the relevant variables
            def get_variables(var_name: str) -> List[tf.Variable]:
                return tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope=f'{scope}/{var_name}/')

            phi, theta, xi, xi_bar = map(get_variables, ['pi', 'Q', 'V', 'V_bar'])

            def train_op(loss, var_list, lr=learning_rate):
                optimizer = tf.train.AdamOptimizer(learning_rate=lr)
                gradients, variables = zip(
                    *optimizer.compute_gradients(loss, var_list=var_list))
                if grad_clip:
                    gradients, norm = tf.clip_by_global_norm(gradients, grad_clip)
                else:
                    norm = tf.global_norm(gradients)
                op = optimizer.apply_gradients(
                    zip(gradients, variables), global_step=self.global_step)
                return op, norm

            self.train_V, self.V_grad = train_op(loss=V_loss, var_list=xi)
            self.train_Q, self.Q_grad = train_op(loss=Q_loss, var_list=theta)
            self.train_pi, self.pi_grad = train_op(loss=pi_loss, var_list=phi)

            # embeddings
            if embed:
                with tf.variable_scope('embed'):
                    lr = embed_args.pop('learning_rate') or learning_rate
                    with tf.variable_scope('o1'):
                        o1_embed = mlp(inputs=self.O1, **embed_args)
                        # o1_embed = tf.Print(o1_embed, [o1_embed], summarize=1e5)

                        # projector stuff
                        self.o1_embed_var = tf.get_variable(
                            'o1_embed_var', shape=(batch_size, embed_args['layer_size']))
                        self.copy_o1_embed = tf.assign(self.o1_embed_var, o1_embed)
                    with tf.variable_scope('a'):
                        a_embed = mlp(inputs=self.A, **embed_args)

                    norm_a_embed = l2_normalize(a_embed, axis=1)

                with tf.variable_scope('embed2'):
                    o2_embed = mlp(inputs=self.O1, **embed_args)

                self.o1_embed = o1_embed
                self.o2_embed = o2_embed
                self.a_embed = a_embed
                self.norm_a_embed = norm_a_embed

                self.embed_loss = tf.reduce_mean(tf.norm(o1_embed + norm_a_embed -
                                                         o2_embed, axis=1))
                self.regularization = tf.reduce_mean(tf.minimum(tf.norm(a_embed,
                                                                        axis=1), 1))
                self.embed_baseline = tf.reduce_mean(tf.norm(norm_a_embed, axis=1))

                embed1_vars = tf.trainable_variables('agent/embed')
                embed2_vars = tf.trainable_variables('agent/embed2')

                self.train_embed, self.embed_grad = train_op(
                    self.embed_loss - self.regularization, embed1_vars, lr)

                self.train_embed = tf.group(self.train_embed, *[
                    tf.assign(var2, tau * var1 + (1 - tau) * var2)
                    for (var1, var2) in zip(embed1_vars, embed2_vars)
                ])

            soft_update_xi_bar_ops = [
                tf.assign(xbar, tau * x + (1 - tau) * xbar)
                for (xbar, x) in zip(xi_bar, xi)
            ]
            self.soft_update_xi_bar = tf.group(*soft_update_xi_bar_ops)
            # self.check = tf.add_check_numerics_ops()
            self.entropy = tf.reduce_mean(self.entropy_from_params(self.parameters))
            # ensure that xi and xi_bar are the same at initialization

            sess.run(tf.global_variables_initializer())

            # ensure that xi and xi_bar are the same at initialization
            hard_update_xi_bar_ops = [tf.assign(xbar, x) for (xbar, x) in zip(xi_bar, xi)]

            hard_update_xi_bar = tf.group(*hard_update_xi_bar_ops)
            sess.run(hard_update_xi_bar)

    @property
    def seq_len(self):
        return self._seq_len

    def train_step(self, step: Step) -> dict:
        feed_dict = {
            self.O1: step.o1,
            self.A:  step.a,
            self.R:  np.array(step.r) * self.reward_scale,
            self.O2: step.o2,
            self.T:  step.t,
        }
        fetch = {attr: getattr(self, attr) for attr in self.default_train_values}
        return self.sess.run(fetch, feed_dict)

    def get_actions(self, o: ArrayLike, sample: bool = True, state=None) -> NetworkOutput:
        A = self.A_sampled1 if sample else self.A_max_likelihood
        return NetworkOutput(output=self.sess.run(A, {self.O1: [o]})[0], state=0)

    def pi_network(self, o: tf.Tensor) -> NetworkOutput:
        with tf.variable_scope('pi'):
            return self.network(o)

    def q_network(self, o: tf.Tensor, a: tf.Tensor, name: str,
                  reuse: bool = None) -> tf.Tensor:
        with tf.variable_scope(name, reuse=reuse):
            oa = tf.concat([o, a], axis=1)
            return tf.reshape(tf.layers.dense(self.network(oa).output, 1, name='q'), [-1])

    def v_network(self, o: tf.Tensor, name: str, reuse: bool = None) -> tf.Tensor:
        with tf.variable_scope(name, reuse=reuse):
            return tf.reshape(tf.layers.dense(self.network(o).output, 1, name='v'), [-1])

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
                self.A:  step.a,
                self.R:  step.r,
                self.O2: step.o2,
                self.T:  step.t
            })

    def _print(self, tensor, name: str):
        return tf.Print(tensor, [tensor], message=name, summarize=1e5)

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
    def transform_action_sample(self, action_sample: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def entropy_from_params(self, params: tf.Tensor) -> tf.Tensor:
        pass
