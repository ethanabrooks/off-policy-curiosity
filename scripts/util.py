from pathlib import Path

import tensorflow as tf

from utils.argparse import ACTIVATIONS, cast_to_int, parse_activation


def add_network_args(parser):
    parser.add_argument(
        '--activation',
        type=parse_activation,
        default=tf.nn.relu,
        choices=ACTIVATIONS.values())
    parser.add_argument('--n-hidden', type=int, required=True)
    parser.add_argument('--layer-size', type=int, required=True)
    parser.add_argument('--no-bias', dest='use_bias', action='store_false')


def add_trainer_args(parser):
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--buffer-size', type=cast_to_int, required=True)
    parser.add_argument('--n-train-steps', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    scales = parser.add_mutually_exclusive_group(required=True)
    scales.add_argument('--reward-scale', type=float, default=1)
    scales.add_argument('--entropy-scale', type=float, default=1)
    parser.add_argument('--learning-rate', type=float, required=True)
    parser.add_argument('--grad-clip', type=float, required=True)


def add_train_args(parser):
    parser.add_argument('--logdir', type=Path, default=None)
    parser.add_argument('--load-path', type=str, default=None)
    parser.add_argument('--save-threshold', type=int, default=None)


def add_hindsight_args(parser):
    parser.add_argument('--n-goals', type=int)
