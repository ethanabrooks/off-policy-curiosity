# stdlib
import argparse
from pathlib import Path

# third party
from gym.wrappers import TimeLimit
import tensorflow as tf
from hsr.util import env_wrapper, xml_setter
from utils.argparse import parse_activation, parse_space, parse_vector, parse_groups, ACTIVATIONS

# first party
from sac.hindsight_wrapper import HSRHindsightWrapper, MBHSRHindsightWrapper
from hsr.env import HSREnv, MoveGripperEnv, MultiBlockHSREnv
from sac.train import HindsightTrainer, Trainer
from scripts.util import add_network_args, add_trainer_args, add_train_args, \
    add_hindsight_args

ENVIRONMENTS = dict(
    multi_block=MultiBlockHSREnv,
    move_block=HSREnv,
    move_gripper=MoveGripperEnv,
)

HINDSIGHT_ENVS = {
    HSREnv: HSRHindsightWrapper,
    MultiBlockHSREnv: MBHSRHindsightWrapper,
}


@env_wrapper
def main(
        env,
        max_steps,
        env_args,
        hindsight_args,
        trainer_args,
        train_args,
        embed_args,
        network_args,
):
    embed_args = {k.replace('embed_', ''): v for k, v in embed_args.items()}
    env_class = env
    env = TimeLimit(max_episode_steps=max_steps, env=env_class(**env_args))

    if hindsight_args:
        trainer = HindsightTrainer(
            env=HINDSIGHT_ENVS[env_class](env=env),
            embed_args=embed_args,
            network_args=network_args,
            **hindsight_args,
            **trainer_args)
    else:
        trainer = Trainer(
            env=env, render=False, network_args=network_args, **trainer_args)
    trainer.train(**train_args)


def add_embed_args(parser):
    parser.add_argument('--embed-n-hidden', type=int)
    parser.add_argument('--embed-layer-size', type=int)
    parser.add_argument(
        '--embed-activation',
        type=parse_activation,
        default=tf.nn.relu,
        choices=ACTIVATIONS.values())
    parser.add_argument('--embed-learning-rate', type=float)


def add_env_args(parser):
    parser.add_argument(
        '--image-dims', type=parse_vector(length=2, delim=','), default='800,800')
    parser.add_argument('--block-space', type=parse_space(dim=4))
    parser.add_argument('--min-lift-height', type=float, default=None)
    parser.add_argument('--no-random-reset', action='store_true')
    parser.add_argument('--obs-type', type=str, default=None)
    parser.add_argument('--randomize-pose', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--render-freq', type=int, default=None)
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--record-separate-episodes', action='store_true')
    parser.add_argument('--record-freq', type=int, default=None)
    parser.add_argument('--record-path', type=Path, default=None)
    parser.add_argument('--steps-per-action', type=int, required=True)


def add_wrapper_args(parser):
    parser.add_argument('--xml-file', type=Path, default='models/world.xml')
    parser.add_argument('--set-xml', type=xml_setter, action='append', nargs='*')
    parser.add_argument('--use-dof', type=str, action='append', default=[])
    parser.add_argument('--geofence', type=float, required=True)
    parser.add_argument('--n-blocks', type=int, required=True)
    parser.add_argument('--goal-space', type=parse_space(dim=3), required=True)  # TODO


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--env',
        choices=ENVIRONMENTS.values(),
        type=lambda k: ENVIRONMENTS[k],
        default=HSREnv)
    parser.add_argument('--max-steps', type=int, required=True)
    add_wrapper_args(parser=parser.add_argument_group('wrapper_args'))
    add_env_args(parser=parser.add_argument_group('env_args'))
    add_trainer_args(parser=parser.add_argument_group('trainer_args'))
    add_network_args(parser=parser.add_argument_group('network_args'))
    add_train_args(parser=parser.add_argument_group('train_args'))
    add_hindsight_args(parser=parser.add_argument_group('hindsight_args'))
    add_embed_args(parser=parser.add_argument_group('embed_args'))
    main(**(parse_groups(parser)))


if __name__ == '__main__':
    cli()
