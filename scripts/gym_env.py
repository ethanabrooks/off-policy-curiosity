import argparse

import click
import gym

from sac.train import Trainer
from scripts.util import add_network_args, add_train_args, add_trainer_args
from utils.argparse import parse_groups
import gym_bandits


def check_probability(ctx, param, value):
    if not (0 <= value <= 1):
        raise click.BadParameter("Param {} should be between 0 and 1".format(value))
    return value


def main(
        env,
        trainer_args,
        train_args,
        render,
        network_args,
):
    trainer = Trainer(env=env, network_args=network_args, **trainer_args)
    trainer.train(**train_args, render=render)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--env', type=gym.make, default=gym.make('CartPole-v0'))
    add_trainer_args(parser=parser.add_argument_group('trainer_args'))
    add_network_args(parser=parser.add_argument_group('network_args'))
    add_train_args(parser=parser.add_argument_group('train_args'))
    main(**(parse_groups(parser)))


if __name__ == '__main__':
    cli()
