# stdlib
import argparse
from collections import namedtuple
from contextlib import contextmanager
from functools import wraps
from itertools import filterfalse
from pathlib import Path
import re
import tempfile
from typing import List, Tuple
from xml.etree import ElementTree as ET

# third party
from gym import spaces
from gym.spaces import Box
from gym.wrappers import TimeLimit
import numpy as np
import tensorflow as tf

# first party
from environments.hindsight_wrapper import HSRHindsightWrapper, MBHSRHindsightWrapper
from environments.hsr import HSREnv, MoveGripperEnv, MultiBlockHSREnv, get_xml_filepath
from sac.train import HindsightTrainer, Trainer


def parametric_relu(_x):
    alphas = tf.get_variable(
        'alpha',
        _x.get_shape()[-1],
        initializer=tf.constant_initializer(0.0),
        dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos + neg


def make_box(*tuples: Tuple[float, float]):
    low, high = map(np.array, zip(*[(map(float, m)) for m in tuples]))
    return spaces.Box(low=low, high=high, dtype=np.float32)


ENVIRONMENTS = dict(
    multi_block=MultiBlockHSREnv,
    move_block=HSREnv,
    move_gripper=MoveGripperEnv,
)


def parse_space(dim: int):
    def _parse_space(arg: str):
        regex = re.compile('\((-?[\.\d]+),(-?[\.\d]+)\)')
        matches = regex.findall(arg)
        if len(matches) != dim:
            raise argparse.ArgumentTypeError(f'Arg {arg} must have {dim} substrings '
                                             f'matching pattern {regex}.')
        return make_box(*matches)

    return _parse_space


def parse_vector(length: int, delim: str):
    def _parse_vector(arg: str):
        vector = tuple(map(float, arg.split(delim)))
        if len(vector) != length:
            raise argparse.ArgumentError(f'Arg {arg} must include {length} float values'
                                         f'delimited by "{delim}".')
        return vector

    return _parse_vector


def cast_to_int(arg: str):
    return int(float(arg))


HINDSIGHT_ENVS = {
    HSREnv: HSRHindsightWrapper,
    MultiBlockHSREnv: MBHSRHindsightWrapper,
}

ACTIVATIONS = dict(
    relu=tf.nn.relu,
    leaky=tf.nn.leaky_relu,
    elu=tf.nn.elu,
    selu=tf.nn.selu,
    prelu=parametric_relu,
    sigmoid=tf.sigmoid,
    tanh=tf.tanh,
    none=None,
)


def parse_activation(arg: str):
    return ACTIVATIONS[arg]


def put_in_xml_setter(arg: str):
    setters = [XMLSetter(*v.split(',')) for v in arg]
    mirroring = [XMLSetter(p.replace('_l_', '_r_'), v)
                 for p, v in setters if '_l_' in p] \
                + [XMLSetter(p.replace('_r_', '_l_'), v)
                   for p, v in setters if '_r_' in p]
    return [s._replace(path=s.path) for s in setters + mirroring]


def env_wrapper(func):
    @wraps(func)
    def _wrapper(set_xml, use_dof, n_blocks, goal_space, xml_file, geofence,
                 env_args: dict, **kwargs):
        xml_filepath = get_xml_filepath(xml_file)
        if set_xml is None:
            set_xml = []
        site_size = ' '.join([str(geofence)] * 3)
        path = Path('worldbody', 'body[@name="goal"]', 'site[@name="goal"]', 'size')
        set_xml += [XMLSetter(path=f'./{path}', value=site_size)]
        with mutate_xml(
                changes=set_xml,
                dofs=use_dof,
                n_blocks=n_blocks,
                goal_space=goal_space,
                xml_filepath=xml_filepath) as temp_path:
            env_args.update(
                geofence=geofence,
                xml_file=temp_path,
                goal_space=goal_space,
            )

            return func(env_args=env_args, **kwargs)

    return lambda wrapper_args, **kwargs: _wrapper(**wrapper_args, **kwargs)


XMLSetter = namedtuple('XMLSetter', 'path value')


@contextmanager
def mutate_xml(changes: List[XMLSetter], dofs: List[str], goal_space: Box, n_blocks: int,
               xml_filepath: Path):
    def rel_to_abs(path: Path):
        return Path(xml_filepath.parent, path)

    def mutate_tree(tree: ET.ElementTree):

        worldbody = tree.getroot().find("./worldbody")
        rgba = [
            "0 1 0 1",
            "0 0 1 1",
            "0 1 1 1",
            "1 0 0 1",
            "1 0 1 1",
            "1 1 0 1",
            "1 1 1 1",
        ]

        if worldbody:
            for i in range(n_blocks):
                pos = ' '.join(map(str, goal_space.sample()))
                name = f'block{i}'

                body = ET.SubElement(worldbody, 'body', attrib=dict(name=name, pos=pos))
                ET.SubElement(
                    body,
                    'geom',
                    attrib=dict(
                        name=name,
                        type='box',
                        mass='1',
                        size=".05 .025 .017",
                        rgba=rgba[i],
                        condim='6',
                        solimp="0.99 0.99 "
                        "0.01",
                        solref='0.01 1'))
                ET.SubElement(body, 'freejoint', attrib=dict(name=f'block{i}joint'))

        for change in changes:
            parent = re.sub('/[^/]*$', '', change.path)
            element_to_change = tree.find(parent)
            if isinstance(element_to_change, ET.Element):
                print('setting', change.path, 'to', change.value)
                name = re.search('[^/]*$', change.path)[0]
                element_to_change.set(name, change.value)

        for actuators in tree.iter('actuator'):
            for actuator in list(actuators):
                if actuator.get('joint') not in dofs:
                    print('removing', actuator.get('name'))
                    actuators.remove(actuator)
        for body in tree.iter('body'):
            for joint in body.findall('joint'):
                if not joint.get('name') in dofs:
                    print('removing', joint.get('name'))
                    body.remove(joint)

        parent = Path(temp[xml_filepath].name).parent

        for include_elt in tree.findall('*/include'):
            original_abs_path = rel_to_abs(include_elt.get('file'))
            tmp_abs_path = Path(temp[original_abs_path].name)
            include_elt.set('file', str(tmp_abs_path.relative_to(parent)))

        for compiler in tree.findall('compiler'):
            abs_path = rel_to_abs(compiler.get('meshdir'))
            compiler.set('meshdir', str(abs_path))

        return tree

    included_files = [
        rel_to_abs(e.get('file')) for e in ET.parse(xml_filepath).findall('*/include')
    ]

    temp = {
        path: tempfile.NamedTemporaryFile()
        for path in (included_files + [xml_filepath])
    }
    try:
        for path, f in temp.items():
            tree = ET.parse(path)
            mutate_tree(tree)
            tree.write(f)
            f.flush()

        yield Path(temp[xml_filepath].name)
    finally:
        for f in temp.values():
            f.close()


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


def parse_groups(parser: argparse.ArgumentParser):
    args = parser.parse_args()

    def is_optional(group):
        return group.title == 'optional arguments'

    def parse_group(group):
        # noinspection PyProtectedMember
        return {a.dest: getattr(args, a.dest, None) for a in group._group_actions}

    # noinspection PyUnresolvedReferences,PyProtectedMember
    groups = [g for g in parser._action_groups if g.title != 'positional arguments']
    optional = filter(is_optional, groups)
    not_optional = filterfalse(is_optional, groups)

    kwarg_dicts = {group.title: parse_group(group) for group in not_optional}
    kwargs = (parse_group(next(optional)))
    del kwargs['help']
    return {**kwarg_dicts, **kwargs}


def add_train_args(parser):
    parser.add_argument('--logdir', type=Path, default=None)
    parser.add_argument('--load-path', type=str, default=None)
    parser.add_argument('--save-threshold', type=int, default=None)


def add_hindsight_args(parser):
    parser.add_argument('--n-goals', type=int)


def add_network_args(parser):
    parser.add_argument(
        '--activation',
        type=parse_activation,
        default=tf.nn.relu,
        choices=ACTIVATIONS.values())
    parser.add_argument('--n-hidden', type=int, required=True)
    parser.add_argument('--layer-size', type=int, required=True)


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
    parser.add_argument('--set-xml', type=put_in_xml_setter, action='append', nargs='*')
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
