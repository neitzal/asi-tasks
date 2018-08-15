import argparse
import os
from multiprocessing import Queue, Process

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from tasks.fubo.gen import generate_sequential_dataset_ball_obstacle
from tasks.rr.gen import generate_sequential_dataset_room_runner
from utils.data_util import _int64_feature, _bytes_feature

compression_mode = 'GZIP'


def example_writer(queue, n_trajectories, filepath, compression_mode):
    options = tf.python_io.TFRecordOptions(
        tf.python_io.TFRecordCompressionType.__dict__[compression_mode])

    if os.path.exists(filepath):
        raise ValueError('filepath {} already exists'.format(filepath))

    writer = tf.python_io.TFRecordWriter(filepath, options)
    pbar = tqdm(total=n_trajectories)

    i_trajectory = 0
    while i_trajectory < n_trajectories:
        trajectory_frames, label = queue.get()
        trajectory_frames = np.asarray(trajectory_frames)

        feature = {'label': _int64_feature(label),
                   'x': _bytes_feature(
                       tf.compat.as_bytes(trajectory_frames.tostring())),
                   'trajectory_length': _int64_feature(len(trajectory_frames)),
                   }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
        i_trajectory += 1
        pbar.update(1)
    writer.close()


def make_dataset(dataset_maker, filepath, rng_seed, n_trajectories, part_size,
                 perturbed_dynamics,
                 first_part, last_part,
                 frameskip, max_timesteps, compression_mode, n_processes):
    queue = Queue()

    additional_info = 'fskip{}_mts{}'.format(frameskip, max_timesteps)
    grounded_filepath = filepath.format(seed=rng_seed, part_size=part_size,
                                        n=n_trajectories,
                                        compression=compression_mode,
                                        additional_info=additional_info)

    data_producers = []

    for seed in range(rng_seed + first_part * n_processes,
                      rng_seed + (first_part + 1) * n_processes):
        print('Starting data producer with seed', seed)
        data_producers.append(
            Process(target=dataset_maker,
                    args=(queue, frameskip, max_timesteps, perturbed_dynamics, seed))
        )
    for data_producer in data_producers:
        data_producer.start()

    assert n_trajectories % part_size == 0

    for i_part in range(first_part, last_part + 1):
        name, ext = os.path.splitext(grounded_filepath)
        filepath = '{}_part{}{}'.format(name, i_part, ext)
        example_writer(queue, part_size, filepath, compression_mode)
    for data_producer in data_producers:
        data_producer.terminate()


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, required=True,
                           help='Dataset identifier')
    argparser.add_argument('--seed', type=int, required=True,
                           help='Random seed')
    argparser.add_argument('--n_processes', type=int, default=1,
                           help='Number of parallel processes')
    argparser.add_argument('--first_part', type=int, default=0,
                           help='From which number to count parts')
    argparser.add_argument('--last_part', type=int, default=0,
                           help='Last part to take over')
    argparser.add_argument('--n_trajectories', type=int, required=True,
                           help='Number of trajectories')
    argparser.add_argument('--part_size', type=int, default=None,
                           help='Number of per part')
    argparser.add_argument('--frameskip', type=int, default=3,
                           help='frameskip value')
    argparser.add_argument('--output_dir', type=str,
                           help='Output directory')
    argparser.add_argument('--perturbed_dynamics', type=int, default=0,
                           help='Use a perturbed version of the task (0=standard, 1=perturbed)',
                           choices=[0, 1])
    args = argparser.parse_args()

    if args.part_size is None:
        args.part_size = args.n_trajectories

    if args.dataset == 'fubo':
        dataset_maker = generate_sequential_dataset_ball_obstacle
        h = 126
        w = 75
        max_timesteps = 750
        dataset_name = '{}_h{}_w{}_v01'.format(args.dataset, h, w)

    elif args.dataset == 'rr':
        dataset_maker = generate_sequential_dataset_room_runner
        h = 64
        w = 64
        max_timesteps = 300
        dataset_name = '{}_h{}_w{}_v01'.format(args.dataset, h, w)
    else:
        raise ValueError('Unrecognized dataset: {}'.format(args.dataset))

    if args.perturbed_dynamics:
        dataset_name += '_perturbed_dynamics'

    file_prefix = os.path.join(args.output_dir,
                               dataset_name + '_s{seed}_n{n}_'
                                              'psize{part_size}_'
                                              'cmpr{compression}_'
                                              '{additional_info}')

    actual_filepath = '{}.tfrecords'.format(file_prefix)

    make_dataset(dataset_maker=dataset_maker,
                 n_trajectories=args.n_trajectories,
                 perturbed_dynamics=args.perturbed_dynamics,
                 part_size=args.part_size,
                 first_part=args.first_part,
                 last_part=args.last_part,
                 filepath=actual_filepath,
                 compression_mode='GZIP',
                 frameskip=args.frameskip,
                 max_timesteps=max_timesteps,
                 rng_seed=args.seed,
                 n_processes=args.n_processes,
                 )


if __name__ == '__main__':
    main()
