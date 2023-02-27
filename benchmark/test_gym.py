"""
An example of RL training using StableBaselines3.

python -m dedo.run_rl_sb3 --env=HangGarment-v1 --rl_algo PPO --logdir=/tmp/dedo

tensorboard --logdir=/tmp/dedo --bind_all --port 6006

Play the saved policy (e.g. logged to PPO_210825_204955_HangGarment-v1):
python -m dedo.run_rl_sb3 --env=HangGarment-v1 --play \
    --load_checkpt=/tmp/dedo/PPO_210825_204955_HangGarment-v1


Note: this code is for research i.e. quick experimentation; it has minimal
comments for now, but if we see further interest from the community -- we will
add further comments, unify the style, improve efficiency and add unittests.

@contactrika

"""
from copy import deepcopy

from stable_baselines3.common.env_util import (make_vec_env, DummyVecEnv, SubprocVecEnv)

from dedo.utils.args import get_args
from dedo.utils.train_utils import init_train

import numpy as np
import torch

import time
import tqdm
import csv

import warnings
warnings.filterwarnings("ignore")


def main(args):
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Init RL training envs and agent.
    args.logdir, args.device = init_train(args.rl_algo, args)
    # Read num_envs from args, since it may be changed by init_train.
    n_envs = args.num_envs

    train_args = deepcopy(args)
    train_args.debug = False  # no debug during training
    train_args.viz = False  # no viz during training
    vec_env = make_vec_env(
        args.env, n_envs=n_envs,
        vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv,
        env_kwargs={'args': train_args}
    )
    vec_env.seed(args.seed)
    print('Created', args.task, 'with observation_space',
          vec_env.observation_space.shape, 'action_space',
          vec_env.action_space.shape)
    print("action_space", vec_env.action_space)

    frame_skip = args.sim_steps_per_action
    total_step = 1000

    vec_env.reset()

    done = False
    t = time.perf_counter()
    for _ in tqdm.trange(total_step):
        action = 2 * np.random.rand(n_envs, vec_env.action_space.shape[0]) - 1
        if n_envs == 1:
            if done:
                done = False
                vec_env.reset()
            else:
                done = vec_env.step(action)[2]
        else:
            vec_env.step(action)
    # FPS
    time_elapsed = time.perf_counter() - t
    fps = frame_skip * total_step * n_envs / time_elapsed
    print(f"FPS = {fps:.2f}")

    fieldnames = ['n_envs', 'frame_skip', 'total_step', 'time_elapsed', 'fps']
    # open the file in the write mode
    with open('results.csv', 'a') as f:
        # create the csv writer
        writer = csv.writer(f)
        # write header
        # writer.writerow(fieldnames)
        # data
        row = [n_envs, frame_skip, total_step, time_elapsed, fps]
        # write a row to the csv file
        writer.writerow(row)


if __name__ == "__main__":

    main(get_args())
