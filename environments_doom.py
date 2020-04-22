import random

import numpy as np

import tensorflow as tf
from filelock import FileLock, Timeout

from algorithms.utils.arguments import default_cfg
from envs.create_env import create_env


DOOM_W = 128
DOOM_H = 72
DOOM_DIMENSION = 3

# use the same number of actions as dmlab
DOOM_ACTION_SET = (
    0, #MOVE_FORWARD
    1, #MOVE_BACKWARD
    2, #MOVE_RIGHT
    3, #MOVE_LEFT
    4, #TURN_RIGHT
    5, #TURN_LEFT
    6, #ATTACK
    7, #SPEED
    8, #USE
)


DOOM_LOCK_PATH = '/tmp/doom_impala_lock'


class PyProcessDoom:
  """VizDoom wrapper for PyProcess."""

  def __init__(self, level, config, num_action_repeats, seed, runfiles_path=None, level_cache=None):
    self._observation_spec = ['RGB_INTERLEAVED']
    env_name = 'doom_benchmark'
    cfg = default_cfg(env=env_name, algo=None)
    cfg.pixel_format = 'HWC'
    cfg.res_w = DOOM_W
    cfg.res_h = DOOM_H
    cfg.wide_aspect_ratio = False
    self._env = create_env(env_name, cfg=cfg)

    lock = FileLock(DOOM_LOCK_PATH)
    attempt = 0
    while True:
        attempt += 1
        try:
            with lock.acquire(timeout=10):
                print('Env created, resetting...')
                self._env.reset()
                print('Env reset completed!')
                break
        except Timeout:
            print('Another instance of this application currently holds the lock, attempt:', attempt)

  def _reset(self):
    self._env.reset()

  def _observation(self, obs):
    # return [obs, '']
    return obs

  def initial(self):
    obs = self._env.reset()
    return self._observation(obs)

  def step(self, action):
    obs, rew, done, info = self._env.step(action)
    done = np.array(done)
    if done:
      obs = self._env.reset()
    observation = self._observation(obs)
    reward = np.array(rew, dtype=np.float32)
    return reward, done, observation

  def close(self):
    self._env.close()

  @staticmethod
  def _tensor_specs(method_name, unused_kwargs, constructor_kwargs):
    """Returns a nest of `TensorSpec` with the method's output specification."""

    observation_spec = [
        tf.contrib.framework.TensorSpec([DOOM_H, DOOM_W, DOOM_DIMENSION], tf.uint8),
        # tf.contrib.framework.TensorSpec([], tf.string),
    ]

    if method_name == 'initial':
      return observation_spec
    elif method_name == 'step':
      return (
          tf.contrib.framework.TensorSpec([], tf.float32),
          tf.contrib.framework.TensorSpec([], tf.bool),
          observation_spec,
      )
