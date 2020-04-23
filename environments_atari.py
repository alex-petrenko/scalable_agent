import numpy as np
import tensorflow as tf

from algorithms.utils.arguments import default_cfg
from envs.create_env import create_env

ATARI_W = ATARI_H = 84
ATARI_DIM = 4

ATARI_ACTION_SET = (
    0,
    1,
    2,
    3,
)

class PyProcessAtari:
    """Atari wrapper for PyProcess."""

    def __init__(self, level, config, num_action_repeats, seed, runfiles_path=None, level_cache=None):
        self._observation_spec = ['RGB_INTERLEAVED']
        env_name = 'atari_breakout'
        cfg = default_cfg(env=env_name, algo=None)
        cfg.pixel_format = 'HWC'
        cfg.res_w = ATARI_W
        cfg.res_h = ATARI_H
        self._env = create_env(env_name, cfg=cfg)

    def _reset(self):
        self._env.reset()

    # def _observation(self, obs):
    #     # return [obs, '']
    #     return obs

    def initial(self):
        obs = self._env.reset()
        return obs
        # return self._observation(obs)

    def step(self, action):
        obs, rew, done, info = self._env.step(action)
        done = np.array(done)
        if done:
            obs = self._env.reset()
        # observation = self._observation(obs)
        reward = np.array(rew, dtype=np.float32)
        return reward, done, obs

    def close(self):
        self._env.close()

    @staticmethod
    def _tensor_specs(method_name, unused_kwargs, constructor_kwargs):
        """Returns a nest of `TensorSpec` with the method's output specification."""

        observation_spec = [
            tf.contrib.framework.TensorSpec([ATARI_H, ATARI_W, ATARI_DIM], tf.uint8),
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
