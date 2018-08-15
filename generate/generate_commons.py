import numpy as np


def get_filtered_instance(instance_sampler,
                          max_timesteps
                          ):
    """Rejection sampling approach"""
    instance = None
    done = False
    while not done:
        instance = instance_sampler()
        rollout_result = instance.rollout(step_timeout=max_timesteps)

        if not rollout_result['aborted']:
            done = True
        instance.reset()
    return instance


def to_grayscale(img, keepdims=True):
    return np.mean(img, axis=-1, keepdims=keepdims)
