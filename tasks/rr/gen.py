import numpy as np
import math
from functools import partial

import skimage.transform

from generate.generate_commons import get_filtered_instance
from tasks.rr.room_runner import sample_building


def generate_sequential_dataset_room_runner(queue,
                                            frameskip,
                                            max_timesteps,
                                            perturbed_dynamics,
                                            rng_seed):
    rng = np.random.RandomState(rng_seed)
    building_sampler = get_building_sampler(perturbed_dynamics, rng)

    while 1:
        label = None

        while label is None:
            building = get_filtered_instance(building_sampler,
                                                        max_timesteps=max_timesteps)
            label = building.get_label()


        assert label is not None

        rollout_result = building.rollout()
        building.reset()

        trajectory_frames = []

        n_steps_added_to_end = 6
        n_steps = rollout_result['n_steps'] + n_steps_added_to_end
        for i_frame in range(0,
                             int(math.ceil(n_steps / frameskip)) + 1):
            current_img = building.render()
            current_img = skimage.transform.downscale_local_mean(current_img, (4, 4, 1))
            current_img = np.clip(current_img, 0, 255).astype(np.uint8)

            trajectory_frames.append(current_img)

            for _ in range(frameskip):
                building.step()

        queue.put((trajectory_frames, label))


def get_building_sampler(perturbed_dynamics, rng):
    room_sampler = partial(sample_building,
                           perturbed_dynamics=perturbed_dynamics,
                           pixels_per_worldunit=16,  # downsampled later by factor of 4
                           rng=rng)
    return room_sampler
