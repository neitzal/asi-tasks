import math
from functools import partial

import numpy as np

from generate.generate_commons import get_filtered_instance, to_grayscale
from tasks.fubo.funnel_board import sample_course


def generate_sequential_dataset_ball_obstacle(queue,
                                              frameskip,
                                              max_timesteps,
                                              perturbed_dynamics,
                                              rng_seed):
    rng = np.random.RandomState(rng_seed)
    course_sampler = get_course_sampler(perturbed_dynamics, rng)

    while 1:
        label = None

        while label is None:
            course = get_filtered_instance(course_sampler,
                                                      max_timesteps=max_timesteps)
            label = course.get_label()

        assert label is not None

        rollout_result = course.rollout()
        course.reset()
        course.ball_body.awake = True

        trajectory_frames = []
        last_img = None
        for i_frame in range(0,
                             int(math.ceil(rollout_result['n_steps'] / frameskip)) + 1):
            current_img = course.render(hide_obstacles=False)
            current_grayscale_img = to_grayscale(current_img)
            if last_img is None:
                last_img = current_grayscale_img

            img_diff = current_grayscale_img.astype(np.int16) - last_img.astype(np.int16)
            diff_pos = np.clip(img_diff, 0, 255)
            diff_neg = np.clip(img_diff, -255, 0)

            combined_img = np.clip(current_img + np.concatenate(
                (-diff_pos, diff_neg, np.zeros_like(diff_pos)),
                axis=-1
            ), 0, 255).astype(np.uint8)
            trajectory_frames.append(combined_img)

            last_img = current_grayscale_img
            for _ in range(frameskip):
                course.step()

            if (course.ball_body.position.y <
                    1.2 * course.arena_bounds_y[0] - course.ball_radius):
                print('Ball is already out of bounds.')
                break

        if (course.ball_body.position.y >
                course.arena_bounds_y[0]
                + course.ball_radius + 1.5 * course.basket_height):
            print('Ball seemed stuck. Skipping...')
            continue

        queue.put((trajectory_frames, label))


def get_course_sampler(perturbed_dynamics, rng):
    course_sampler = partial(sample_course,
                             ball_init_y_bias=0.0,
                             grid_n_x=4,
                             grid_n_y=6,
                             obstacle_thickness=0.2,
                             ball_radius=0.25,
                             perturbed_dynamics=perturbed_dynamics,
                             dt=1 / 30.,
                             rng=rng,
                             pixels_per_worldunit=6)
    return course_sampler