import Box2D
import numpy as np
from Box2D import b2FixtureDef, b2PolygonShape, b2CircleShape

from gizeh_drawing_util import drawing_util


class ObstacleCourse:
    arena_bounds_x = (-6., 6.)
    arena_bounds_y = (-6., 6.)

    basket_slack = 0.5  # Camera is moved by this amount to make the basket visible

    obstacle_friction = .5
    obstacle_restitution = .1

    funnel_friction = .5
    funnel_restitution_max = 0.6
    funnel_restitution_min = 0.02

    ball_density = 1.0
    ball_friction = 0.5
    ball_restitution = 0.2

    velocity_iterations = 6
    position_iterations = 2

    basket_width = 2.1
    basket_height = 0.2
    basket_vertices = [(-basket_width / 2, -0.01),
                       (-basket_width / 2, -basket_height - 0.01),
                       (basket_width / 2, -basket_height - 0.01),
                       (basket_width / 2, -0.01)]

    obstacle_color = (0.0, 0.0, 0.0)
    bg_color = (1.0, 1.0, 1.0)
    ball_color = (0.0, 0.0, 1.0)
    basket_colors = [(0.0, 1.0, 0.0),
                     (1.0, 1.0, 0.0),
                     (1.0, 0.0, 0.0),
                     (1.0, 0.0, 1.0),
                     (0.0, 0.0, 1.0),
                     ]

    def __init__(self, dt, ball_radius=0.2, pixels_per_worldunit=8):
        """
        :param dt: Timestep delta
        """
        self.dt = dt
        self.ball_radius = ball_radius
        self.ball_init_position = None
        self.ball_init_velocity = (0, -0.001)
        self.last_row_xs = []
        self.camera = drawing_util.Camera(pos=(0, -self.basket_slack),
                                          fov_dims=(max(abs(self.arena_bounds_x[0]),
                                                        abs(self.arena_bounds_x[1])),
                                                    max(abs(self.arena_bounds_y[0]),
                                                        abs(self.arena_bounds_y[1]))))

        self.display_objects = dict()
        self.world = Box2D.b2World(gravity=(0, -10.0), doSleep=True)
        self.obstacle_bodies = []
        ball_fixture = b2FixtureDef(shape=b2CircleShape(radius=self.ball_radius),
                                    density=self.ball_density,
                                    friction=self.ball_friction,
                                    restitution=self.ball_restitution)
        self.ball_body = self.world.CreateDynamicBody(position=self.ball_init_position,
                                                      fixtures=ball_fixture)
        self.pixels_per_worldunit = pixels_per_worldunit

        drawing_util.add_circle(self.display_objects,
                                self.ball_body,
                                self.ball_radius,
                                name='ball',
                                drawing_layer=1,
                                color=self.ball_color
                                )

    def add_basket(self, i, x):
        basket_fixture = b2FixtureDef(shape=b2PolygonShape(vertices=self.basket_vertices),
                                      friction=1.0,
                                      restitution=0.0)
        basket_body = self.world.CreateStaticBody(
            position=(x, self.arena_bounds_y[0] - self.ball_radius),
            fixtures=basket_fixture)
        drawing_util.add_polygon(self.display_objects, basket_body,
                                 self.basket_vertices,
                                 'basket_{}'.format(i), 2, self.basket_colors[i])

    def step(self):
        self.world.Step(self.dt, self.velocity_iterations, self.position_iterations)

    def reset(self):
        if self.ball_init_position is None:
            raise ValueError(
                'Call set_ball_init_position before resetting the environment')
        self.ball_body.position = self.ball_init_position
        self.ball_body.angle = 0
        self.ball_body.linearVelocity = self.ball_init_velocity
        self.ball_body.angularVelocity = 0
        self.ball_body.awake = True

    def set_ball_init_position(self, x, y):
        self.ball_body.position = self.ball_init_position = (x, y)

    def add_funnel(self, position, restitution, width=4.0, height=2.0,
                   thickness=0.2, gap=0.5):
        resolution = 16
        alphas = np.linspace(-np.pi / 2, np.pi / 2, resolution)

        # xs = (width / 4) * np.sin(np.pi / 2 * np.sin(alphas))

        # xs = (width / 4) * np.sin(np.sin(np.sin(alphas) * np.pi/2)* np.pi/2)
        # ys = height * (alphas / np.pi)
        amp = 0.5 * (width - 2 * thickness - gap)
        xs = amp * (alphas / np.pi)
        ys = height * np.arcsin(np.arcsin(alphas / (np.pi / 2)) / (np.pi / 2)) / np.pi

        def funnel_part_fixtures(xs, ys):
            all_fixture_vertices = [
                list(zip(xs[i:i + 2] - thickness / 2, ys[i:i + 2]))
                + list(zip(reversed(xs[i:i + 2] + thickness / 2), reversed(ys[i:i + 2])))
                for i in range(resolution - 1)
            ]
            fixtures = []

            for fixture_vertices in all_fixture_vertices:
                fixture = b2FixtureDef(shape=b2PolygonShape(vertices=fixture_vertices),
                                       density=1.0,
                                       friction=self.funnel_friction,
                                       restitution=restitution)
                fixtures.append(fixture)
            return fixtures, all_fixture_vertices

        displacement = 0.5 * (thickness + gap + amp)
        fixtures_r, all_fixture_vertices_r = funnel_part_fixtures(xs + displacement, ys)
        fixtures_l, all_fixture_vertices_l = funnel_part_fixtures(-xs - displacement, ys)

        fixtures = fixtures_l + fixtures_r
        all_fixture_vertices = all_fixture_vertices_l + all_fixture_vertices_r

        funnel_body = self.world.CreateStaticBody(position=position,
                                                  angle=0,
                                                  fixtures=fixtures)
        for fixture_vertices in all_fixture_vertices:
            drawing_util.add_polygon(self.display_objects,
                                     funnel_body,
                                     fixture_vertices,
                                     name='funnel{}'.format(len(self.display_objects)),
                                     drawing_layer=1,
                                     color=self.obstacle_color)

    def add_obstacle(self, width, thickness, position, angle):
        vertices = [(-width / 2, thickness / 2),
                    (-width / 2, -thickness / 2),
                    (width / 2, -thickness / 2),
                    (width / 2, thickness / 2)]

        fixture = b2FixtureDef(shape=b2PolygonShape(vertices=vertices),
                               density=1.0,
                               friction=self.obstacle_friction,
                               restitution=self.obstacle_restitution)
        obstacle_body = self.world.CreateStaticBody(position=position,
                                                    angle=angle,
                                                    fixtures=fixture)
        number = len(self.obstacle_bodies)
        drawing_util.add_polygon(self.display_objects,
                                 obstacle_body,
                                 vertices,
                                 name='obstacle{}'.format(number),
                                 drawing_layer=1,
                                 color=self.obstacle_color)
        self.obstacle_bodies.append(obstacle_body)

    def rollout(self, step_timeout=1000):
        self.reset()
        step_counter = 0
        last_position = None
        timeout_occurred = False
        while self.ball_body.position.y > self.arena_bounds_y[0]:
            if step_counter >= step_timeout:
                timeout_occurred = True
                break

            last_position = (self.ball_body.position.x, self.ball_body.position.y)
            step_counter += 1
            self.ball_body.awake = True
            self.step()

        # Interpolate linearly over last timestep to get the x position of the intersection
        if not timeout_occurred and last_position is not None:
            alpha = (last_position[1] - self.arena_bounds_y[0]) / (
                    last_position[1] - self.ball_body.position.y)
            intersection_x = alpha * last_position[0] + (1 - alpha) * (
                self.ball_body.position.x)
        elif not timeout_occurred:
            assert (last_position is None)
            intersection_x = self.ball_body.position.x
        else:
            assert timeout_occurred
            return {
                'n_steps': step_counter,
                'destination_x': None,
                'aborted': True,
            }
        # Check within-bounds
        if not self.arena_bounds_x[0] <= intersection_x <= self.arena_bounds_x[1]:
            return {
                'n_steps': step_counter,
                'destination_x': None,
                'aborted': True,
            }

        return {
            'n_steps': step_counter,
            'destination_x': intersection_x,
            'aborted': False
        }

    def get_label(self):
        """
        Determine the label
        """
        rollout_result = self.rollout()  # TODO: redundant, could use a cached value
        destination_x = rollout_result['destination_x']
        if destination_x is None:
            print('rollout result negative!')
            return None
        label = np.argmin(np.abs(destination_x - np.array(self.last_row_xs)))
        for i, x in enumerate(self.last_row_xs):
            self.add_basket(i, x)

        self.reset()
        return label

    def render(self, hide_obstacles=False):
        if hide_obstacles:
            display_objects = {k: v for k, v in self.display_objects.items() if
                               not k.startswith('obstacle')}
        else:
            display_objects = self.display_objects

        return drawing_util.render_visual_state({'camera': self.camera,
                                                 'display_objects': display_objects},
                                                pixels_per_worldunit=self.pixels_per_worldunit,
                                                bg_color=self.bg_color,
                                                scale_to_multiple_of_ten=False)


def sample_course(ball_init_y_bias,
                  grid_n_x,
                  grid_n_y,
                  ball_radius,
                  perturbed_dynamics,
                  obstacle_thickness,
                  dt,
                  rng,
                  pixels_per_worldunit):
    grid_spacing_x = 10.0 / grid_n_x
    grid_spacing_y = 12.0 / 4
    arena_height = (grid_n_y + 1) * grid_spacing_y

    ball_init_x = (0.5 + rng.randint(grid_n_x) - (grid_n_x - 1) / 2) * grid_spacing_x
    ball_init_y = arena_height / 2 - 4 * ball_radius + ball_init_y_bias

    ObstacleCourse.arena_bounds_x = (-6.25, 6.25)
    # ObstacleCourse.arena_bounds_y = (-8.125, 8.125)
    ObstacleCourse.arena_bounds_y = (-arena_height / 2, arena_height / 2)

    if perturbed_dynamics:
        ObstacleCourse.ball_restitution = 0.05
        ObstacleCourse.funnel_restitution_min = 0.02
        ObstacleCourse.funnel_restitution_max = 0.02

    course = ObstacleCourse(dt, ball_radius, pixels_per_worldunit=pixels_per_worldunit)

    course.last_row_xs = []

    course.set_ball_init_position(ball_init_x, ball_init_y)
    if grid_spacing_x <= 2 * ball_radius:
        raise ValueError('Gaps are too small for the ball. '
                         'Please lower the number of horizontal obstacles.')

    for i in range(grid_n_y + 1):
        for j in range(grid_n_x + (i + 1) % 2):
            if i > 0:
                obstacle_width = rng.uniform(0.6, 0.8) * (
                        grid_spacing_x - 2 * ball_radius)
                x = (((i % 2) / 2 + j - 0.5 - (grid_n_x - 1) / 2) * grid_spacing_x
                     )
                y = (i - 1 - (grid_n_y - 1) / 2) * grid_spacing_y
                abs_angle = rng.uniform(np.pi / 24, np.pi / 5)
                angle = rng.choice([-1, 1]) * abs_angle
                if (i + 1) % 2 == 1:
                    if j == 0:
                        angle = -abs_angle
                    if j == grid_n_x:
                        angle = abs_angle

                course.add_obstacle(obstacle_width, obstacle_thickness,
                                    (x, y), angle)

            if i < grid_n_y and (rng.uniform() < .7 or i == 0):
                x = (((i % 2) / 2 + j - 0.5 - (grid_n_x - 1) / 2) * grid_spacing_x
                     + rng.uniform(-grid_spacing_x / 32, grid_spacing_x / 32))
                y = (((i - 0.5) - (grid_n_y - 1) / 2) * grid_spacing_y
                     + rng.uniform(-grid_spacing_y / 16, 3 * grid_spacing_y / 16))
                if rng.uniform(0, 1) < 0.5:
                    restitution = course.funnel_restitution_min
                else:
                    restitution = course.funnel_restitution_max

                gap_width = rng.uniform(2.09, 2.2) * ball_radius
                funnel_height = rng.uniform(1.4, 1.52)
                course.add_funnel((x, y), restitution=restitution,
                                  width=1.1 * grid_spacing_x, height=funnel_height,
                                  thickness=rng.uniform(0.15, 0.3),
                                  gap=gap_width)
                if i == 0:
                    course.last_row_xs.append(x)

    course.reset()
    return course


def scale_up(img, factor):
    return np.repeat(np.repeat(img, factor, axis=-3), factor, axis=-2)
