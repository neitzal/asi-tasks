from collections import deque

import numpy as np
from Box2D import Box2D, b2PolygonShape, b2FixtureDef, b2Vec2

from gizeh_drawing_util import drawing_util


class Room:
    def __init__(self, y, x, height, width):
        self.y = y
        self.x = x
        self.height = height
        self.width = width
        self.children = []
        self.door = None
        self.connected_child = None

    @property
    def unconnected_children(self):
        return [child for child in self.children
                if child is not self.connected_child]


class Building:
    def __init__(self,
                 height,
                 width,
                 noise_level,
                 pixels_per_worldunit,
                 dt=1 / 10.0,
                 rng=None
                 ):
        self.display_objects = dict()
        self.camera = drawing_util.Camera(pos=(width / 2, height / 2),
                                          fov_dims=(width / 2, height / 2))
        self.pixels_per_worldunit = pixels_per_worldunit

        self.height = height
        self.width = width
        self.free_cells = np.ones((height, width), dtype=np.int32)

        self.horizontal_walls = np.zeros((height + 1, width), dtype=np.int32)
        self.vertical_walls = np.zeros((height, width + 1), dtype=np.int32)

        # make outlining arena walls
        self.vertical_walls[:, 0] = 1
        self.vertical_walls[:, -1] = 1

        self.horizontal_walls[0, :] = 1
        self.horizontal_walls[-1, :] = 1

        self.room_map = [[None for _ in range(width)]
                         for _ in range(height)]

        self.wall_thickness = 0.2

        self.agent_radius = 0.24

        self.rooms = []

        self.agent_init_position = None
        self.label = None
        self.final_room = None

        # Physics
        self.world = Box2D.b2World(gravity=(0, 0), doSleep=True)
        self.agent_body = None  # Created later
        self.dt = dt
        self.noise_level = noise_level

        self._previous_room = None
        self._current_running_style = 'direct'

        # Colors
        self.room_color = (1.0, 1.0, 1.0)
        self.bg_color = (0.15, 0.15, 0.15)
        self.wall_color = (0.0, 0.0, 0.0)
        self.door_color = (0.9, 0.95, 1.0)
        self.agent_color = (0.0, 0.8, 0.0)

        if rng is None:
            self.noise_seed = np.random.randint(0, 2 ** 32)
        else:
            self.noise_seed = rng.randint(0, 2 ** 32)
        self.noise_rng = np.random.RandomState(self.noise_seed)

    def add_init_room(self, y, x, height, width):
        room = self._add_room(y, x, height, width, parent=None)
        return room

    def add_room_connected_to(self, previous_room, height, width, rng):
        potential_locations = [
                                  (x, y)
                                  for x in range(previous_room.x - width + 1,
                                                 previous_room.x + previous_room.width)
                                  for y in [previous_room.y + previous_room.height,
                                            previous_room.y - height]
                              ] + [
                                  (x, y)
                                  for x in [previous_room.x - width,
                                            previous_room.x + previous_room.width]
                                  for y in range(previous_room.y - height + 1,
                                                 previous_room.y + previous_room.height)
                              ]
        free_locations = [location for location in potential_locations
                          if self.is_free(y=location[1], x=location[0],
                                          height=height, width=width)]

        if not free_locations:
            return None

        location = free_locations[rng.randint(len(free_locations))]
        next_room = self._add_room(y=location[1], x=location[0],
                                   height=height, width=width, parent=previous_room)
        return next_room

    def is_free(self, y, x, height, width):
        queried_cells = self.free_cells[y:y + height, x:x + width]
        s = np.sum(queried_cells)
        return s == height * width

    def _add_room(self, y, x, height, width, parent):
        room = Room(y, x, height, width)
        room_vertices = self.get_room_vertices(height, width)
        drawing_util.add_polygon(self.display_objects,
                                 drawing_util.DummyBody((x, y), 0),
                                 room_vertices,
                                 'room_{}'.format(len(self.display_objects)),
                                 drawing_layer=0,
                                 color=self.room_color)
        all_wall_vertices = self.get_wall_vertices(height, width)
        for wall_vertices in all_wall_vertices:
            drawing_util.add_polygon(self.display_objects,
                                     drawing_util.DummyBody((x, y), 0),
                                     wall_vertices,
                                     'wall_{}'.format(len(self.display_objects)),
                                     drawing_layer=2,
                                     color=self.wall_color)
        # Bookkeeping
        self.free_cells[y:y + height, x:x + width] = 0
        self.horizontal_walls[y, x:x + width] = 1
        self.horizontal_walls[y + height, x:x + width] = 1
        self.vertical_walls[y:y + height, x] = 1
        self.vertical_walls[y:y + height, x + width] = 1

        if parent is not None:
            parent.children.append(room)
        self.rooms.append(room)

        for i in range(height):
            for j in range(width):
                self.room_map[y + i][x + j] = room

        return room

    def get_room_vertices(self, height, width):
        room_vertices = [(0, 0),
                         (width, 0),
                         (width, height),
                         (0, height)]
        return room_vertices

    def get_wall_vertices(self, height, width):
        all_wall_vertices = [[(_x - w / 2, _y - h / 2),
                              (_x + w / 2, _y - h / 2),
                              (_x + w / 2, _y + h / 2),
                              (_x - w / 2, _y + h / 2)]
                             for _x, _y, w, h in [(0, height / 2,
                                                   self.wall_thickness, height),
                                                  (width, height / 2,
                                                   self.wall_thickness, height),
                                                  (width / 2, 0,
                                                   width, self.wall_thickness),
                                                  (width / 2, height,
                                                   width, self.wall_thickness),
                                                  ]]
        return all_wall_vertices

    def add_random_door(self, from_room, to_room, rng):
        adjacent_cells = find_adjacent_cells(from_room.y, from_room.x,
                                             from_room.height, from_room.width,
                                             to_room.y, to_room.x,
                                             to_room.height, to_room.width)
        door_cell = adjacent_cells[rng.randint(len(adjacent_cells))]
        self._make_door(door_cell[0], door_cell[1], door_cell[2], door_cell[3])
        from_room.door = door_cell

    def _make_door(self, y, x, dy, dx):
        overhang = 0.4
        door_vertices = [(0.25 + overhang * min(0, dx), 0.25 + overhang * min(0, dy)),
                         (0.25 + overhang * min(0, dx), 0.75 + overhang * max(0, dy)),
                         (0.75 + overhang * max(0, dx), 0.75 + overhang * max(0, dy)),
                         (0.75 + overhang * max(0, dx), 0.25 + overhang * min(0, dy))]
        drawing_util.add_polygon(self.display_objects,
                                 drawing_util.DummyBody((x, y), 0),
                                 door_vertices,
                                 'door_{}'.format(len(self.display_objects)),
                                 drawing_layer=2,
                                 color=self.door_color)
        if dx == 1:
            self.vertical_walls[y, x + 1] = 0
        elif dx == -1:
            self.vertical_walls[y, x] = 0
        elif dy == 1:
            self.horizontal_walls[y + 1, x] = 0
        elif dy == -1:
            self.horizontal_walls[y, x] = 0

    def _build_physics_scene(self):
        horizontal_wall_vertices = [(0, -self.wall_thickness / 2),
                                    (1, -self.wall_thickness / 2),
                                    (1, self.wall_thickness / 2),
                                    (0, self.wall_thickness / 2)]
        vertical_wall_vertices = [(-self.wall_thickness / 2, 0),
                                  (-self.wall_thickness / 2, 1),
                                  (self.wall_thickness / 2, 1),
                                  (self.wall_thickness / 2, 0)]

        def make_wall(y, x, vertices):
            wall_fixture = b2FixtureDef(
                shape=b2PolygonShape(vertices=vertices),
                friction=0.4,
                restitution=0.3)
            wall_body = self.world.CreateStaticBody(
                position=(x, y),
                fixtures=wall_fixture)

            return wall_body

        for y in range(self.height + 1):
            for x in range(self.width + 1):
                try:
                    if self.horizontal_walls[y, x] == 1:
                        make_wall(y, x, horizontal_wall_vertices)
                except IndexError:
                    pass

                try:
                    if self.vertical_walls[y, x] == 1:
                        make_wall(y, x, vertical_wall_vertices)
                except IndexError:
                    pass

    def add_agent(self, y, x):
        self.agent_init_position = (x, y)

        agent_vertices = [(0.5 - self.agent_radius, 0.5 - self.agent_radius),
                          (0.5 - self.agent_radius, 0.5 + self.agent_radius),
                          (0.5 + self.agent_radius, 0.5 + self.agent_radius),
                          (0.5 + self.agent_radius, 0.5 - self.agent_radius)]
        agent_fixture = b2FixtureDef(
            shape=b2PolygonShape(vertices=agent_vertices),
            friction=0.4,
            restitution=0.3,
        )
        self.agent_body = self.world.CreateDynamicBody(
            position=(x, y),
            fixtures=agent_fixture,
            fixedRotation=True,
        )

        visual_radius = 1.5 * self.agent_radius
        agent_vertices_visual = [(0.5 - visual_radius, 0.5 - visual_radius),
                                 (0.5 - visual_radius, 0.5 + visual_radius),
                                 (0.5 + visual_radius, 0.5 + visual_radius),
                                 (0.5 + visual_radius, 0.5 - visual_radius)]
        drawing_util.add_polygon(self.display_objects,
                                 self.agent_body,
                                 agent_vertices_visual,
                                 'agent',
                                 drawing_layer=5,
                                 color=self.agent_color)

        self.position_indicator = drawing_util.DummyBody(b2Vec2(x, y), angle=0)
        self.target_indicator = drawing_util.DummyBody(b2Vec2(x, y), angle=0)

    def color_room(self, room, color):
        room_vertices = self.get_room_vertices(room.height, room.width)
        drawing_util.add_polygon(self.display_objects,
                                 drawing_util.DummyBody((room.x, room.y), 0),
                                 room_vertices,
                                 'room_{}'.format(len(self.display_objects)),
                                 drawing_layer=1,
                                 color=color)

    @property
    def agent_x_discrete(self):
        return int(self.agent_body.position.x + 0.5)

    @property
    def agent_y_discrete(self):
        return int(self.agent_body.position.y + 0.5)

    @property
    def current_room(self):
        return self.room_map[self.agent_y_discrete][self.agent_x_discrete]

    def step(self):
        current_room = self.current_room

        if self._previous_room is not current_room:
            # Select new style of walking through this room
            if self.noise_rng.uniform(0, 1) < self.noise_level:
                self._current_running_style = 'direct'
            else:
                self._current_running_style = 'explore'
            self._previous_room = current_room

        if current_room is None:
            return

        elif current_room.door is not None:
            door_y, door_x, door_dy, door_dx = current_room.door

            # center point
            target_y = door_y + 0.5 * door_dy
            target_x = door_x + 0.5 * door_dx

            pre_delta_y = target_y - self.agent_body.position.y
            pre_delta_x = target_x - self.agent_body.position.x

            alpha = 0.45
            beta = 1.0
            gamma = -0.2
            modification_y = door_dy * (
                    gamma + alpha - min(2 * alpha, abs(pre_delta_x / beta)))
            modification_x = door_dx * (
                    gamma + alpha - min(2 * alpha, abs(pre_delta_y / beta)))

            target_y = target_y + modification_y
            target_x = target_x + modification_x

            distance_from_door = np.linalg.norm([pre_delta_x, pre_delta_y])

            target_center_modifier = min(1.0, distance_from_door / 2.0)
            # target_center_modifier = 1.0
            target_y = ((1 - target_center_modifier) * target_y
                        + target_center_modifier * (
                                current_room.y + current_room.height / 2 - 0.5))
            target_x = ((1 - target_center_modifier) * target_x
                        + target_center_modifier * (
                                current_room.x + current_room.width / 2 - 0.5))


        else:
            # current_room.door is None
            target_y = current_room.y + current_room.height / 2 - 0.5
            target_x = current_room.x + current_room.width / 2 - 0.5

        self.target_indicator.position.y = target_y
        self.target_indicator.position.x = target_x
        delta_y = target_y - self.agent_body.position.y
        delta_x = target_x - self.agent_body.position.x

        if self._current_running_style == 'explore':
            force_multiplier = 3.5
        else:
            force_multiplier = 10.0

        drag_multiplier = 5.0

        clipped_delta_y = np.sign(delta_y)
        clipped_delta_x = np.sign(delta_x)

        self.agent_body.ApplyForceToCenter((force_multiplier * clipped_delta_x,
                                            force_multiplier * clipped_delta_y),
                                           wake=True)

        # Drag
        self.agent_body.ApplyForceToCenter(
            -drag_multiplier * self.agent_body.linearVelocity,
            wake=True)

        self.world.Step(self.dt, velocityIterations=5, positionIterations=8)

        discrete_x = int(self.agent_body.position.x + 0.5)
        discrete_y = int(self.agent_body.position.y + 0.5)
        self.position_indicator.position.x = discrete_x
        self.position_indicator.position.y = discrete_y

    def get_label(self):
        return self.label

    def reset(self):
        if self.agent_init_position is None:
            raise ValueError('Please initialize the agent first')

        self.agent_body.position = self.agent_init_position
        self.agent_body.linearVelocity = (0, 0)
        self.agent_body.angularVelocity = 0
        self.agent_body.angle = 0
        self.noise_rng = np.random.RandomState(self.noise_seed)

    def rollout(self, step_timeout=500):
        self.reset()
        step_counter = 0
        timeout_occurred = False
        while self.current_room is not self.final_room:
            if step_counter >= step_timeout:
                timeout_occurred = True
                break

            step_counter += 1
            self.step()

        if timeout_occurred:
            return {'n_steps': None,
                    'aborted': True}
        else:
            return {'n_steps': step_counter,
                    'aborted': False}

    def render(self, object_filter=None, override_bg_color=None):
        if object_filter is None:
            display_objects = self.display_objects
        else:
            display_objects = {k: v for k, v in self.display_objects.items()
                               if object_filter(k)}
        if override_bg_color:
            bg_color = override_bg_color
        else:
            bg_color = self.bg_color

        return drawing_util.render_visual_state(
            {'camera': self.camera,
             'display_objects': display_objects},
            pixels_per_worldunit=self.pixels_per_worldunit,
            bg_color=bg_color,
            scale_to_multiple_of_ten=False)


def find_adjacent_cells(from_y, from_x, from_h, from_w,
                        to_y, to_x, to_h, to_w):
    dx = to_x - from_x
    dy = to_y - from_y

    if dx == from_w:
        return [(y, from_x + from_w - 1,
                 0,  # dy
                 1  # dx
                 )
                for y in range(max(from_y, to_y),
                               min(from_y + from_h, to_y + to_h))]
    elif dx == -to_w:
        return [(y, from_x,
                 0,  # dy
                 -1  # dx
                 )
                for y in range(max(from_y, to_y),
                               min(from_y + from_h, to_y + to_h))]
    elif dy == from_h:
        return [(from_y + from_h - 1, x,
                 1,  # dy
                 0  # dx
                 )
                for x in range(max(from_x, to_x),
                               min(from_x + from_w, to_x + to_w))]
    elif dy == -to_h:
        return [(from_y, x,
                 -1,  # dy
                 0  # dx
                 )
                for x in range(max(from_x, to_x),
                               min(from_x + from_w, to_x + to_w))]
    else:
        raise ValueError('Rooms are not adjacent')


def sample_building(rng=None,
                    pixels_per_worldunit=16,
                    perturbed_dynamics=0,
                    building_height=16,
                    building_width=16,
                    ):
    if rng is None:
        rng = np.random.RandomState()

    if perturbed_dynamics:
        noise_level = 1.0
    else:
        noise_level = 0.5

    building = Building(height=building_height,
                        width=building_width,
                        noise_level=noise_level,
                        pixels_per_worldunit=pixels_per_worldunit,
                        rng=rng)

    init_pos_y = rng.randint(5, building_height - 3 - 5)
    init_pos_x = rng.randint(5, building_width - 2 - 5)

    first_room = building.add_init_room(init_pos_y, init_pos_x, 3, 2)
    room_buffer = deque([first_room])
    j = 0
    while room_buffer:
        room = room_buffer.popleft()

        children = []
        for i in range(2 if j % 8 == 0 else 1):
            if rng.uniform(0.0, 1.0) < 0.5:
                height, width = 3, 2
            else:
                height, width = 2, 3

            next_room = building.add_room_connected_to(room,
                                                       height=height,
                                                       width=width,
                                                       rng=rng)
            if next_room is not None:
                room_buffer.append(next_room)
                children.append(next_room)

        if children:
            subsequent_room = rng.choice(children)
            building.add_random_door(room, subsequent_room, rng=rng)
            room.connected_child = subsequent_room

        j += 1

    # Make label
    building.label = rng.choice([0, 1], p=[0.5, 0.5])

    # Find final room:
    _room = first_room
    while _room.connected_child is not None:
        _room = _room.connected_child
    building.final_room = _room

    # Find fake room:
    _room = first_room.unconnected_children[0]
    while _room.connected_child is not None:
        _room = _room.connected_child
    fake_room = _room

    final_room_color = (0.5, 0.5, 1.0)
    fake_room_color = (1.0, 0.5, 0.5)

    if building.label == 1:
        building.color_room(building.final_room, final_room_color)
        building.color_room(fake_room, fake_room_color)
    else:
        building.color_room(building.final_room, fake_room_color)
        building.color_room(fake_room, final_room_color)

    building._build_physics_scene()
    building.add_agent(y=init_pos_y, x=init_pos_x)
    return building


