"""
Drawing util built with gizeh (https://github.com/Zulko/gizeh)
Objects are linked to box2D bodies.
In general, everything that has the attributes 'position' and 'angle' can be used as a body
If you want to draw something unrelated to box2D, use DummyBody instead.
"""

from collections import namedtuple
from copy import copy

import gizeh
import numpy as np

import math

DummyBody = namedtuple('DummyBody', 'position angle')


def add_polygons(display_objects, body, name, drawing_layer, color):
    for i, fixture in enumerate(body.fixtures):
        part_name = name + '_part' + str(i)
        if part_name in display_objects:
            raise KeyError('Key {} already exists in display_objects!'.format(part_name))
        display_objects[part_name] = (drawing_layer, Polygon(vertices=fixture.shape.vertices,
                                                             body=body,
                                                             color=color))


def add_polygon(display_objects, root_body, vertices, name, drawing_layer, color):
    if name in display_objects:
        raise KeyError('Key {} already exists in display_objects!'.format(name))
    display_objects[name] = (drawing_layer, Polygon(vertices=vertices,
                                                    body=root_body,
                                                    color=color))


def add_circle(display_objects, root_body, radius, name, drawing_layer, color):
    if name in display_objects:
        raise KeyError('Key {} already exists in display_objects!'.format(name))
    display_objects[name] = (drawing_layer, Circle(radius,
                                                   body=root_body,
                                                   color=color))


def add_polygon_at_pos(display_objects, position, vertices, name, drawing_layer, color):
    if name in display_objects:
        raise KeyError('Key {} already exists in display_objects!'.format(name))
    display_objects[name] = (drawing_layer, Polygon(vertices=vertices,
                                                    body=DummyBody(position, angle=0),
                                                    color=color))


def add_circle_at_pos(display_objects, position, radius, name, drawing_layer, color):
    if name in display_objects:
        raise KeyError('Key {} already exists in display_objects!'.format(name))
    display_objects[name] = (drawing_layer, Circle(radius,
                                                   body=DummyBody(position, angle=0),
                                                   color=color))


class Camera(object):
    def __init__(self, pos, fov_dims, control_gain=10.0, control_damp=5.0):
        """
        Helper to translate the abstract world representation into pixel space

        :param pos: Position of camera center in world space
        :param fov_dims: (half_width, half_height) for field of view in world space
        """
        self.pos = pos
        self.fov_dims = fov_dims
        self.vel = (0, 0)
        self.control_gain = control_gain
        self.control_damp = control_damp
        self.target = (0, 0)

    def to_pixel_space(self, world_position, scale_factor):
        """
        Convert a world position to the pixel coordinates in the rendered image
        :param world_position:
        :param scale_factor: Number of pixels per world unit (in each dimension)
        :return: pixel coordinates, (0, 0) === top-left
        """
        world_diff = (world_position[0] - self.pos[0], world_position[1] - self.pos[1])
        return (scale_factor*(self.fov_dims[0] + world_diff[0]),
                scale_factor*(self.fov_dims[1] - world_diff[1]))

    def step(self, dt):
        discrepancy = tuple(self.target[i] - self.pos[i] for i in range(2))
        self.vel = tuple(self.vel[i] + dt*(self.control_gain*discrepancy[i] - self.control_damp*self.vel[i])
                         for i in range(2))
        self.pos = tuple(self.pos[i] + dt*self.vel[i] for i in range(2))

    def get_img_dims(self, scale_factor):
        return (int(round(2*scale_factor*self.fov_dims[0])),
                int(round(2*scale_factor*self.fov_dims[1])))


FrozenBody = namedtuple('FrozenBody', ['position', 'angle'])


class DisplayableObject(object):
    def __init__(self, body, color):
        self.body = body
        self.color = color

    def draw(self, surface, camera, scale_factor):
        """
        Draw the shape at the current pixel-position on the surface
        :param surface: Gizeh surface
        :param camera: Camera object for world-pixel-space-conversion
        """
        raise NotImplementedError

    def get_drawable_clone(self):
        raise NotImplementedError


class Circle(DisplayableObject):
    def __init__(self, radius, body, color, displacement=(0, 0)):
        super().__init__(body, color)
        self.radius = radius
        self.displacement = displacement

    def draw(self, surface, camera, scale_factor):
        current_world_x = self.body.position[0] + (self.displacement[0]*math.cos(self.body.angle)
                                                   - self.displacement[1]*math.sin(self.body.angle))
        current_world_y = self.body.position[1] + (self.displacement[0]*math.sin(self.body.angle)
                                                   + self.displacement[1]*math.cos(self.body.angle))

        shape = gizeh.circle(r=scale_factor*self.radius,
                             # TODO: relative rotation must affect relative world position
                             xy=camera.to_pixel_space((current_world_x, current_world_y),
                                                      scale_factor),
                             fill=self.color)
        shape.draw(surface)

    def get_drawable_clone(self):
        return Circle(self.radius,
                      FrozenBody(tuple(self.body.position), self.body.angle),
                      self.color,
                      self.displacement)


class Polygon(DisplayableObject):
    def __init__(self, vertices, body, color):
        super().__init__(body, color)
        self.vertices = vertices

    def draw(self, surface, camera, scale_factor):
        scaled_vertices = [(scale_factor*v[0], -scale_factor*v[1]) for v in self.vertices]
        shape = gizeh.polyline(points=scaled_vertices,
                               xy=camera.to_pixel_space((self.body.position[0],
                                                         self.body.position[1]), scale_factor),
                               angle=-self.body.angle,
                               fill=self.color)
        shape.draw(surface)

    def get_drawable_clone(self):
        return Polygon(copy(self.vertices),
                       FrozenBody(tuple(self.body.position), self.body.angle),
                       self.color)


def render_visual_state(state, pixels_per_worldunit,
                        bg_color=(0.3, 0.3, 0.3),
                        scale_to_multiple_of_ten=True) -> np.ndarray:
    """
    Converts a 'visual state' to a rendered image

    :param state: dict with entries 'display_objects' (iterable of tuples (int, DisplayableObject)) and 'camera'
    :param pixels_per_worldunit: number of rendered pixels per world unit
    :param bg_color: background-color-tuple (r,g,b)
    :return: rendered image as 3D-numpy array
    """
    width, height = state['camera'].get_img_dims(pixels_per_worldunit)

    if bg_color is not None and len(bg_color) == 3:
        bg_color = bg_color + (1.,)

    if scale_to_multiple_of_ten:
        surface_dims = (int(np.ceil(width / 10)) * 10,
                        int(np.ceil(height / 10)) * 10)
    else:
        surface_dims = (int(round(width)), int(round(height)))

    # Ensure that output image has dimensions as a multiple of 10 (video player compatibility)
    surface = gizeh.Surface(*surface_dims, bg_color)

    # The elements of state['display_objects'] have the drawing order as their first entry
    for _, display_object in sorted(state['display_objects'].values(),
                                    key=lambda t: t[0]):
        display_object.draw(surface, state['camera'], pixels_per_worldunit)

    img = surface.get_npimage()
    return img


def render_info(info_value, shape):
    vertical_spacing = 22
    surface = gizeh.Surface(width=shape[1], height=shape[0])
    for i, (key, value) in enumerate(info_value.items()):
        text = gizeh.text('{0}: {1:.4}'.format(key, value), fontfamily="Helvetica", fontsize=18,
                          fill=(1, 1, 1), xy=(10, (i + 0.5)*vertical_spacing), angle=0, h_align='left')
        text.draw(surface)
    return surface.get_npimage()
