
import bpy
import sys
import os
import contextlib
import mathutils
from mathutils import Vector
import numpy as np
import math


argv = sys.argv
argv = argv[argv.index("--") + 1:]

### Scene setup
# Clear scene
bpy.ops.wm.read_factory_settings(use_empty=True)


def spherical_to_cartesian(elev_deg, azim_deg, r):
    elev = math.radians(elev_deg)
    azim = math.radians(azim_deg)
    x = r * math.cos(elev) * math.cos(azim)
    z = r * math.cos(elev) * math.sin(azim)
    y = r * math.sin(elev)
    return mathutils.Vector((x, y, z))


# Define lights
light_specs = [
    {"location": (2, 3, -2), "energy": 1000},   # Key light
    {"location": (-2, 3, -2), "energy": 300},   # Fill light
    {"location": (-2, -3, -2), "energy": 300},   # Fill light
    {"location": (-2, 2, 2), "energy": 500},    # Rim/back light
]

for spec in light_specs:
    bpy.ops.object.light_add(type='AREA', location=spec["location"])
    light = bpy.context.object
    light.data.energy = spec["energy"]
    light.data.size = 1.25  # slightly smaller = sharper shadows

    # Aim light at object center
    direction = mathutils.Vector((0, 0, 0)) - light.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    light.rotation_euler = rot_quat.to_euler()

### Set render settings
scene = bpy.context.scene
scene.render.film_transparent = True
scene.render.engine = 'CYCLES'
scene.render.image_settings.file_format = 'PNG'
prefs = bpy.context.preferences
prefs.addons["cycles"].preferences.compute_device_type = "CUDA"  # or 'OPTIX' for RTX
prefs.addons["cycles"].preferences.get_devices()
for device in prefs.addons["cycles"].preferences.devices:
    device.use = True
scene.cycles.device = "GPU"
scene.render.resolution_x = 224
scene.render.resolution_y = 224
scene.render.use_freestyle = True
view_layer = bpy.context.view_layer
view_layer.use_freestyle = True
scene.render.line_thickness = 0.66

# Ensure a Freestyle line set exists
freestyle_settings = view_layer.freestyle_settings
if not freestyle_settings.linesets:
    line_set = freestyle_settings.linesets.new(name="LineSet")
else:
    line_set = freestyle_settings.linesets[0]

# Ensure a line style exists
if not line_set.linestyle:
    line_style = bpy.data.linestyles.new(name="LineStyle")
    line_set.linestyle = line_style
else:
    line_style = line_set.linestyle

# Customize line style
line_style.use_chaining = True
line_style.thickness = 0.66
line_style.color = (0, 0, 0)  # black lines
scene.cycles.samples = 512
scene.cycles.max_bounces = 3
scene.cycles.diffuse_bounces = 2
scene.cycles.glossy_bounces = 0
scene.cycles.transmission_bounces = 2
scene.cycles.transparent_max_bounces = 2
scene.cycles.volume_bounces = 1


# Load STL
elevation = float(argv[0])
azimuth = float(argv[1])
distance = float(argv[2])
mesh_path = argv[3]
output_img = argv[4]

bpy.ops.import_mesh.stl(filepath=mesh_path)
# Center object
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
obj.location = (0, 0, 0)

mat = bpy.data.materials.new(name="GrayMaterial")
mat.diffuse_color = (0.20098039, 0.29117647, 0.50882353, 1)  # RGBA gray
mat.use_nodes = False
obj.data.materials.append(mat)

# Render
bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
bbox_center = sum(bbox, mathutils.Vector()) / 8

# Estimate object "radius" from center to furthest corner
max_extent = max((v - bbox_center).length for v in bbox)
# Desired FOV (horizontal) in degrees
fov_deg = 45
sensor_width = 36  # mm, default for Blender full-frame camera
sdistance = 1.6*max_extent / math.tan(math.radians(fov_deg / 2))
print(sdistance,distance)

bpy.ops.object.camera_add()
cam = bpy.context.object
bpy.context.scene.camera = cam
cam.data.lens = 65

cam_location = spherical_to_cartesian(elevation, azimuth, distance)
# Add and orient camera
cam.location=cam_location
target_location = mathutils.Vector((0, 0, 0))
direction = (target_location - cam_location).normalized()
up = mathutils.Vector((0, 1, 0))
right = direction.cross(up).normalized()
true_up = right.cross(direction).normalized()
rot_matrix = mathutils.Matrix((right, true_up, -direction)).transposed()
cam.rotation_euler = rot_matrix.to_euler()

# Output path
scene.render.filepath = f"{output_img}.png"

# Render
bpy.ops.render.render(write_still=True)

    
