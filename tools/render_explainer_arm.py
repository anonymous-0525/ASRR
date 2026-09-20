"""Render original orthographic robot components for the 2D web explainer.

Run from the repository root: blender --background --python tools/render_explainer_arm.py
"""

import math
from pathlib import Path

import bpy
from mathutils import Vector


OUTPUT = Path(__file__).resolve().parents[1] / "docs/static/images/explainer/arm"


def material(name, color, metallic=0.0, roughness=0.35):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    return mat


def finish(obj, name, mat, bevel=0.04):
    obj.name = name
    obj.data.materials.append(mat)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel:
        modifier = obj.modifiers.new("Machined edges", "BEVEL")
        modifier.width = bevel
        modifier.segments = 4
        obj.modifiers.new("Face normals", "WEIGHTED_NORMAL")
    return obj


def box(name, center, size, mat, bevel=0.04):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj = bpy.context.object
    obj.scale = size
    return finish(obj, name, mat, bevel)


def disc(name, center, radius, depth, mat):
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=radius, depth=depth, location=center)
    return finish(bpy.context.object, name, mat, 0.015)


def clear_parts():
    for obj in list(bpy.data.objects):
        if obj.type == "MESH":
            bpy.data.objects.remove(obj, do_unlink=True)


def render(name, center, width, resolution):
    camera.location = (*center, 8)
    camera.data.ortho_scale = width
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.filepath = str(OUTPUT / f"{name}.png")
    bpy.ops.render.render(write_still=True)


bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
OUTPUT.mkdir(parents=True, exist_ok=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 24
scene.cycles.use_denoising = True
scene.render.film_transparent = True
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.resolution_percentage = 100
scene.view_settings.view_transform = "Standard"
scene.world.color = (0.35, 0.35, 0.35)
white = material("Pearl alloy", (0.72, 0.78, 0.8), 0.35)
edge = material("Brushed aluminum", (0.4, 0.48, 0.52), 0.72)
dark = material("Graphite joints", (0.035, 0.055, 0.071), 0.25)
rubber = material("Finger pads", (0.015, 0.023, 0.03), 0.05, 0.65)

bpy.ops.object.camera_add(location=(0, 0, 8))
camera = bpy.context.object
camera.data.type = "ORTHO"
camera.rotation_euler = (0, 0, 0)
scene.camera = camera
for position, energy, size in [((-2, 4, 6), 450, 5), ((4, -1, 5), 220, 4)]:
    bpy.ops.object.light_add(type="AREA", location=position)
    light = bpy.context.object
    light.data.energy = energy
    light.data.shape = "DISK"
    light.data.size = size
    light.rotation_euler = (Vector((1, 0, 0)) - light.location).to_track_quat("-Z", "Y").to_euler()

for name, thickness in [("upper", 0.38), ("forearm", 0.29)]:
    clear_parts()
    box("Rear housing", (1.2, 0, 0), (2.43, thickness, 0.18), dark, 0.12)
    box("Alloy cover", (1.2, 0, 0.12), (2.2, thickness - 0.04, 0.17), white, 0.10)
    box("Service seam", (1.2, -thickness * 0.23, 0.216), (1.64, 0.014, 0.018), edge, 0.005)
    box("End cuff", (2.13, 0, 0.20), (0.095, thickness - 0.055, 0.08), edge, 0.02)
    for x in (0.22, 2.02):
        disc("Fastener", (x, thickness * 0.22, 0.216), 0.018, 0.016, dark)
    render(name, (1.2, 0), 3.2, (960, 200))

clear_parts()
disc("Bearing housing", (0, 0, 0), 0.29, 0.2, dark)
disc("Bearing rim", (0, 0, 0.13), 0.235, 0.1, edge)
disc("Servo face", (0, 0, 0.20), 0.183, 0.07, white)
disc("Axle", (0, 0, 0.25), 0.07, 0.04, dark)
for angle in (45, 135, 225, 315):
    a = math.radians(angle)
    disc("Face bolt", (0.15 * math.cos(a), 0.15 * math.sin(a), 0.243), 0.014, 0.014, dark)
render("joint", (0, 0), 0.72, (216, 216))

clear_parts()
box("Mounting plate", (0, -0.59, 0), (1.1, 0.2, 0.2), dark, 0.055)
box("Pedestal", (0, -0.30, 0.05), (0.64, 0.55, 0.26), white, 0.07)
box("Pedestal front inset", (0, -0.34, 0.195), (0.39, 0.30, 0.035), edge, 0.035)
for x in (-0.42, 0.42):
    disc("Mount bolt", (x, -0.58, 0.125), 0.042, 0.04, edge)
render("base", (0, -0.25), 1.6, (384, 360))

clear_parts()
box("Wrist adapter", (0, 0, 0), (0.31, 0.24, 0.14), edge)
box("Parallel gripper", (0, -0.16, 0.02), (0.59, 0.25, 0.21), dark, 0.055)
box("Gripper cover", (0, -0.15, 0.14), (0.40, 0.15, 0.05), white, 0.03)
for x in (-0.21, 0.21):
    box("Finger", (x, -0.39, 0.04), (0.105, 0.30, 0.15), edge, 0.022)
    box("Finger pad", (x * 0.75, -0.50, 0.14), (0.07, 0.14, 0.035), rubber, 0.016)
render("gripper", (0, -0.25), 0.95, (285, 270))
