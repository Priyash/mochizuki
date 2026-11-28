"""The Blender side of the 3D pipeline: it builds and renders a plan and decides nothing.

Runs inside Blender 5.x (or a Python with the bpy module). Nothing here is imported by the
rest of the pipeline: engine/cgi/render.py starts Blender on blender/build.py.
"""
