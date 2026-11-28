"""The 3D CG anime pipeline: performance -> rig -> toon render -> (optional) AI finish.

Every shot is built from the same character models, props and sets, so it is coherent
by construction, and it is rendered by Blender's real-time engine (EEVEE), so a whole
chapter renders natively in 4K in hours on the Mac instead of days of video-model takes.
Emotion is directed, not hoped for: each shot carries an acting sheet (beats, gaze,
restraint, timing) that drives the faces, and a captured human performance can replace
or enrich it. See docs/cgi-pipeline.md.

Modules, in the order a shot flows through them (all plain Python, no Blender needed):

    direction   the acting sheet of a shot: camera, beats, timing; drafted from shots.csv
                and style_bible/acting.json, then approved by the owner
    layout      where the camera goes: framing, lens, angle and moves, in metres
    timing      anime timing: key poses (pose-to-pose), on twos/threes, holds, impacts, blinks
    capture     a face performance from the iPhone (Live Link Face CSV) or a video (MediaPipe)
    retarget    ARKit's 52 face shapes -> the character's shapes, plus the anime layer
                (restraint, wet eyes, blush) driven by the acting sheet
    palette     the light rig of a scene or chapter (style_bible/lights.json)
    plan        everything above folded into one JSON file per shot: the only thing Blender reads
    render      runs Blender on a plan, checks the frames, encodes the shot
    cli         python -m engine cgi ...

The Blender side lives in cgi/blender/ and runs inside Blender (or a Python with the bpy
module). It only executes a plan: props, sets, characters, lights, camera, toon shading,
outlines, render. Every decision is made here, where it can be tested.
"""
