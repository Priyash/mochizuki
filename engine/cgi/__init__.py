"""The 3D CG anime pipeline: performance -> rig -> toon render -> (optional) AI finish.

Every shot is built from the same character models, props and sets, so it is coherent
by construction, and it is rendered by Blender's real-time engine (EEVEE), so a whole
chapter renders natively in 4K in hours on the Mac instead of days of video-model takes.
Emotion is directed, not hoped for: each shot carries an acting sheet (beats, gaze,
restraint, timing) that drives the faces, and a captured human performance can replace
or enrich it. See docs/cgi-pipeline.md.

Sub-components, in the order a shot flows through them (all plain Python, no Blender needed
except render/blender):
    acting/     how the characters perform: direction (the acting sheet), arc (the intended feeling),
                culture (display rules), timing, capture, retarget
    staging/    what the camera sees: layout (camera), palette (lights), figure (characters from code)
    plan        everything above folded into one JSON file per shot: the only thing Blender reads
    render/     runner (runs Blender on a plan, checks the frames, encodes the shot) and blender/, which
                runs inside Blender and only executes a plan: props, sets, characters, lights, camera,
                toon shading, outlines. Every decision is made before it, where it can be tested.
    continuity  the continuity supervisor: every cut checked from the plans (the 180 and 30 degree rules,
                eyelines, light, props, positions)
    auto        a chapter, unattended: voices (engine/audio), renders and dubs, board, cuts, review sheet
    cli         python -m engine cgi ...
    config      settings: the engine's defaults, overridden by the film's config.json -> cgi
"""
