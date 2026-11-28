"""Acting: how the characters perform in a shot.

    direction   the acting sheet: camera, beats, gaze, restraint, timing; drafted from the shot sheet
                and the film's style_bible/acting.json, approved by the owner
    timing      anime timing: key poses, on twos and threes, holds, impacts, blinks
    capture     a face performance from the iPhone (Live Link Face CSV) or a video (MediaPipe)
    retarget    ARKit's 52 face shapes -> the character's shapes, plus the anime layer
    arc         the emotional arc: feelings on the circumplex, against the film's intended arc
                (style_bible/arc.json); drafted shots with no feeling take theirs from it
    culture     the culture pack: display rules (style_bible/culture.json) played into every sheet,
                never over a field the hand-written sheet gives
"""
