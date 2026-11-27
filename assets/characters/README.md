# Character models

The 3D pipeline (`docs/cgi-pipeline.md`) looks here for each character's model, by chapter
(`config.json` → `cgi.characters`, defaults in `pipeline/cgi/config.py`):

| File | Who | Chapters |
| --- | --- | --- |
| `tsumugi-07.vrm` | Tsumugi, 7 | 1 |
| `tsumugi-08.vrm` | Tsumugi, 8 | 2 |
| `tsumugi-27.vrm` | Tsumugi, 27 | 3 |
| `tsumugi-87.vrm` | Tsumugi, 87 | 4, 5 |
| `sayo.vrm` | Sayo | 1, 2 |
| `sayo-very-old.vrm` | Sayo, very old | 3 |

Akari is drawn by code (a paper lantern that acts with its light), and so are the props.

Make them in VRoid Studio (free, on the Mac): File → Export → VRM. Keep VRoid's own face shapes
(the `Fcl_...` shape keys); the pipeline drives them directly. Make Tsumugi once and age her by
changing the body and face sliders of a copy, so she stays recognisably the same person.

A shot whose model is missing still renders, with a stand-in maquette that has the right height,
posture and a simple face that acts, so layout, timing and emotion can be judged first.

`.vrm` and `.blend` files are **not committed** (see `.gitignore`): this repository is public and the
models are your own work. Back them up somewhere private.
