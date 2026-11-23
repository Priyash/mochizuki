"""Runs ON Kaggle (or locally with --engine-set stub). One session, one batch.

Reads manifest.json, makes every job it can before the deadline, and writes
results.json after each job, so a session that is cut off still reports what it made.

    python worker.py --manifest manifest.json --inputs DIR --out DIR [--engine-set stub]

Engines
  keyframe / variant : Qwen-Image-Edit-2511 (Apache 2.0), GGUF transformer if available
  take               : Wan 2.2 TI2V-5B or I2V-A14B (Apache 2.0), image-to-video
  upscale            : Real-CUGAN 3x via spandrel, then exact 3840x2160 (cover + centre crop)

This file has not been run on a GPU by its author. The first Kaggle run is the test:
start with the three test shots (see README).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

KIND_TO_ENGINE = {"ref": "refs", "keyframe": "keyframe", "variant": "keyframe", "upscale": "upscale"}


# ---------------------------------------------------------------- shared helpers
def wan_frames(seconds: float, fps: int, max_frames: int) -> int:
    """Wan needs 4k+1 frames. Smallest such count covering the shot, capped by the model."""
    need = max(1, math.ceil(seconds * fps))
    frames = ((need - 1 + 3) // 4) * 4 + 1
    cap = ((max_frames - 1) // 4) * 4 + 1
    return max(5, min(frames, cap))


def wan_segments(seconds: float, fps: int, max_frames: int) -> list[int]:
    """Frame counts of the clips that cover a shot. Each is 4k+1 frames and at most max_frames.

    Long shots are chained: each next clip starts from the previous clip's last frame,
    so it shares one frame with it. Total new frames >= the shot's length.
    """
    need = max(1, math.ceil(seconds * fps))
    cap = ((max_frames - 1) // 4) * 4 + 1
    first = wan_frames(seconds, fps, max_frames)
    segs, have = [first], first
    while have < need:
        nxt = wan_frames((need - have + 1) / fps, fps, max_frames)
        segs.append(nxt)
        have += nxt - 1
    return segs


def cover_crop_box(src_w: int, src_h: int, dst_w: int, dst_h: int) -> tuple[int, int, int, int]:
    """Scale to cover dst exactly, then centre-crop. Returns (scaled_w, scaled_h, left, top)."""
    scale = max(dst_w / src_w, dst_h / src_h)
    sw, sh = max(dst_w, round(src_w * scale)), max(dst_h, round(src_h * scale))
    return sw, sh, (sw - dst_w) // 2, (sh - dst_h) // 2


def seed_for(key: str) -> int:
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg  # noqa: PLC0415
    return imageio_ffmpeg.get_ffmpeg_exe()


def log(msg: str) -> None:
    print(f"[worker {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- stub engines (CPU, for tests)
class StubImage:
    """Solid images at the configured size. Tests the contract, not the art."""

    def __init__(self, cfg: dict):
        self.w, self.h = cfg.get("width", 1664), cfg.get("height", 928)

    def run(self, job: dict, inputs: Path, out: Path) -> dict:
        for ref in job.get("refs", []) + ([job["image"]] if job.get("image") else []):
            if not (inputs / ref).exists():
                raise FileNotFoundError(f"input {ref} missing from the batch")
        w, h = job.get("width", self.w), job.get("height", self.h)
        subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                        f"color=c=0x4a5568:s={w}x{h}", "-frames:v", "1", str(out)], check=True)
        return {"width": w, "height": h}


class StubVideo:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def run(self, job: dict, inputs: Path, out: Path) -> dict:
        c = self.cfg
        segs = wan_segments(job["seconds"], c["fps"], c["max_frames"])
        frames = sum(segs) - (len(segs) - 1)
        img = inputs / job["image"]
        subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-y", "-loop", "1", "-i", str(img),
                        "-frames:v", str(frames), "-r", str(c["fps"]),
                        "-vf", f"scale={c['width']}:{c['height']},format=yuv420p", str(out)], check=True)
        return {"frames": frames, "segments": segs, "fps": c["fps"], "width": c["width"], "height": c["height"]}


class StubUpscale:
    def __init__(self, cfg: dict):
        self.w, self.h = cfg["target_width"], cfg["target_height"]

    def run(self, job: dict, inputs: Path, out: Path) -> dict:
        src = inputs / job["source"]
        vf = f"scale={self.w}:{self.h}:force_original_aspect_ratio=increase,crop={self.w}:{self.h}"
        if job["media"] == "video":
            args = ["-i", str(src), "-vf", vf + ",format=yuv420p", "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "30"]
        else:
            args = ["-i", str(src), "-vf", vf, "-frames:v", "1"]
        subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-y", *args, str(out)], check=True)
        return {"width": self.w, "height": self.h}


# ---------------------------------------------------------------- GPU engines
def pick_dtype(name: str):
    import torch  # noqa: PLC0415
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16          # T4 (Turing) has no native bf16


def find_gguf(repo: str, subdir: str, prefer: list[str]) -> str | None:
    """Download the first GGUF in `prefer` order that the repo has. None if the repo has none."""
    from huggingface_hub import hf_hub_download, list_repo_files  # noqa: PLC0415
    try:
        files = list_repo_files(repo)
    except Exception as e:  # noqa: BLE001
        log(f"GGUF repo {repo} not reachable ({e}); falling back")
        return None
    for q in prefer:
        for f in files:
            if f.endswith(".gguf") and q in f and (not subdir or f.startswith(subdir + "/")):
                log(f"using {repo}/{f}")
                return hf_hub_download(repo, f)
    log(f"no GGUF matching {prefer} in {repo}/{subdir}")
    return None


def snapshot(model: str, skip: list[str]) -> str:
    """Download a diffusers repo, leaving out the weights of the `skip` folders (they come from a
    GGUF instead). Their config.json is still needed to build the model around the GGUF."""
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    ignore = [f"{d}/*{ext}" for d in skip for ext in (".safetensors", ".bin", ".pt", ".pth", ".gguf")]
    return snapshot_download(model, ignore_patterns=ignore)


def check_not_blank(arr, what: str) -> None:
    import numpy as np  # noqa: PLC0415
    a = np.asarray(arr, dtype="float32")
    if not np.isfinite(a).all() or float(a.max()) - float(a.min()) < 0.02:
        raise RuntimeError(f"{what} came out blank or NaN; this GPU may not handle the chosen dtype. "
                           "Try engines.dtype = 'fp32' for this stage, or a smaller quantisation.")


class QwenKeyframes:
    """Qwen-Image-Edit-2511: reference stills in, a new frame out. 1 to 3 references."""

    PIPELINE = "QwenImageEditPlusPipeline"

    def __init__(self, cfg: dict, dtype_name: str):
        import diffusers  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from diffusers import (GGUFQuantizationConfig, PipelineQuantizationConfig,  # noqa: PLC0415
                               QwenImageTransformer2DModel)
        self.cfg, self.torch = cfg, torch
        dtype = pick_dtype(dtype_name)
        g = cfg.get("transformer_gguf") or {}
        gguf = find_gguf(g["repo"], g.get("dir", ""), g.get("prefer", [])) if g else None
        kwargs = {"torch_dtype": dtype}
        quantize = ["text_encoder"]
        if gguf:
            local = snapshot(cfg["model"], skip=["transformer"])
            kwargs["transformer"] = QwenImageTransformer2DModel.from_single_file(
                gguf, quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                config=local, subfolder="transformer", torch_dtype=dtype)
        else:
            local = snapshot(cfg["model"], skip=[])
            quantize.append("transformer")
        kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": dtype},
            components_to_quantize=quantize)
        self.pipe = getattr(diffusers, self.PIPELINE).from_pretrained(local, **kwargs)
        self.pipe.enable_model_cpu_offload()
        self.pipe.set_progress_bar_config(disable=True)

    def run(self, job: dict, inputs: Path, out: Path) -> dict:
        from PIL import Image  # noqa: PLC0415
        c = self.cfg
        if job.get("image"):                                   # a mouth-shape variant edits the keyframe
            images = [Image.open(inputs / job["image"]).convert("RGB")]
        else:
            images = [Image.open(inputs / r).convert("RGB") for r in job.get("refs", [])][: c.get("max_refs", 3)]
        if not images:
            raise RuntimeError("no reference stills for this shot: add them to style_bible/refs (gate 1)")
        with self.torch.inference_mode():
            result = self.pipe(image=images, prompt=job["prompt"], negative_prompt=c.get("negative_prompt", " "),
                               true_cfg_scale=c.get("true_cfg_scale", 4.0), guidance_scale=1.0,
                               num_inference_steps=c.get("steps", 40), width=c["width"], height=c["height"],
                               generator=self.torch.Generator("cpu").manual_seed(seed_for(job["key"])))
        img = result.images[0]
        check_not_blank(img, "keyframe")
        img.save(out)
        return {"width": img.width, "height": img.height, "seed": seed_for(job["key"])}


class QwenStills(QwenKeyframes):
    """Qwen-Image-2512 text-to-image: the candidate reference stills for gate 1."""

    PIPELINE = "QwenImagePipeline"

    def run(self, job: dict, inputs: Path, out: Path) -> dict:
        c = self.cfg
        w, h = job.get("width", 1664), job.get("height", 928)
        with self.torch.inference_mode():
            result = self.pipe(prompt=job["prompt"], negative_prompt=c.get("negative_prompt", " "),
                               true_cfg_scale=c.get("true_cfg_scale", 4.0), num_inference_steps=c.get("steps", 50),
                               width=w, height=h,
                               generator=self.torch.Generator("cpu").manual_seed(seed_for(job["key"])))
        img = result.images[0]
        check_not_blank(img, "reference still")
        img.save(out)
        return {"width": img.width, "height": img.height, "seed": seed_for(job["key"])}


class WanVideo:
    """Wan 2.2 image-to-video. Prompts are encoded first on the CPU, then the encoder is freed,
    so the text encoder and both 14B experts never sit in RAM together."""

    def __init__(self, cfg: dict, dtype_name: str, prompts: list[str], negative: str):
        import torch  # noqa: PLC0415
        from diffusers import (AutoencoderKLWan, GGUFQuantizationConfig,  # noqa: PLC0415
                               WanImageToVideoPipeline, WanTransformer3DModel)
        from transformers import AutoTokenizer, UMT5EncoderModel  # noqa: PLC0415
        self.cfg, self.torch = cfg, torch
        dtype = pick_dtype(dtype_name)
        g = cfg.get("transformer_gguf")
        skip = ["transformer", "transformer_2"] if g else []
        local = snapshot(cfg["model"], skip=skip)

        # 1. text encoding on the CPU in bf16 (T5-class encoders overflow in fp16)
        tok = AutoTokenizer.from_pretrained(local, subfolder="tokenizer")
        enc = UMT5EncoderModel.from_pretrained(local, subfolder="text_encoder", torch_dtype=torch.bfloat16)
        self.embeds = {}
        for text in list(dict.fromkeys(prompts + [negative])):
            self.embeds[text] = self._encode(tok, enc, text)
        del enc
        gc.collect()

        # 2. the video pipeline without a text encoder
        kwargs = {"torch_dtype": dtype, "text_encoder": None, "tokenizer": None,
                  "vae": AutoencoderKLWan.from_pretrained(local, subfolder="vae", torch_dtype=torch.float32)}
        if g:
            q = GGUFQuantizationConfig(compute_dtype=dtype)
            hi = find_gguf(g["repo"], g["high_noise_dir"], g["prefer"])
            lo = find_gguf(g["repo"], g["low_noise_dir"], g["prefer"])
            if not hi or not lo:
                raise RuntimeError(f"GGUF weights not found in {g['repo']}")
            kwargs["transformer"] = WanTransformer3DModel.from_single_file(
                hi, quantization_config=q, config=local, subfolder="transformer", torch_dtype=dtype)
            kwargs["transformer_2"] = WanTransformer3DModel.from_single_file(
                lo, quantization_config=q, config=local, subfolder="transformer_2", torch_dtype=dtype)
        if cfg.get("expand_timesteps"):
            kwargs["expand_timesteps"] = True
        self.pipe = WanImageToVideoPipeline.from_pretrained(local, **kwargs)
        self.pipe.enable_model_cpu_offload()
        try:
            self.pipe.vae.enable_tiling()
        except Exception:  # noqa: BLE001 - tiling is an optimisation only
            pass
        self.pipe.set_progress_bar_config(disable=True)
        self.negative = negative

    def _encode(self, tok, enc, text: str):
        """Same as diffusers' Wan _get_t5_prompt_embeds, on the CPU."""
        torch = self.torch
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean  # noqa: PLC0415
        t = tok([prompt_clean(text)], padding="max_length", max_length=512, truncation=True,
                add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
        with torch.inference_mode():
            h = enc(t.input_ids, t.attention_mask).last_hidden_state
        n = int(t.attention_mask.gt(0).sum())
        e = torch.zeros_like(h)
        e[:, :n] = h[:, :n]
        return e

    def run(self, job: dict, inputs: Path, out: Path) -> dict:
        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        torch, c = self.torch, self.cfg
        segs = wan_segments(job["seconds"], c["fps"], c["max_frames"])
        img = Image.open(inputs / job["image"]).convert("RGB")
        sw, sh, left, top = cover_crop_box(img.width, img.height, c["width"], c["height"])
        img = img.resize((sw, sh), Image.LANCZOS).crop((left, top, left + c["width"], top + c["height"]))
        clips = []
        for n, frames in enumerate(segs):
            args = dict(image=img, prompt_embeds=self.embeds[job["prompt"]],
                        negative_prompt_embeds=self.embeds[self.negative],
                        height=c["height"], width=c["width"], num_frames=frames,
                        num_inference_steps=c.get("steps", 40), guidance_scale=c.get("guidance_scale", 5.0),
                        generator=torch.Generator("cpu").manual_seed(seed_for(job["key"]) + n),
                        output_type="np")
            if "guidance_scale_2" in c:
                args["guidance_scale_2"] = c["guidance_scale_2"]
            with torch.inference_mode():
                video = self.pipe(**args).frames[0]        # (frames, h, w, 3) floats 0..1
            check_not_blank(video, "video")
            video = (np.clip(video, 0, 1) * 255).round().astype("uint8")
            clips.append(video if n == 0 else video[1:])  # the next clip starts on the last frame
            img = Image.fromarray(video[-1])
        full = np.concatenate(clips, axis=0)
        write_video(full, out, c["fps"], crf=10)
        return {"frames": int(full.shape[0]), "segments": segs, "fps": c["fps"], "width": c["width"],
                "height": c["height"], "seed": seed_for(job["key"])}


def write_video(frames, out: Path, fps: int, crf: int) -> None:
    """uint8 frames (n, h, w, 3) to H.264 at high quality."""
    h, w = frames.shape[1:3]
    p = subprocess.Popen([ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                          "-crf", str(crf), "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed to write the video")


class Upscale4K:
    """Upscale, then an exact cover-resize and centre crop to 3840x2160.

    Takes use a video-stable model so frames do not flicker; stills use a sharper
    still-image model. Each is loaded the first time it is needed.
    """

    def __init__(self, cfg: dict, dtype_name: str):
        import torch  # noqa: PLC0415
        self.cfg, self.torch, self.dtype_name = cfg, torch, dtype_name
        self.models: dict[str, tuple] = {}
        self.model, self.half = None, False

    def _load(self, media: str):
        if media in self.models:
            self.model, self.half = self.models[media]
            return
        torch = self.torch
        from spandrel import ImageModelDescriptor, ModelLoader  # noqa: PLC0415
        w = self.cfg[f"{media}_weights"]
        model = ModelLoader().load_from_file(self._weights(w))
        if not isinstance(model, ImageModelDescriptor):
            raise RuntimeError("upscale weights are not an image-to-image model")
        half = bool(model.supports_half) and pick_dtype(self.dtype_name) == torch.float16
        model.cuda().eval()
        if half:
            model.half()
        log(f"{media} upscaler {model.architecture} x{model.scale}, half={half}")
        self.models[media] = (model, half)
        self.model, self.half = model, half

    @staticmethod
    def _weights(w: dict) -> str:
        import urllib.request  # noqa: PLC0415

        def fetch(url: str) -> str:
            dest = Path("/tmp") / Path(url).name
            if not dest.exists():
                urllib.request.urlretrieve(url, dest)
            return str(dest)

        if w.get("repo"):
            try:
                from huggingface_hub import hf_hub_download  # noqa: PLC0415
                return hf_hub_download(w["repo"], w["file"], repo_type=w.get("repo_type", "model"))
            except Exception as e:  # noqa: BLE001
                if not w.get("fallback_url"):
                    raise
                log(f"weights {w['repo']} unavailable ({e}); using {w['fallback_url']}")
                return fetch(w["fallback_url"])
        return fetch(w["url"])

    def _frame(self, rgb):
        """uint8 HxWx3 -> uint8 3840x2160x3."""
        torch = self.torch
        import torch.nn.functional as F  # noqa: PLC0415
        tw, th = self.cfg["target_width"], self.cfg["target_height"]
        x = torch.from_numpy(rgb).cuda().permute(2, 0, 1)[None].float() / 255.0
        if self.half:
            x = x.half()
        with torch.inference_mode():
            y = self.model(x).float().clamp(0, 1)
            sw, sh, left, top = cover_crop_box(y.shape[3], y.shape[2], tw, th)
            if (sw, sh) != (y.shape[3], y.shape[2]):
                y = F.interpolate(y, size=(sh, sw), mode="bicubic", align_corners=False, antialias=True)
            y = y[:, :, top:top + th, left:left + tw].clamp(0, 1)
        return (y[0].permute(1, 2, 0) * 255).round().byte().cpu().numpy()

    def run(self, job: dict, inputs: Path, out: Path) -> dict:
        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        src = inputs / job["source"]
        tw, th = self.cfg["target_width"], self.cfg["target_height"]
        self._load(job["media"])
        if job["media"] == "image":
            Image.fromarray(self._frame(np.asarray(Image.open(src).convert("RGB")))).save(out)
            return {"width": tw, "height": th}
        import imageio_ffmpeg  # noqa: PLC0415
        reader = imageio_ffmpeg.read_frames(str(src), pix_fmt="rgb24")
        meta = next(reader)
        w, h = meta["size"]
        fps = meta.get("fps") or 24
        writer = subprocess.Popen([ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "rawvideo",
                                   "-pix_fmt", "rgb24", "-s", f"{tw}x{th}", "-r", str(fps), "-i", "-",
                                   "-c:v", "libx264", "-preset", "veryfast", "-crf", str(self.cfg.get("crf", 12)),
                                   "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
        n = 0
        for raw in reader:
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
            writer.stdin.write(self._frame(frame).tobytes())
            n += 1
        writer.stdin.close()
        if writer.wait() != 0:
            raise RuntimeError("ffmpeg failed to write the 4K video")
        return {"width": tw, "height": th, "frames": n, "fps": fps}


# ---------------------------------------------------------------- the run
def build_engine(name: str, jobs: list[dict], engines: dict, engine_set: str):
    dtype = engines.get("dtype", "auto")
    if name == "refs":
        cfg = engines["refs"]
        return StubImage(cfg) if engine_set == "stub" else QwenStills(cfg, dtype)
    if name == "keyframe":
        cfg = engines["keyframe"]
        return StubImage(cfg) if engine_set == "stub" else QwenKeyframes(cfg, dtype)
    if name == "upscale":
        cfg = engines["upscale"]
        return StubUpscale(cfg) if engine_set == "stub" else Upscale4K(cfg, dtype)
    cfg = engines[name]
    if engine_set == "stub":
        return StubVideo(cfg)
    return WanVideo(cfg, dtype, [j["prompt"] for j in jobs], engines.get("negative_prompt", ""))


def free_engine(engine) -> None:
    del engine
    gc.collect()
    try:
        import torch  # noqa: PLC0415
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def run(manifest: Path, inputs: Path, out: Path, engine_set: str, expect_batch: str | None = None) -> dict:
    t0 = time.time()
    m = json.loads(manifest.read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    results = {"batch_id": m["batch_id"], "engine_set": engine_set, "started": t0, "jobs": {},
               "groups": {}, "error": None}

    def save():
        results["elapsed"] = time.time() - t0
        (out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    if expect_batch and m["batch_id"] != expect_batch:
        results["error"] = f"stale inputs: kernel expected {expect_batch}, dataset has {m['batch_id']}"
        save()
        return results
    deadline = t0 + float(m.get("deadline_seconds", 10 * 3600))
    groups: dict[str, list[dict]] = {}
    for job in m["jobs"]:
        name = "keyframe" if job["kind"] == "ref" and job.get("refs") else KIND_TO_ENGINE.get(job["kind"], job["model"])
        groups.setdefault(name, []).append(job)
    for name, jobs in groups.items():
        if time.time() + 600 > deadline:
            break
        g0 = time.time()
        try:
            log(f"loading engine {name} for {len(jobs)} jobs")
            engine = build_engine(name, jobs, m["engines"], engine_set)
        except Exception as e:  # noqa: BLE001
            log(f"engine {name} failed to load: {e!r}"[:600])
            log(traceback.format_exc()[-1500:])
            for job in jobs:
                results["jobs"][job["key"]] = {"ok": False, "error": f"engine {name} failed to load: {e!r}"[:800]}
            results["groups"][name] = {"load_error": traceback.format_exc()[-2000:]}
            save()
            continue
        results["groups"][name] = {"load_seconds": time.time() - g0}
        for job in jobs:
            est = float(job.get("est_seconds", 0))
            if time.time() + est > deadline:
                continue                                   # left for the next session
            j0 = time.time()
            ext = ".mp4" if job["kind"] == "take" or job.get("media") == "video" else ".png"
            target = out / f"{job['key']}{ext}"
            try:
                info = engine.run(job, inputs, target)
                results["jobs"][job["key"]] = {"ok": True, "file": target.name,
                                               "gpu_seconds": time.time() - j0, **(info or {})}
            except Exception as e:  # noqa: BLE001
                oom = "out of memory" in str(e).lower()
                results["jobs"][job["key"]] = {"ok": False, "error": repr(e)[:800], "oom": oom,
                                               "gpu_seconds": time.time() - j0}
                log(f"{job['key']} failed: {e!r}")
                if oom:
                    free_engine(None)
            save()
        free_engine(engine)
        engine = None
    save()
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--inputs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--engine-set", default="gpu", choices=["gpu", "stub"])
    ap.add_argument("--expect-batch")
    a = ap.parse_args(argv)
    r = run(a.manifest, a.inputs, a.out, a.engine_set, a.expect_batch)
    ok = sum(1 for v in r["jobs"].values() if v.get("ok"))
    log(f"done: {ok}/{len(r['jobs'])} jobs ok in {r['elapsed'] / 60:.1f} min")
    return 0 if r.get("error") is None else 3


if __name__ == "__main__":
    sys.exit(main())
