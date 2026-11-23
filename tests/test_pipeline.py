"""Tests run on a copy of the project. Nothing calls Kaggle, fal or a GPU, and nothing is spent.

The sync tests use the mock provider. The end-to-end test runs the real worker script
through LocalKaggle with stub engines, so the batch contract and the 4K geometry are
exercised; the GPU models themselves are not (see README, "What is tested").
"""
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from pipeline import assemble as asm
from pipeline import kaggle as kg
from pipeline import plan as planner
from pipeline import providers, steps
from pipeline.project import Project, wan_frames, wan_segments

REPO = Path(__file__).resolve().parents[1]
HAS_FFMPEG = shutil.which("ffmpeg") is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")


def write_png(path: Path, w: int = 64, h: int = 36) -> None:
    import struct
    import zlib
    raw = b"".join(b"\x00" + bytes([90, 110, 160]) * w for _ in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))  # noqa: E731
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def load_worker():
    spec = importlib.util.spec_from_file_location("worker", REPO / "worker" / "worker.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def copy_project(tmp_path, csv_edit=None, cfg_edit=None) -> Project:
    shutil.copy(REPO / "config.json", tmp_path / "config.json")
    shutil.copytree(REPO / "style_bible", tmp_path / "style_bible")
    for r in json.loads((REPO / "style_bible" / "ref_prompts.json").read_text())["refs"]:
        write_png(tmp_path / "style_bible" / "refs" / f"{r['name']}.png")   # stand-in reference stills
    text = (REPO / "shots.csv").read_text(encoding="utf-8")
    if csv_edit:
        text = csv_edit(text)
    (tmp_path / "shots.csv").write_text(text, encoding="utf-8")
    cfg = json.loads((tmp_path / "config.json").read_text())
    cfg["kaggle"]["username"] = "tester"
    if cfg_edit:
        cfg_edit(cfg)
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return Project(tmp_path)


@pytest.fixture
def proj(tmp_path):
    return copy_project(tmp_path)


def sync(proj):
    led = steps.open_ledger(proj)
    return led, steps.SyncExecutor(providers.get("mock", proj))


def board(proj, led, ex, chapter=1):
    steps.keyframes(proj, led, ex, chapter)
    for s in proj.chapter(chapter):
        led.select(led.jobs(s.id, "keyframe")[0]["key"])
    led.approve(chapter, "board", "human")


def replace_row(text, shot_id, old, new):
    lines = text.splitlines()
    i = next(n for n, line in enumerate(lines) if line.startswith(shot_id + ","))
    assert old in lines[i]
    lines[i] = lines[i].replace(old, new)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- the plan
def test_film_is_ten_minutes_and_chapters_hit_their_targets(proj):
    assert sum(s.seconds for s in proj.shots) == 600
    per = {c: sum(s.seconds for s in proj.chapter(c)) for c in range(1, 6)}
    assert per == {1: 120, 2: 80, 3: 100, 4: 170, 5: 130}


def test_free_plan_costs_no_money_and_reports_gpu_time(proj):
    pl = planner.plan(proj)
    assert pl["shots"] == 138
    assert pl["best"] == 0 and pl["worst"] == 0
    assert 0 < pl["gpu_hours_best"] < pl["gpu_hours_worst"]
    per = [planner.plan(proj, c) for c in range(1, 6)]
    assert sum(x["gpu_hours_worst"] for x in per) == pytest.approx(pl["gpu_hours_worst"], abs=0.3)


def test_every_video_shot_runs_on_kaggle_and_stills_make_no_video(proj):
    for s in proj.shots:
        if s.type == "Still":
            assert s.takes == 0 and not proj.model(s.model).get("generates_video")
        else:
            assert proj.provider_of(s) == "kaggle" and s.takes > 0


def test_frame_counts_follow_wan_rules_and_cover_every_shot():
    for fps, cap in ((16, 81), (24, 121)):
        for seconds in range(1, 13):
            segs = wan_segments(seconds, fps, cap)
            assert all((f - 1) % 4 == 0 and 5 <= f <= cap for f in segs)
            assert sum(segs) - (len(segs) - 1) >= seconds * fps     # chained clips share one frame
            assert segs[0] == wan_frames(seconds, fps, cap)
    assert wan_segments(5, 16, 81) == [81]
    assert len(wan_segments(9, 16, 81)) == 2


def test_worker_and_pipeline_agree_on_frames():
    w = load_worker()
    for fps, cap in ((16, 81), (24, 121)):
        for seconds in (1, 3, 5, 6, 7, 10, 12):
            assert w.wan_frames(seconds, fps, cap) == wan_frames(seconds, fps, cap)
            assert w.wan_segments(seconds, fps, cap) == wan_segments(seconds, fps, cap)


@pytest.mark.parametrize("src", [(1280, 720), (1280, 704), (1664, 928), (5120, 2880), (720, 1280)])
def test_4k_geometry_covers_the_frame_exactly(src):
    w = load_worker()
    sw, sh, left, top = w.cover_crop_box(*src, 3840, 2160)
    assert sw >= 3840 and sh >= 2160
    assert 0 <= left and left + 3840 <= sw and 0 <= top and top + 2160 <= sh
    assert abs(sw / sh - src[0] / src[1]) < 0.01                # no stretching


# ---------------------------------------------------------------- keyframes and gates
def test_keyframes_are_not_made_twice(proj):
    led, ex = sync(proj)
    first = steps.keyframes(proj, led, ex, 1)
    second = steps.keyframes(proj, led, ex, 1)
    assert first["made"] == 27 * 3 and second["made"] == 0
    assert len(second["waiting_for_verdict"]) == 27


def test_second_keyframe_batch_only_after_every_candidate_fails(proj):
    led, ex = sync(proj)
    s = proj.chapter(1)[1]
    steps.keyframes(proj, led, ex, 1, only=s.id)
    first = led.jobs(s.id, "keyframe")
    led.verdict(first[0]["key"], "fail", "agent")
    assert steps.keyframes(proj, led, ex, 1, only=s.id)["made"] == 0      # two still unjudged
    for j in first[1:]:
        led.verdict(j["key"], "fail", "agent")
    assert steps.state(proj, led, s) == "needs_keyframes"
    assert steps.keyframes(proj, led, ex, 1, only=s.id)["made"] == 3
    for j in led.jobs(s.id, "keyframe"):
        led.verdict(j["key"], "fail", "agent")
    out = steps.keyframes(proj, led, ex, 1, only=s.id)                    # cap of 6 reached
    assert out["made"] == 0 and out["needs_human"] == [s.id]


def test_video_needs_board_approval(proj):
    led, ex = sync(proj)
    steps.keyframes(proj, led, ex, 1)
    with pytest.raises(PermissionError):
        steps.video(proj, led, ex, 1)


def test_upscale_needs_animatic_approval(proj):
    led, ex = sync(proj)
    board(proj, led, ex)
    with pytest.raises(PermissionError):
        steps.upscale(proj, led, ex, 1)


def test_only_a_person_can_approve_a_gate(proj):
    led, _ = sync(proj)
    for gate in ("board", "animatic"):
        with pytest.raises(PermissionError):
            led.approve(1, gate, "agent")


# ---------------------------------------------------------------- takes
def test_one_take_at_a_time_and_stop_after_a_pass(proj):
    led, ex = sync(proj)
    board(proj, led, ex)
    hero = next(s for s in proj.chapter(1) if s.type == "Hero")
    steps.video(proj, led, ex, 1)
    assert len(led.jobs(hero.id, "take")) == 1
    again = steps.video(proj, led, ex, 1)
    assert again["made"] == 0 and hero.id in again["waiting_for_verdict"]
    led.verdict(led.jobs(hero.id, "take")[0]["key"], "pass", "agent")
    steps.video(proj, led, ex, 1)
    assert len(led.jobs(hero.id, "take")) == 1
    assert steps.state(proj, led, hero) == "take_passed"


def test_take_cap_hands_the_shot_to_a_person(proj):
    led, ex = sync(proj)
    board(proj, led, ex)
    hero = next(s for s in proj.chapter(1) if s.type == "Hero")
    for _ in range(hero.takes + 3):
        steps.video(proj, led, ex, 1, only=hero.id)
        for j in led.jobs(hero.id, "take"):
            if j["verdict"] is None:
                led.verdict(j["key"], "fail", "agent")
    assert len(led.jobs(hero.id, "take")) == hero.takes
    assert steps.state(proj, led, hero) == "needs_human"


class Broken:
    def run(self, spec, out):
        raise RuntimeError("model crashed")


def test_repeated_failures_stop_and_ask_a_person(proj):
    led, ex = sync(proj)
    board(proj, led, ex)
    hero = next(s for s in proj.chapter(1) if s.type == "Hero")
    bad = steps.SyncExecutor(Broken())
    for _ in range(5):
        steps.video(proj, led, bad, 1, only=hero.id)
    assert led.failures(hero.id, "take") == proj.config["kaggle"]["max_failures_per_shot"]
    assert steps.state(proj, led, hero) == "needs_human"


def test_lip_flap_is_a_fallback_for_dialogue(tmp_path):
    base = Project(REPO)
    target = next(s for s in base.chapter(1) if s.sound.startswith("Sayo:"))
    assert target.model == "Wan 2.2 14B"                    # dialogue is animated by default
    proj = copy_project(tmp_path, csv_edit=lambda t: replace_row(
        t, target.id, "Hero,Wan 2.2 14B,4", "Still,Editor lip flap,0"))
    led, ex = sync(proj)
    board(proj, led, ex)
    lip = proj.shot(target.id)
    assert steps.state(proj, led, lip) == "needs_variants"
    steps.video(proj, led, ex, 1)
    assert len(led.jobs(lip.id, "variant")) == 3 and not led.jobs(lip.id, "take")
    assert steps.state(proj, led, lip) == "ready_for_editor"


def test_a_cut_off_job_blocks_the_shot_until_resolved(proj):
    led, _ = sync(proj)
    s = proj.chapter(1)[0]
    led.begin("cut.off.key", s.id, s.chapter, "keyframe", 99, s.model, 0)
    assert steps.state(proj, led, s) == "pending_unknown"
    led.resolve("cut.off.key", charged=False)
    assert steps.state(proj, led, s) == "todo"
    assert led.failures(s.id, "keyframe") == 0                # a cut-off is not the model's fault


# ---------------------------------------------------------------- Kaggle batches
def test_batch_respects_the_session_budget_and_defers_the_rest(tmp_path):
    proj = copy_project(tmp_path, cfg_edit=lambda c: c["kaggle"].update(session_hours=1, setup_minutes=5))
    led, ex = sync(proj)
    board(proj, led, ex)
    out = kg.build(proj, led, "video", 1, engine_set="stub")
    assert out["batch"] and out["deferred"] > 0
    assert out["est_hours"] <= 1.0
    manifest = json.loads((proj.media / "batches" / out["batch"] / "dataset" / "manifest.json").read_text())
    assert len(manifest["jobs"]) == out["jobs"]
    for job in manifest["jobs"]:
        assert (proj.media / "batches" / out["batch"] / "dataset" / job["image"]).exists()
    with pytest.raises(kg.KaggleError):                       # one open batch at a time
        kg.build(proj, led, "video", 1, engine_set="stub")


def test_weekly_gpu_cap_stops_new_batches(proj):
    led, _ = sync(proj)
    led.create_batch("old", "video", 1, 28 * 3600)
    led.set_batch("old", status="done", submitted_at=kg.week_start() + 60)
    with pytest.raises(kg.KaggleError, match="used up"):
        kg.build(proj, led, "keyframes", 1, engine_set="stub")


def test_paid_routed_shots_are_left_out_of_kaggle_batches(tmp_path):
    proj = copy_project(tmp_path, csv_edit=lambda t: replace_row(t, "1-04", "Wan 2.2 14B", "Kling 4"))
    led, ex = sync(proj)
    board(proj, led, ex)
    out = kg.build(proj, led, "video", 1, engine_set="stub")
    shots = {j["shot_id"] for j in led.batch_jobs(out["batch"])}
    assert "1-04" not in shots and "1-03" in shots


def test_kaggle_cli_parsing_with_a_fake_runner():
    replies = {
        ("kernels", "status"): (0, 'tester/mochizuki-worker has status "KernelWorkerStatus.COMPLETE"\n'),
        ("quota", "-v"): (0, "resource,used,remaining,total,refreshAt\nGPU,12.5h,17.5h,30h,2026-11-29\n"),
        ("datasets", "status"): (1, "404 not found"),
    }
    calls = []

    def runner(args):
        calls.append(args)
        return replies.get(tuple(args[:2]), (0, ""))

    cli = kg.KaggleCLI(runner)
    assert cli.status("tester/mochizuki-worker") == "complete"
    assert cli.gpu_hours_left() == 17.5
    assert cli.dataset_status("tester/mochizuki-inputs") == "missing"
    cli.publish_dataset(Path("/tmp/x"), "tester/mochizuki-inputs", "b1")
    assert calls[-1][:2] == ["datasets", "create"]            # first publish creates, later ones version
    for text, want in (('has status "running"', "running"), ('has status "ERROR"', "error"),
                       ('has status "cancelAcknowledged"', "cancelled"), ("garbage", "unknown")):
        assert kg.KaggleCLI(lambda a, t=text: (0, t)).status("x") == want


def test_kaggle_week_starts_on_saturday_utc():
    mon = 1795305600 + 86400 + 5 * 3600             # Mon 2026-11-23 05:00 UTC
    start = kg.week_start(mon)
    assert (mon - start) == 2 * 86400 + 5 * 3600     # Sat 2026-11-21 00:00 UTC, as Kaggle's quota reports
    assert kg.week_start(start) == start


# ---------------------------------------------------------------- the whole loop, through 4K
def short_chapter(text):
    keep = ("1-01,", "1-02,", "1-03,", "1-04,", "1-14,")       # still, 5B, 14B, 14B 6 s, 14B 7 s dialogue
    lines = text.splitlines()
    return "\n".join([lines[0]] + [x for x in lines[1:] if x.startswith(keep) or not x.startswith("1-")]) + "\n"


def drive(proj, led, backend, steps_max=30):
    seen = []
    for _ in range(steps_max):
        out = kg.tick(proj, led, backend, 1, engine_set="stub")
        seen.append(out["action"])
        if out["action"] == "idle":
            return out, seen
    raise AssertionError(f"loop did not settle: {seen}")


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height:format=duration", "-of", "json", str(path)],
                       capture_output=True, text=True, check=True)
    d = json.loads(r.stdout)
    return d["streams"][0]["width"], d["streams"][0]["height"], float(d["format"]["duration"])


@needs_ffmpeg
def test_local_kaggle_end_to_end_to_4k(tmp_path):
    proj = copy_project(tmp_path, csv_edit=short_chapter)
    led = steps.open_ledger(proj)
    local = kg.LocalKaggle(proj.media)
    chapter = proj.chapter(1)
    length = sum(s.seconds for s in chapter)

    # keyframes: the loop runs a batch, then waits for a person
    out, seen = drive(proj, led, local)
    assert "built_and_submitted" in seen and out["to_review"] == 3 * len(chapter)
    for s in chapter:
        led.select(led.jobs(s.id, "keyframe")[0]["key"])
    assert drive(proj, led, local)[0]["reason"] == "waiting for a person"   # the board is a human gate
    led.approve(1, "board", "human")

    # takes: fail the first take of 1-03 once, pass everything else
    drive(proj, led, local)
    for item in steps.review_queue(proj, led, 1):
        led.verdict(item["key"], "fail" if item["shot"].startswith("CU small hands") else "pass", "agent")
    drive(proj, led, local)
    for item in steps.review_queue(proj, led, 1):
        led.verdict(item["key"], "pass", "agent")
    assert len(led.jobs("1-03", "take")) == 2
    assert all(steps.state(proj, led, s) in ("take_passed", "ready_for_editor") for s in chapter)

    # the 7 s shot is two chained clips: the take is the full length, not the model's 5 s cap
    take = next(j for j in led.jobs("1-14", "take") if j["verdict"] == "pass")
    assert probe(proj.fetch(take["path"]))[2] >= 7.0

    anim = asm.assemble(proj, led, 1)
    assert probe(anim["file"])[:2] == (640, 360)
    led.approve(1, "animatic", "human")

    # 4K
    drive(proj, led, local)
    assert steps.status(proj, led, 1)["4k"] == {"done": len(chapter)}
    final = asm.assemble(proj, led, 1, four_k=True)
    w, h, seconds = probe(final["file"])
    assert (w, h) == (3840, 2160)
    assert final["from"]["4k"] == len(chapter)
    assert seconds == pytest.approx(length, abs=0.1)


@needs_ffmpeg
def test_a_stale_dataset_is_not_counted_as_a_failure(tmp_path):
    proj = copy_project(tmp_path, csv_edit=short_chapter)
    led = steps.open_ledger(proj)
    local = kg.LocalKaggle(proj.media)
    built = kg.build(proj, led, "keyframes", 1, engine_set="stub")
    manifest = proj.media / "batches" / built["batch"] / "dataset" / "manifest.json"
    m = json.loads(manifest.read_text())
    m["batch_id"] = "someone-elses-batch"              # Kaggle served an older dataset version
    manifest.write_text(json.dumps(m))
    kg.submit(proj, led, local, built["batch"])
    res = kg.collect(proj, led, local, built["batch"])
    assert res["stale"] and res["made"] == 0
    s = proj.chapter(1)[0]
    assert led.failures(s.id, "keyframe") == 0 and steps.state(proj, led, s) == "todo"


# ---------------------------------------------------------------- GitHub Actions producer
class FakeGH:
    """Stands in for `gh release ...`: each release is a folder."""

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def __call__(self, args):
        verb, tag = args[1], args[2]
        d = self.root / tag
        if verb == "view":
            return (0, "\n".join(sorted(f.name for f in d.iterdir()))) if d.exists() else (1, "release not found")
        if verb == "create":
            d.mkdir()
            return 0, ""
        if verb == "upload":
            shutil.copy(args[3], d / Path(args[3]).name)
            return 0, ""
        if verb == "download":
            name, dest = args[args.index("-p") + 1], Path(args[args.index("-D") + 1])
            shutil.copy(d / name, dest / name)
            return 0, ""
        return 1, "unknown"


def test_release_store_puts_each_file_in_its_chapter_release(proj):
    from pipeline.store import tag_for
    m = proj.media
    assert tag_for(proj, m / "keyframes" / "1-03" / "k.png") == "media-ch1"
    assert tag_for(proj, m / "takes" / "4-26" / "t.mp4") == "media-ch4"
    assert tag_for(proj, m / "4k" / "2-01" / "u.mp4") == "4k-ch2"
    assert tag_for(proj, m / "animatic" / "chapter_3_4k.mp4") == "cuts"
    assert tag_for(proj, m / "batches" / "b1" / "x") is None


def test_a_reviewer_cannot_approve_and_only_the_owner_can(proj, tmp_path):
    from pipeline import ci, review
    led, ex = sync(proj)
    steps.keyframes(proj, led, ex, 1)
    s = proj.chapter(1)[0]
    folder = tmp_path / "reviews"
    folder.mkdir()
    (folder / "a.json").write_text(json.dumps({"select": [led.jobs(s.id, "keyframe")[0]["key"]],
                                               "approve": {"gate": "board", "chapter": 1},
                                               "verdicts": [{"key": "nope", "verdict": "pass"}]}))
    out = review.apply(led, folder)
    assert out["applied"] == 1 and out["refused"] and out["errors"] and not list(folder.glob("*.json"))
    assert not led.approved(1, "board")
    assert ci.command(proj, led, tmp_path, "/approve board 1", "claude[bot]", "owner").startswith("Only @owner")
    assert "Not approved" in ci.command(proj, led, tmp_path, "/approve board 1", "owner", "owner")
    assert ci.command(proj, led, tmp_path, "/chapter 3", "owner", "owner").endswith("chapter 3.")
    assert ci.chapter_of(tmp_path) == 3


@needs_ffmpeg
def test_github_producer_loop_with_claude_reviews_and_owner_gates(tmp_path, monkeypatch):
    from pipeline import ci
    from pipeline.store import ReleaseStore
    (tmp_path / "proj").mkdir()
    proj = copy_project(tmp_path / "proj", csv_edit=short_chapter)
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/mochizuki")
    monkeypatch.setenv("GITHUB_REPOSITORY_OWNER", "owner")
    releases = tmp_path / "releases"
    proj.store = ReleaseStore("owner/mochizuki", runner=FakeGH(releases))
    led = steps.open_ledger(proj)
    local = kg.LocalKaggle(proj.media)
    state = tmp_path / "state"

    def run():
        out = ci.step(proj, led, 1, state, local, engine_set="stub")
        for folder in ("keyframes", "takes", "4k"):          # every Actions run starts on a fresh machine
            shutil.rmtree(proj.media / folder, ignore_errors=True)
        return out

    def claude_reviews(n):
        q = json.loads((state / "review" / "ch1" / "queue.json").read_text())
        assert q["items"]
        verdicts, select = [], []
        for it in q["items"]:
            if it["what"] == "keyframe":
                assert (state / "review" / "ch1" / it["candidates"][0]["image"]).exists()
                select.append(it["candidates"][0]["key"])
            else:
                assert (state / "review" / "ch1" / it["image"]).exists()
                verdicts.append({"key": it["key"], "verdict": "pass", "note": "on model"})
        (state / "reviews").mkdir(exist_ok=True)
        (state / "reviews" / f"claude-{n}.json").write_text(json.dumps({"verdicts": verdicts, "select": select}))

    assert run()["notice"]["kind"] == "running"                  # keyframe batch submitted
    assert run()["notice"]["kind"] == "review"                   # collected; Claude's turn
    claude_reviews(1)
    out = run()
    assert out["notice"]["kind"] == "board" and "/approve board 1" in out["notice"]["text"]
    assert (state / "review" / "ch1" / "board.jpg").exists()
    assert not led.approved(1, "board")
    assert ci.command(proj, led, state, "/approve board 1", "owner", "owner").startswith("Approved")

    assert run()["notice"]["kind"] == "running"                  # takes
    assert run()["notice"]["kind"] == "review"
    claude_reviews(2)
    out = run()
    assert out["notice"]["kind"] == "animatic" and (releases / "cuts" / "chapter_1.mp4").exists()
    assert ci.command(proj, led, state, "/approve animatic 1", "owner", "owner").startswith("Approved")

    assert run()["notice"]["kind"] == "running"                  # 4K
    out = run()
    assert out["notice"]["kind"] == "done", out["notice"]
    w, h, seconds = probe(releases / "cuts" / "chapter_1_4k.mp4")
    assert (w, h) == (3840, 2160) and seconds == pytest.approx(sum(s.seconds for s in proj.chapter(1)), abs=0.1)
    assert all(not Path(j["path"]).is_absolute() for s in proj.chapter(1) for j in led.jobs(s.id) if j["path"])


def test_no_reference_stills_means_no_keyframe_batch(tmp_path):
    proj = copy_project(tmp_path)
    for f in (tmp_path / "style_bible" / "refs").glob("*.png"):
        f.unlink()
    (tmp_path / "style_bible" / "ref_prompts.json").unlink()
    led = steps.open_ledger(proj)
    with pytest.raises(ValueError, match="reference stills"):
        kg.build(proj, led, "keyframes", 1, engine_set="stub")
    assert not led.active_batch()


@needs_ffmpeg
def test_reference_stills_are_painted_picked_by_claude_and_approved_by_the_owner(tmp_path, monkeypatch):
    from pipeline import ci
    from pipeline.store import ReleaseStore
    proj = copy_project(tmp_path)
    for f in ("tsumugi_ch1", "sayo"):                       # two refs still to make
        (tmp_path / "style_bible" / "refs" / f"{f}.png").unlink()
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/mochizuki")
    monkeypatch.setenv("GITHUB_REPOSITORY_OWNER", "owner")
    proj.store = ReleaseStore("owner/mochizuki", runner=FakeGH(tmp_path / "releases"))
    led = steps.open_ledger(proj)
    local = kg.LocalKaggle(proj.media)
    state = tmp_path / "state"
    assert kg.next_stage(proj, led, 1) == "refs"

    assert ci.step(proj, led, 1, state, local, engine_set="stub")["notice"]["kind"] == "running"
    out = ci.step(proj, led, 1, state, local, engine_set="stub")
    assert out["notice"]["kind"] == "review"
    q = json.loads((state / "review" / "ch1" / "queue.json").read_text())
    refs_items = [i for i in q["items"] if i["what"] == "ref"]
    assert {i["name"] for i in refs_items} == {"tsumugi_ch1", "sayo"}
    assert all(len(i["candidates"]) == 4 for i in refs_items)
    assert not [i for i in q["items"] if i["what"] == "keyframe"]          # keyframes wait for gate 1
    (state / "reviews").mkdir(exist_ok=True)
    (state / "reviews" / "c.json").write_text(json.dumps({"select": [i["candidates"][1]["key"] for i in refs_items]}))
    out = ci.step(proj, led, 1, state, local, engine_set="stub")
    assert out["notice"]["kind"] == "refs" and (state / "review" / "ch1" / "refs.jpg").exists()
    assert kg.next_stage(proj, led, 1) is None

    assert "rejected" in ci.command(proj, led, state, "/reject sayo", "owner", "owner").lower()
    assert steps.ref_state(proj, led, {"name": "sayo"}) == "review"        # three candidates left to choose from
    led.select(next(j["key"] for j in led.jobs("ref-sayo", "ref") if j["verdict"] is None))
    assert ci.command(proj, led, state, "/approve refs", "claude[bot]", "owner").startswith("Only")
    reply = ci.command(proj, led, state, "/approve refs", "owner", "owner")
    assert reply.startswith("Approved") and (tmp_path / "style_bible" / "refs" / "sayo.png").exists()
    assert not proj.refs_missing() and kg.next_stage(proj, led, 1) == "keyframes"
