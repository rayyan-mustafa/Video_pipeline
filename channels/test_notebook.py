"""Simulate both Colab runtimes against the real cells in stickman.ipynb.

    .venv\Scripts\python.exe test_notebook.py

The notebook now runs twice: once on a T4 (draw) and once on a free CPU runtime
(review, encode, upload). Colab bills wall-clock on the runtime rather than GPU
use, so the two steps that never touch the card - the 15-minute frame review and
the 12-minute x264 encode - were costing ~49 compute units per 60 videos for
nothing.

These checks execute the ACTUAL cell sources with HAS_GPU forced either way, so
a phase guard cannot rot without this failing.
"""
import contextlib, io as _io, json, io, os, re, sys, tempfile
from pathlib import Path

NB = str(Path(__file__).resolve().parent / "stickman.ipynb")
nb = json.load(io.open(NB, encoding="utf-8"))
cells = nb["cells"]
codes = [(i, "".join(c["source"])) for i, c in enumerate(cells) if c["cell_type"] == "code"]
ok = lambda m: print(f"  ok  {m}")


def cell_with(needle):
    return next(s for _, s in codes if needle in s)


def run(src, g):
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            exec(compile(src, "<cell>", "exec"), g)
        except SystemExit:
            pass
    return buf.getvalue()


# ---------------------------------------------------------------- phase cell
phase = cell_with("WHICH PHASE AM I IN?")

g = {}
out = run(phase, g)
detected = "GPU RUNTIME" if g["HAS_GPU"] else "CPU RUNTIME"
assert g["PHASE"] in ("gpu", "cpu")
assert "PHASE 2 of 2" in out or "PHASE 1 of 2" in out
ok(f"Step 0 detects a runtime without importing torch -> {detected}")
assert "Step 24" in out and "Step 25" in out if not g["HAS_GPU"] else True
ok("Step 0 prints a per-phase checklist")

# ---------------------------------------------------------------- 1c stubs
flux = cell_with("PHASE GUARD  --  no FLUX on the CPU phase")
g = {"HAS_GPU": False}
out = run(flux, g)
assert "FLUX not loaded" in out
assert g["release_flux"]() is None and g["flux_loaded"]() is False
try:
    g["get_flux"]()
    raise AssertionError("get_flux should refuse on CPU")
except RuntimeError as e:
    assert "T4 GPU" in str(e)
ok("Step 1c on CPU: stubs defined, get_flux() explains instead of NameError")

# ---------------------------------------------------------------- 1a install
env = cell_with("ENV_VERSION")
assert 'MARKER = f"/content/.pipeline_env_{ENV_VERSION}_{' in env
gpu_pkgs = env.split("if not HAS_GPU:")[1].split("# --- core")[0]
for heavy in ("bitsandbytes", "diffusers", "transformers", "accelerate", "torch"):
    assert heavy not in gpu_pkgs, f"CPU install list must not contain {heavy}"
for light in ("Pillow", "soundfile", "ipywidgets", "google-api-python-client"):
    assert light in gpu_pkgs, f"CPU install list is missing {light}"
ok("Step 1a CPU list has no torch/diffusers/bitsandbytes, keeps what CPU needs")

# ---------------------------------------------------------------- 1b verify
ver = cell_with("VERIFY THE ENVIRONMENT")

# missing deps must fail loudly and name the fix
try:
    run(ver, {"HAS_GPU": False})
    _missing_ok = False
except RuntimeError as e:
    _missing_ok = "re-run Step 1a" in str(e)
assert _missing_ok, "Step 1b did not catch a missing CPU dependency"

# with the light stack present it should pass and point at the next step
import importlib.util as _iu
_real_find, _real_run = _iu.find_spec, __import__("subprocess").run
_iu.find_spec = lambda n: object()
class _R: returncode = 0
__import__("subprocess").run = lambda *a, **k: _R()
try:
    out = run(ver, {"HAS_GPU": False})
finally:
    _iu.find_spec = _real_find
    __import__("subprocess").run = _real_run
assert "CPU phase environment OK" in out, out
assert "skip to Step 24" in out, out
ok("Step 1b on CPU: passes with the light stack, fails loudly without it")

# ---------------------------------------------------------------- 24 queueing
W = Path(tempfile.mkdtemp(prefix="phase_")); os.chdir(W)
os.makedirs("frames", exist_ok=True)
scenes = [{"scene_index": i, "text": f"Line {i}.", "section": 1, "duration": 2.0,
           "action_clause": f"action {i}", "caption": "", "visual_prompt": "p"} for i in range(4)]
Path("scenes_planned.json").write_text(json.dumps(scenes), encoding="utf-8")

rev = cell_with("def on_redraw")
assert "NO GPU HERE? QUEUE IT INSTEAD OF DRAWING IT." in rev
assert 'REDRAW_QUEUE = "redraw_queue.json"' in rev
# the queue branch must return BEFORE anything touches get_flux
q_branch = rev.split("if not HAS_GPU:")[1].split("if not flux_loaded()")[0]
assert "get_flux" not in q_branch and "return" in q_branch
ok("Step 24 queue branch returns before any GPU call")

# exercise the queue logic itself, lifted verbatim from the cell
body = re.search(r"if not HAS_GPU:\n(.*?)\n            return", rev, re.S).group(1)
body = "\n".join(l[12:] if l.startswith(" " * 12) else l for l in body.splitlines())
ns = {"os": os, "json": json, "REDRAW_QUEUE": "redraw_queue.json",
      "idx": 2, "seed_box": type("B", (), {"value": 42})(),
      "_save": lambda: None, "status": type("S", (), {"value": ""})()}
exec(compile(body, "<queue>", "exec"), ns)
ns["idx"] = 0
exec(compile(body, "<queue>", "exec"), ns)
ns["idx"] = 2                      # re-queue the same frame
exec(compile(body, "<queue>", "exec"), ns)
q = json.loads(Path("redraw_queue.json").read_text(encoding="utf-8"))
assert [e["scene_index"] for e in q] == [0, 2], q
ok("Step 24 queue: appends, and re-queuing a frame replaces rather than duplicates")

# ---------------------------------------------------------------- 24b redraw
rd = cell_with("BATCH REDRAW")
g = {"HAS_GPU": False}
try:
    run(rd, g)
    raise AssertionError("24b should refuse without a GPU")
except RuntimeError as e:
    assert "T4 GPU" in str(e)
ok("Step 24b refuses to run on the CPU phase")

# stale entries dropped, not fatal
Path("redraw_queue.json").write_text(json.dumps(
    [{"scene_index": 0, "seed": 1}, {"scene_index": 99, "seed": 1}]), encoding="utf-8")
drawn = []


class _Pipe:
    def __call__(self, *a, **k):
        from PIL import Image
        import random
        drawn.append(k)
        im = Image.new("RGB", (32, 18))
        im.putdata([(random.randint(0, 255),) * 3 for _ in range(32 * 18)])
        return type("R", (), {"images": [im]})()


sys.path.insert(0, str(Path(__file__).resolve().parent))
import _torch_stub as torch_stub  # a Generator/inference_mode shim; no real torch here
g = {"HAS_GPU": True, "get_flux": lambda: _Pipe(), "release_flux": lambda *a, **k: None,
     "FLUX_SEED": 42, "FRAME_WIDTH": 32, "FRAME_HEIGHT": 18, "FLUX_STEPS": 4,
     "finish_frame": lambda im, cap: im, "torch": torch_stub}
out = run(rd, g)
assert "ignoring 1 queued frame" in out, out
assert "All 1 redrawn" in out, out
assert not Path("redraw_queue.json").exists()
assert Path("frames/frame_0000.png").exists()
ok("Step 24b: draws the queue, drops stale entries, clears the file")

# ---------------------------------------------------------------- 25 warning
enc = cell_with("THIS STEP DOES NOT TOUCH THE GPU")
assert "if HAS_GPU:" in enc.split("def _audio_seconds")[0]
ok("Step 25 warns when it is burning GPU wall-clock")

# ---------------------------------------------------------------- ordering
order = [i for i, c in enumerate(cells) if c["cell_type"] == "markdown"
         and re.search(r"Step (24b|24|25|26|27)\b", "".join(c["source"]))]
labels = [re.search(r"Step (\d+b?)", "".join(cells[i]["source"])).group(1) for i in order]
assert labels == ["24", "24b", "26", "25", "27"], labels
ok("order: 24 review -> 24b redraw -> 26 thumbnail (GPU) -> 25 encode (CPU) -> 27")

# ------------------------------------------------- read sheets vs the aligner
# pipeline/readsheet.py carries a copy of build_chunks() so run_script.py can
# write read sheets on the laptop, where there is no torch and no notebook. A
# copy that drifts is worse than no copy: the words you read would stop being
# the words Step 17 expects to hear, the difflib match rate would drop, and the
# scene boundaries in that section would quietly become guesses.
#
# So execute the REAL cell's chunker and compare it against the package's on a
# script built to hit the merge and split paths.
align_cell = cell_with("def build_chunks")
chunk_src = align_cell[align_cell.index("MAX_CHUNK_CHARS"): align_cell.index('if NARRATION')]
nb_ns = {"re": re}
exec(compile(chunk_src, "<chunker>", "exec"), nb_ns)

from pipeline import readsheet  # noqa: E402  (after sys.path was set above)

assert nb_ns["MAX_CHUNK_CHARS"] == readsheet.MAX_CHUNK_CHARS
assert nb_ns["MIN_CHUNK_CHARS"] == readsheet.MIN_CHUNK_CHARS

sample = []
for sec in (1, 2, 3):
    sample += [
        {"sentence": "Short one.", "section": sec},                       # merges
        {"sentence": "A middling sentence with enough words to stand on its own here.",
         "section": sec},
        {"sentence": ("A very long sentence, which keeps going past the limit, "
                      "with several clauses, each of them separated by a comma, "
                      "so the splitter has somewhere to cut it, and then some "
                      "more words after that, and more again, until it is well "
                      "over two hundred and eighty characters long in total."),
         "section": sec},
        {"sentence": "Tiny.", "section": sec},                            # boundary case
    ]

nb_chunks = nb_ns["build_chunks"](sample)
pkg_chunks = readsheet.build_chunks(sample)
assert nb_chunks == pkg_chunks, (
    "readsheet.build_chunks has drifted from notebook Step 17:\n"
    f"  notebook: {nb_chunks}\n  package:  {pkg_chunks}")
ok(f"read sheets: package chunker matches Step 17's, {len(nb_chunks)} chunks")

# The drift that actually bites: a short sentence at the top of a section can
# belong to a chunk the aligner files under the PREVIOUS section. Grouping by
# the section tag alone would put it in the wrong read sheet, so assert the
# sheets are built from the chunks and not from the tags.
by_tag = {}
for it in sample:
    by_tag.setdefault(it["section"], []).append(it["sentence"])
by_chunk = {s: " ".join(t) for s, t in readsheet.section_texts(sample).items()}
assert any(" ".join(by_tag[s]).split() != by_chunk[s].split() for s in by_chunk), (
    "this fixture no longer exercises the boundary case the read sheets exist for")
ok("read sheets follow the aligner's grouping, not the section tags")

print("\nBOTH PHASES VERIFIED")
