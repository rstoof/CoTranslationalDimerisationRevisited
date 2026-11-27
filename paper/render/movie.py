"""Movie and snapshots of a three-dimensional time-lapse (ctd.chain3d.timelapse).

Renders the frames with headless Chrome and the NGL viewer, writes paper/figures/movie.mp4,
movie.gif, snapshot_*.png and movie.json. Optional: the outputs are stored in the repository.

    python3 paper/render/movie.py [seed]

The trajectory is chosen from results/sim3d.pkl (the one that pairs nearest to the middle of the
window in which both interfaces are exposed). Its frames are read from results/clips/, written by
compute_3d.py, or the trajectory is computed again.
"""
from __future__ import annotations

import json
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from render import PORT, chain_groups, run_page, stop, wait_for  # noqa: E402

FIG = ROOT / "paper" / "figures"
# ribosomes pale, so that the chains stand out
PALE = dict(rib1=dict(rna="#8fbdf2", protein="#c9dffa"), rib2=dict(rna="#f4a781", protein="#fbd5c2"), mrna="#3a3a38")
CHAIN = {0: "#0d366b", 1: "#8a3008"}                 # leading and trailing chain outside the interface
IFACE_WAIT, IFACE, RNA = "#b9b0f2", "#5a45d6", "#3a3a38"
PAIRED = {0: "#d81b60", 1: "#a3124a"}                # the two strands of the paired interface
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
W, H = 960, 720


# ---------------------------------------------------------------------------------- geometry of a frame
def pose_of(f):
    p = np.asarray(f["pose"], float)
    return p[:9].reshape(3, 3), p[9:]


def smooth(p, sub=4):
    """Catmull-Rom interpolation through the points."""
    p = np.asarray(p, float)
    if len(p) < 3:
        return p
    q = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    out = []
    for i in range(1, len(q) - 2):
        for s in np.arange(sub) / sub:
            out.append(0.5 * ((2 * q[i]) + (-q[i - 1] + q[i + 1]) * s + (2 * q[i - 1] - 5 * q[i] + 4 * q[i + 1] - q[i + 2]) * s * s
                              + (-q[i - 1] + 3 * q[i] - 3 * q[i + 1] + q[i + 2]) * s ** 3))
    out.append(p[-1])
    return np.array(out)


def chain_tube(x, anchor, attached, bound, meta, which, competent, sub=4):
    """Tube through the beads of one chain (N-terminus first) and on to the tunnel exit."""
    n = len(x)
    if n == 0:
        return None
    pts = np.vstack([x, anchor[None]]) if attached else np.asarray(x, float)
    col = []
    for i in range(len(pts)):
        k = min(i, n - 1)
        inside = meta["i0"] <= k < meta["i1"]
        col.append(PAIRED[which] if (k < len(bound) and bound[k]) else (IFACE if competent else IFACE_WAIT) if inside else CHAIN[which])
    sm = smooth(pts, sub)
    idx = np.minimum((np.arange(len(sm)) + sub // 2) // sub, len(pts) - 1)
    colors = [col[i] for i in idx]
    radii = [5.0 if c == PAIRED[which] else 5.5 if c in (IFACE, IFACE_WAIT) else 4.0 for c in colors]
    return dict(xyz=np.round(sm, 1).tolist(), colors=colors, radii=radii)


def rna_tube(q1, q2, c1, c2, contour):
    """Slack line for the RNA between the two ribosomes. Its path is schematic: the simulation treats
    this stretch of RNA as a spring."""
    r = np.linalg.norm(q2 - q1)
    slack = np.sqrt(max(min(contour, 1.6 * r + 80.0) ** 2 - r * r, 0.0)) * 0.35
    mid = (q1 + q2) / 2
    out = mid - (c1 + c2) / 2
    out = out / (np.linalg.norm(out) + 1e-9)
    s = np.linspace(0, 1, 24)[:, None]
    pts = (1 - s) ** 2 * q1 + 2 * s * (1 - s) * (mid + slack * out) + s ** 2 * q2
    return dict(xyz=np.round(pts, 1).tolist(), colors=[RNA] * len(pts), radii=[5.0] * len(pts))


def exposed_interface(n_beads, meta):
    return max(min(n_beads, meta["i1"]) - meta["i0"], 0)


def tubes_of(f, meta, d):
    R, t = pose_of(f)
    comp = [exposed_interface(len(f["xa"]), meta) >= meta["gate"], exposed_interface(len(f["xb"]), meta) >= meta["gate"]]
    tubes = []
    for which, (xs, anchor) in enumerate(((f["xa"], d.exit1), (f["xb"], R @ d.exit2 + t))):
        tb = chain_tube(np.asarray(xs, float), anchor, f["attached"][which], f["bound"], meta, which, comp[which])
        if tb:
            tubes.append(tb)
    if f["rib"] == (1, 1):
        tubes.append(rna_tube(d.q1, R @ d.q2 + t, d.centre[1], R @ d.centre[2] + t, meta["contour"]))
    return tubes


def _axes(frames, d, meta):
    """Viewing axes for frames with both ribosomes. e1 runs from the leading to the trailing ribosome
    (the viewer shows -e1 to the right). e2 points to the side of the pair on which the interfaces of
    the chains are, so that they are seen in profile; once a stretch is paired, e2 is turned so that
    the paired stretch lies in the plane of the image. Also returns the position of the interfaces
    relative to the middle of the pair."""
    out, prev = [], None
    for f in frames:
        R, t = pose_of(f)
        c1, c2 = d.centre[1], R @ d.centre[2] + t
        e1 = c2 - c1; e1 /= np.linalg.norm(e1)
        chains = [np.asarray(x, float) for x in (f["xa"], f["xb"]) if len(x)]
        b = np.asarray(f["bound"]).astype(bool)
        paired = [x[:len(b)][b[:len(x)]] for x in chains] if b.sum() >= 8 else []
        part = paired or [x[meta["i0"]:meta["i1"]] for x in chains if len(x) > meta["i0"] + 3] or chains
        w = np.vstack(part).mean(0) - (c1 + c2) / 2 if part else d.u1 + R @ d.u2
        e2 = w - (w @ e1) * e1
        if np.linalg.norm(e2) < 1e-3:
            e2 = d.u1 - (d.u1 @ e1) * e1
        if np.linalg.norm(e2) < 1e-3:
            e2 = np.cross(e1, [0.0, 0.0, 1.0])
        e2 /= np.linalg.norm(e2)
        if paired:
            P = np.vstack(paired)
            a = np.linalg.svd(P - P.mean(0), full_matrices=False)[2][0]
            a = a - (a @ e1) * e1
            if np.linalg.norm(a) > 0.35:
                a /= np.linalg.norm(a)
                e2 = a if a @ (prev if prev is not None else e2) >= 0 else -a
        prev = e2
        out.append((e1, e2, w))
    return out


def _extent(f, d, e1, e2, e3, radius=150.0):
    R, t = pose_of(f)
    pts = [c + radius * s * a for on, c in ((f["rib"][0], d.centre[1]), (f["rib"][1], R @ d.centre[2] + t)) if on
           for a in (e1, e2) for s in (-1, 1)]
    P = np.array(pts + list(np.asarray(f["xa"], float)) + list(np.asarray(f["xb"], float)))
    u, v, z = P @ e1, P @ e2, P @ e3
    return u.min(), u.max(), v.min(), v.max(), z.mean()


TOP, BOTTOM = 84, 118                 # pixels kept free for the captions above and below the scene


def _fit(du, dv, margin):
    """Height of the view (A) that shows a box of du x dv in the part of the image between the captions."""
    return float(max((dv + 2 * margin) * H / (H - TOP - BOTTOM), (du + 2 * margin) * H / W))


def _camera(e1, e2, e3, mid, view_h):
    shift = (BOTTOM - TOP) / 2 * view_h / H          # move the scene up, into the free part of the image
    centre = mid[0] * e1 + (mid[1] - shift) * e2 + mid[2] * e3
    return dict(e1=e1.tolist(), e2=e2.tolist(), e3=e3.tolist(), centre=centre.tolist(), view_height=view_h, mid=list(map(float, mid)))


def cameras_follow(frames, d, meta, margin=28.0, sigma=3.0):
    """Cameras that follow the pair of ribosomes: leading one on the right, tunnel exits towards the top
    where possible, smoothed over the frames; one magnification for the whole clip."""
    from scipy.ndimage import gaussian_filter1d
    raw = _axes(frames, d, meta)
    E1 = gaussian_filter1d(np.array([r[0] for r in raw]), sigma, axis=0, mode="nearest")
    E2 = gaussian_filter1d(np.array([r[1] for r in raw]), sigma, axis=0, mode="nearest")
    E1 /= np.linalg.norm(E1, axis=1, keepdims=True)
    E2 -= np.sum(E2 * E1, axis=1, keepdims=True) * E1
    E2 /= np.linalg.norm(E2, axis=1, keepdims=True)
    E3 = np.cross(E1, E2)
    # the viewer looks along +e3: if the interfaces are behind the ribosomes, look from the other side
    if np.median([r[2] @ e3 for r, e3 in zip(raw, E3)]) > 20.0:
        E1, E3 = -E1, -E3
    ext = np.array([_extent(f, d, a, b, c) for f, a, b, c in zip(frames, E1, E2, E3)])
    view_h = max(_fit(x[1] - x[0], x[3] - x[2], margin) for x in ext)
    mid = gaussian_filter1d(np.stack([(ext[:, 0] + ext[:, 1]) / 2, (ext[:, 2] + ext[:, 3]) / 2, ext[:, 4]], axis=1), sigma, axis=0, mode="nearest")
    return [_camera(a, b, c, m, view_h) for a, b, c, m in zip(E1, E2, E3, mid)]


def cameras_fixed(frames, d, cam, margin=28.0):
    """The view of `cam`, kept for all frames; returns the cameras and the magnification they need."""
    e1, e2, e3, mid = (np.array(cam[k]) for k in ("e1", "e2", "e3", "mid"))
    ext = np.array([_extent(f, d, e1, e2, e3) for f in frames])
    du = 2 * max(abs(ext[:, 0].min() - mid[0]), abs(ext[:, 1].max() - mid[0]))
    dv = 2 * max(abs(ext[:, 2].min() - mid[1]), abs(ext[:, 3].max() - mid[1]))
    view_h = max(cam["view_height"], _fit(du, dv, margin))
    return [_camera(e1, e2, e3, mid, view_h) for _ in frames], view_h


# ---------------------------------------------------------------------------------- rendering
def render_frames(scene, work, name="movie", timeout=3600):
    """Render all frames into the directory `work`; returns the paths of the images."""
    (work / f"{name}_scene.json").write_text(json.dumps(scene))
    procs = run_page(work, "movie.html", name, scene, port=PORT + 1)
    try:
        wait_for(work, name, timeout)
    finally:
        stop(procs)
    return [work / f"{name}_frame_{k:04d}.png" for k in range(len(scene["frames"]))]


def build_scene(frames, cameras, meta, d, factors=None):
    g = chain_groups(d)
    mchain = dict(auth=g["mrna"]["auth"][0], label=g["mrna"]["label"][0])
    lo, hi = int(d.mrna_seq.min()), int(d.mrna_seq.max())
    cut = int(d.psite_nt[0] + d.psite_nt[1]) // 2
    out = [dict(pose=[float(a) for a in f["pose"]], rib=[int(a) for a in f["rib"]], tubes=tubes_of(f, meta, d), camera=cam,
                factor=(factors or {}).get(k, 1)) for k, (f, cam) in enumerate(zip(frames, cameras))]
    return dict(structure="9GFT.cif.gz", groups=g, colors=PALE, width=W, height=H, frames=out,
                mrna=dict(rib1=dict(chain=mchain, **{"from": cut + 4, "to": hi}), rib2=dict(chain=mchain, **{"from": lo, "to": cut - 4})))


# ---------------------------------------------------------------------------------- captions
def font(size, bold=False):
    for f, idx in (("/System/Library/Fonts/HelveticaNeue.ttc", 1 if bold else 0), ("/System/Library/Fonts/Helvetica.ttc", 1 if bold else 0),
                   ("/System/Library/Fonts/Supplemental/Arial.ttf", 0)):
        try:
            return ImageFont.truetype(f, size, index=idx)
        except OSError:
            continue
    return ImageFont.load_default()


def state_text(f, meta):
    n_pair = int(np.sum(f["bound"])) * meta["res_per_bead"]
    comp = [exposed_interface(len(f["xa"]), meta) >= meta["gate"], exposed_interface(len(f["xb"]), meta) >= meta["gate"]]
    if tuple(f["rib"]) == (0, 0):
        return "both chains released: " + ("a dimer formed during translation" if n_pair else "two monomers")
    if tuple(f["rib"]) == (0, 1):
        return "leading chain released; " + ("the dimer hangs on the trailing ribosome" if n_pair else "no dimer was formed")
    if n_pair:
        return f"{n_pair} interface residues paired: the chains join the two ribosomes"
    if all(comp):
        return "both interfaces exposed: the chains are competent"
    if comp[0]:
        return "interface of the leading chain exposed; trailing chain not yet competent"
    return "interfaces still emerging from the ribosomes"


def times(meta):
    """Translation times (s) at which the trailing chain becomes competent and at which the chains are released."""
    b = meta["res_per_bead"]
    codon_comp = meta["tunnel"] + (meta["i0"] + meta["gate"]) * b
    step_comp = -(-codon_comp // b)
    return step_comp * b / meta["v"], -(-(meta["L"] - meta["spacing"]) // b) * b / meta["v"], -(-meta["L"] // b) * b / meta["v"]


def annotate(png, f, meta, clip_us, size=(W, H)):
    im = Image.open(png).convert("RGB")
    if im.size != size:
        im = im.resize(size, Image.LANCZOS)
    dr = ImageDraw.Draw(im)
    s = size[0] / 960
    dr.text((22 * s, 16 * s), f"translation: {f['time']:4.1f} s", fill=INK, font=font(int(26 * s), True))
    where = ("ribosomes at codon %d (leading) and %d (trailing) of %d" % (f["codon_a"], f["codon_b"], meta["L"])
             if tuple(f["rib"]) == (1, 1) else "trailing ribosome at codon %d of %d" % (f["codon_b"], meta["L"])
             if tuple(f["rib"]) == (0, 1) else "translation of both chains completed")
    dr.text((22 * s, 50 * s), where, fill=INK2, font=font(int(16 * s)))
    # legend
    x0, y0 = size[0] - 268 * s, 20 * s
    for j, (col, lab) in enumerate(((IFACE_WAIT, "interface, emerging"), (IFACE, "interface, competent"), (PAIRED[0], "paired"),
                                    (CHAIN[0], "rest of the chains"))):
        y = y0 + j * 22 * s
        dr.line([(x0, y + 8 * s), (x0 + 30 * s, y + 8 * s)], fill=col, width=int(6 * s))
        if j == 3:
            dr.line([(x0 + 15 * s, y + 8 * s), (x0 + 30 * s, y + 8 * s)], fill=CHAIN[1], width=int(6 * s))
        dr.text((x0 + 40 * s, y), lab, fill=INK2, font=font(int(15 * s)))
    # time line of translation
    t_comp, t_rel_a, t_rel_b = times(meta)
    xl, xr, yb = 22 * s, size[0] - 22 * s, size[1] - 34 * s
    X = lambda t: xl + (xr - xl) * min(t, t_rel_b) / t_rel_b
    dr.rectangle([xl, yb, xr, yb + 8 * s], fill="#e1e0d9")
    dr.rectangle([X(t_comp), yb, X(t_rel_a), yb + 8 * s], fill=IFACE_WAIT)
    xm = X(f["time"])
    dr.polygon([(xm, yb - 2 * s), (xm - 7 * s, yb - 13 * s), (xm + 7 * s, yb - 13 * s)], fill=INK)
    dr.text((xl, yb + 12 * s), "0 s", fill=MUTED, font=font(int(13 * s)))
    lab = "both interfaces exposed"
    fw = dr.textlength(lab, font=font(int(13 * s)))
    dr.text(((X(t_comp) + X(t_rel_a)) / 2 - fw / 2, yb + 12 * s), lab, fill=INK2, font=font(int(13 * s)))
    end = f"{t_rel_b:.0f} s"
    dr.text((xr - dr.textlength(end, font=font(int(13 * s))), yb + 12 * s), end, fill=MUTED, font=font(int(13 * s)))
    dr.text((22 * s, yb - 62 * s), state_text(f, meta), fill=INK, font=font(int(18 * s), True))
    dr.text((22 * s, yb - 38 * s), f"clip of {clip_us:.1f} microseconds of motion; translation jumps ahead between clips",
            fill=MUTED, font=font(int(13 * s)))
    return im


# ---------------------------------------------------------------------------------- choice of clips
def choose_clips(movie, meta, steps):
    """Clips (lists of frames) that tell the story, as groups that share a camera."""
    clips, ev = movie["clips"], movie["event"]
    b = meta["res_per_bead"]
    step_at = lambda codon: min(-(-codon // b), len(clips) - 1)
    comp = step_at(meta["tunnel"] + (meta["i0"] + meta["gate"]) * b)          # trailing chain competent from this step
    rel_a = next(k for k, c in enumerate(clips) if tuple(c[-1]["rib"]) != (1, 1))
    rel_b = next(k for k, c in enumerate(clips) if tuple(c[-1]["rib"]) == (0, 0))
    last = lambda c, n=20: c[-n:]
    story = [("emerging", last(clips[step_at(70)])), ("leading", last(clips[max(comp - 6, step_at(70) + 1)]))]
    if ev is not None:
        e, k = ev["step"], ev["index"]
        if e - comp >= 2:
            story.append(("competent", last(clips[(comp + e) // 2])))
        story.append(("pairing", ev["frames"][max(k - 20, 0):k + 61]))
        if rel_a - e >= 6:
            story.append(("paired", last(clips[(e + rel_a) // 2])))
    else:
        story.append(("competent", last(clips[(comp + rel_a) // 2])))
    # the ending is shown with one view: the leading ribosome leaves, then the trailing one
    ending = [("before release", last(clips[rel_a - 1])), ("released", last(clips[rel_a])),
              ("before end", last(clips[rel_b - 1])), ("free", last(clips[min(rel_b + 1, len(clips) - 1)]))]
    return story, ending


def zip_time(movie, meta):
    """Time (s) from the first pairing until 90% of the interface exposed by both chains is paired."""
    ev = movie["event"]
    if ev is None:
        return None
    for k in range(ev["index"], len(ev["frames"])):
        f = ev["frames"][k]
        if np.sum(f["bound"]) >= 0.9 * (min(len(f["xa"]), len(f["xb"]), meta["i1"]) - meta["i0"]):
            return (k - ev["index"]) * meta["fine"] * meta["sweep_time"]
    return None


def make(steps, meta, movie, d, tag="movie", stride=1):
    story, ending = choose_clips(movie, meta, steps)
    frames, cameras, labels = [], [], []
    for name, clip in story:
        clip = clip[::stride]
        frames += clip; labels += [name] * len(clip); cameras += cameras_follow(clip, d, meta)
    before = ending[0][1][::stride]
    after = [f for _, clip in ending[1:] for f in clip[::stride]]
    cams_before = cameras_follow(before, d, meta)
    cams_after, view_h = cameras_fixed(after, d, cams_before[-1])
    view_h = max(view_h, cams_before[0]["view_height"])
    for c in cams_before + cams_after:                       # one magnification for the whole ending
        new = _camera(*(np.array(c[k]) for k in ("e1", "e2", "e3", "mid")), view_h)
        c.update(new)
    frames += before + after
    cameras += cams_before + cams_after
    labels += [ending[0][0]] * len(before) + [name for name, clip in ending[1:] for _ in clip[::stride]]
    # snapshots for the figure: one frame of each part of the story, rendered at twice the size
    want = ["leading", "competent", "pairing", "paired", "released", "free"]
    shots = {}
    for name in want:
        idx = [k for k, lab in enumerate(labels) if lab == name]
        if not idx:
            continue
        if name == "pairing":            # the first frame in which interface residues have paired
            k0 = next((k for k in idx if np.sum(frames[k]["bound"]) > 0), idx[len(idx) // 2])
            k = k0
        else:
            k = idx[len(idx) // 2]
        shots[name] = k
    if "paired" not in shots and "pairing" in shots:      # late pairing: use the end of the pairing clip
        shots["paired"] = [k for k, lab in enumerate(labels) if lab == "pairing"][-1]
    scene = build_scene(frames, cameras, meta, d, factors={k: 2 for k in shots.values()})
    with tempfile.TemporaryDirectory(prefix="ctd-movie-") as tmpdir:
        return _finish(frames, cameras, labels, shots, scene, meta, movie, Path(tmpdir), tag, stride)


def _finish(frames, cameras, labels, shots, scene, meta, movie, work, tag, stride):
    """Render the frames, write the movie, the snapshots and movie.json."""
    t0 = time.time()
    pngs = render_frames(scene, work, tag)
    print(f"rendered {len(frames)} frames in {time.time() - t0:.0f} s")
    clip_us = meta["fine"] * stride * meta["sweep_time"] * 1e6
    per_clip = {name: sum(lab == name for lab in labels) * clip_us for name in set(labels)}
    ims = [annotate(p, f, meta, per_clip[lab]) for p, f, lab in zip(pngs, frames, labels)]
    tmp = work / "mp4"
    tmp.mkdir()
    k_out = 0
    for k, im in enumerate(ims):          # hold the first and last frame of every clip a little longer
        first = k == 0 or labels[k] != labels[k - 1]
        last = k == len(ims) - 1 or labels[k] != labels[k + 1]
        for _ in range(8 if last else 4 if first else 1):
            im.save(tmp / f"f{k_out:05d}.png"); k_out += 1
    fps = 10
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(tmp / "f%05d.png"), "-pix_fmt", "yuv420p",
                    "-c:v", "libx264", "-crf", "25", "-preset", "slow", "-movflags", "+faststart", str(FIG / f"{tag}.mp4")], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(tmp / "f%05d.png"), "-filter_complex",
                    "fps=6,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];"
                    "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle", str(FIG / f"{tag}.gif")], check=True)
    info = dict(seed=meta["seed"], frames=len(frames), fps=fps, frame_ns=clip_us * 1e3, clip_frames=20,
                motion_competent_us=meta["sweeps_competent"] * meta["sweep_time"] * 1e6,
                motion_other_us=meta["sweeps_other"] * meta["sweep_time"] * 1e6, first_pair_codon=meta["first_pair_codon"],
                first_pair_time=None if meta["first_pair_codon"] is None else meta["first_pair_codon"] / meta["v"], shots={})
    zt = zip_time(movie, meta)
    info["zip_us"] = None if zt is None else zt * 1e6
    for name, k in shots.items():
        im = Image.open(pngs[k]).convert("RGB")
        a = np.asarray(im)
        ink = np.argwhere(a.min(-1) < 247)
        (y0, x0), (y1, x1) = ink.min(0), ink.max(0)
        m = 16
        im.crop((max(x0 - m, 0), max(y0 - m, 0), min(x1 + m, a.shape[1]), min(y1 + m, a.shape[0]))).save(FIG / f"snapshot_{name.replace(' ', '_')}.png", optimize=True)
        f = frames[k]
        info["shots"][name] = dict(time=f["time"], codon_a=f["codon_a"], codon_b=f["codon_b"], paired=int(np.sum(f["bound"])) * meta["res_per_bead"],
                                   state=state_text(f, meta), view_height=cameras[k]["view_height"],
                                   px_nm=cameras[k]["view_height"] / 10.0 / a.shape[0])
    (FIG / f"{tag}.json").write_text(json.dumps(info, indent=1))
    for ext in ("mp4", "gif"):
        print("wrote", FIG / f"{tag}.{ext}", f"({(FIG / f'{tag}.{ext}').stat().st_size / 1e6:.1f} MB)")
    return info


def main():
    from ctd import chain3d as c3
    res = pickle.load(open(ROOT / "results" / "sim3d.pkl", "rb"))
    runs = res["timelapse"]
    t_comp, t_rel_a, _ = times(runs[0][1])
    mid = (t_comp + t_rel_a) / 2 * runs[0][1]["v"]
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    else:
        paired = [m for _, m in runs if m["first_pair_codon"] is not None]
        if not paired:
            sys.exit("no trajectory with a pairing event")
        seed = min(paired, key=lambda m: abs(m["first_pair_codon"] - mid))["seed"]
    stored = next(m for _, m in runs if m["seed"] == seed)
    path = ROOT / "results" / "clips" / f"seed_{seed}.pkl"
    if path.exists():
        print("trajectory with seed", seed, "- frames from", path)
        steps, meta, movie = pickle.load(open(path, "rb"))
    else:
        print("trajectory with seed", seed, "- computing it again with all frames kept")
        steps, meta, movie = c3.timelapse(seed=seed, clips=True)
    assert meta["first_pair_sweep"] == stored["first_pair_sweep"], "the trajectory differs from the stored one"
    make(steps, meta, movie, c3.disome())


if __name__ == "__main__":
    main()
