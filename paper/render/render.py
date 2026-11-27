"""Render the two-ribosome structure as ribbons with base stubs, using headless Chrome and the
NGL viewer. Optional: the rendered images are stored in paper/figures, so the figures can be
rebuilt without Chrome.

    python3 paper/render/render.py            # paper/figures/render_structure.png (+ .json)
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT = 8791
COLORS = dict(rib1=dict(rna="#2a78d6", protein="#9ec5f4"), rib2=dict(rna="#eb6834", protein="#f6b08c"),
              mrna="#0b0b0b", nascent="#e87ba4")


def chain_groups(d):
    """Chains of each ribosome, by both naming conventions of the mmCIF file."""
    at = d.at
    g = {k: dict(auth=set(), label=set()) for k in ("rib1", "rib2", "mrna", "nascent")}
    vemp = None
    for asym in dict.fromkeys(at["asym"]):
        m = at["asym"] == asym
        r = int(d.rib[m][0]); auth = str(at["auth_asym"][m][0])
        key = {1: "rib1", 2: "rib2", 0: "mrna"}.get(r)
        if key:
            g[key]["auth"].add(auth); g[key]["label"].add(str(asym))
    return {k: dict(auth=sorted(v["auth"]), label=sorted(v["label"])) for k, v in g.items()}


def view_axes(d):
    e1 = d.exit2 - d.exit1; e1 /= np.linalg.norm(e1)
    um = d.u1 + d.u2; e2 = um - (um @ e1) * e1; e2 /= np.linalg.norm(e2)
    return e1, e2, np.cross(e1, e2)


def render(scene: dict, name: str, out_png: Path, timeout=180):
    """Send one scene to the page and wait for the image. Returns the projected points."""
    with tempfile.TemporaryDirectory(prefix="ctd-render-") as tmp:
        work = Path(tmp)
        (work / f"{name}_scene.json").write_text(json.dumps(scene))
        procs = run_page(work, "scene.html", name, scene)
        try:
            wait_for(work, name, timeout)
            out_png.write_bytes((work / f"{name}_image.png").read_bytes())
            return json.loads((work / f"{name}_points.json").read_text())
        finally:
            stop(procs)


def run_page(work: Path, page: str, name: str, scene: dict, port: int = PORT):
    """Start the helper server and headless Chrome on a page; returns both processes."""
    server = subprocess.Popen([sys.executable, str(HERE / "server.py"), str(port), str(work)])
    time.sleep(0.8)
    chrome = subprocess.Popen([CHROME, "--headless=new", f"--user-data-dir={work / 'profile'}", "--no-first-run",
                               "--disable-extensions", f"--window-size={scene['width'] + 40},{scene['height'] + 40}",
                               "--enable-unsafe-swiftshader", f"http://127.0.0.1:{port}/{page}?run={name}"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return chrome, server


def wait_for(work: Path, name: str, timeout: float):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if (work / f"{name}_error.txt").exists():
            raise RuntimeError((work / f"{name}_error.txt").read_text()[:2000])
        if (work / f"{name}_done.txt").exists():
            return
        time.sleep(0.5)
    raise TimeoutError("rendering did not finish")


def stop(procs):
    for p in procs:
        p.terminate()
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()


def base_scene(d, width=1500, height=1200, view_height=480.0, centre_shift=(0.0, -118.0)):
    e1, e2, e3 = view_axes(d)
    centre = (d.exit1 + d.exit2) / 2 + centre_shift[0] * e1 + centre_shift[1] * e2
    return dict(structure="9GFT.cif.gz", groups=chain_groups(d), colors=COLORS, width=width, height=height, factor=1,
                view_height=view_height, e1=e1.tolist(), e2=e2.tolist(), e3=e3.tolist(), centre=centre.tolist(), settle=3000)


def main():
    from ctd import pairgeom as pg
    d = pg.Disome()
    scene = base_scene(d)
    pts = dict(exit1=d.exit1, exit2=d.exit2, tip1=d.exit1 + 70 * d.u1, tip2=d.exit2 + 70 * d.u2,
               mrna_mid=d.mrna[len(d.mrna) // 2], mrna_end=d.mrna[np.argmax(d.mrna_seq)], c1=d.centre[1], c2=d.centre[2])
    scene["points"] = [np.asarray(v).tolist() for v in pts.values()]
    out = ROOT / "paper" / "figures" / "render_structure.png"
    res = render(scene, "structure", out)
    W, H = res["canvas"]
    proj = {k: [p[0] / W, p[1] / H] for k, p in zip(pts, res["points"])}       # fractions of the image, origin bottom-left
    meta = dict(points=proj, view_height_A=scene["view_height"], aspect=W / H, chain_key=res["key"],
                exit_distance_A=float(np.linalg.norm(d.exit2 - d.exit1)))
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print("wrote", out, "| chain naming used by the viewer:", res["key"])
    print(json.dumps(proj))


if __name__ == "__main__":
    main()
