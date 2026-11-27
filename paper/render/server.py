"""Local-only helper server for headless rendering.

Serves the files in this directory and the structures in data/pdb, and stores what the page
posts back (images, projected coordinates) in the output directory given on the command line.
"""
import http.server
import os
import sys
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
PDB = os.path.join(HERE, "..", "..", "data", "pdb")
OUT = sys.argv[2]


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=HERE, **k)

    def log_message(self, *a):
        pass

    def translate_path(self, path):
        p = urllib.parse.urlparse(path).path
        if p.startswith("/pdb/"):
            return os.path.join(PDB, os.path.basename(p))
        if p.startswith("/scene/"):
            return os.path.join(OUT, os.path.basename(p))
        return super().translate_path(path)

    def do_POST(self):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        name = os.path.basename(q.get("name", ["out.bin"])[0])
        n = int(self.headers.get("Content-Length", 0))
        with open(os.path.join(OUT, name), "wb") as f:
            f.write(self.rfile.read(n))
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
