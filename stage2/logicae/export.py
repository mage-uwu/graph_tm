"""LogicAE export: copy the run's models and log off the pod.

Copies every checkpoint / export in the run directory (pt.ltc, pt.ltc.best, ckpt_<step>.ltc and the
fine-tuned <name>.ltc / .lth plus their .best copies), logicae.log and any sweep_*.log into one directory under a
random token, writes a sha256 manifest (`sha256  size  name`) and serves it read-only over HTTP on
`port` for `hours` (detached, no directory listings: only /<token>/<file>). Stdlib only.

  python3 export.py [--dir /root/lae] [--wait] [--hours 3] [--port 8888]

--wait polls <dir>/phase until the runner has written "done". Status lines go to stdout and, when
writable, to the container log (/proc/1/fd/1), so they show in the pod's logs:
  LAE-EXPORT serving N files on :8888 for 3 h (token X)
  LAE-EXPORT <sha256>  <size>  <name>
"""
import argparse
import glob
import hashlib
import http.server
import os
import shutil
import subprocess
import sys
import time

SUFFIXES = (".ltc", ".ltc.best", ".lth", ".lth.best")


def say(msg):
    line = f"== [{time.strftime('%H:%M:%S')}] {msg}" if not msg.startswith("LAE-EXPORT ") else msg
    print(line, flush=True)
    try:
        with open("/proc/1/fd/1", "a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def files(d):
    out = [f for f in sorted(glob.glob(os.path.join(d, "*")))
           if f.endswith(SUFFIXES) and os.path.isfile(f) and not os.path.basename(f).startswith("probe.")]
    logs = [os.path.join(d, "logicae.log")] + sorted(glob.glob(os.path.join(d, "sweep_*.log")))
    return out + [f for f in logs if os.path.isfile(f)]


def export(d, port, hours):
    tok = os.urandom(12).hex()
    out = os.path.join(d, "export", tok)
    os.makedirs(out)
    lines = []
    for f in files(d):
        name = os.path.basename(f)
        dst = os.path.join(out, name)
        shutil.copy(f, dst)
        h = hashlib.sha256()
        with open(dst, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        lines.append(f"{h.hexdigest()}  {os.path.getsize(dst)}  {name}")
    with open(os.path.join(out, "MANIFEST.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    subprocess.Popen(["timeout", str(int(hours * 3600)), sys.executable, os.path.abspath(__file__), "--serve", out,
                      "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)
    say(f"LAE-EXPORT serving {len(lines)} files on :{port} for {hours:g} h (token {tok})")
    for l in lines:
        say(f"LAE-EXPORT {l}")


def serve(d, port):
    tok = os.path.basename(d.rstrip("/"))

    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=d, **k)

        def send_head(self):
            parts = self.path.split("?")[0].strip("/").split("/")
            if len(parts) != 2 or parts[0] != tok or not os.path.isfile(os.path.join(d, parts[1])):
                self.send_error(404)
                return None
            self.path = "/" + parts[1]
            return super().send_head()

        def log_message(self, *a):
            pass

    http.server.ThreadingHTTPServer(("", port), H).serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.environ.get("W", "/root/lae"))
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--hours", type=float, default=3)
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--serve", metavar="DIR")
    a = ap.parse_args()
    if a.serve:
        return serve(a.serve, a.port)
    if a.wait:
        say(f"LAE-EXPORT waiting for {a.dir}/phase to read 'done'")
        while True:
            try:
                with open(os.path.join(a.dir, "phase")) as fh:
                    if fh.read().strip() == "done":
                        break
            except OSError:
                pass
            time.sleep(60)
    export(a.dir, a.port, a.hours)


if __name__ == "__main__":
    main()
