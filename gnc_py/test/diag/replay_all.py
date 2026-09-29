"""Replay every captured solve under the current build and environment and summarize
error codes and residual inflated-grid hits ("clean" = error_code 0 and no hit).

    python3 replay_all.py <tag> <capture.json> [...] [--shift-start=DY]

--shift-start moves each solve's start back along -y by DY [m] (away from the person box).
"""
import sys

import diag_common


def main():
    tag = sys.argv[1]
    shift = next((float(a.split("=", 1)[1]) for a in sys.argv[2:] if a.startswith("--shift-start=")), 0.0)
    rows = []
    for path in [a for a in sys.argv[2:] if not a.startswith("--")]:
        capture = diag_common.load_capture(path)
        grid = diag_common.build_grid(capture)
        for call in capture["calls"]:
            call = dict(call, args=list(call["args"]))
            waypoints = list(call["args"][0])
            waypoints[1] -= shift
            call["args"][0] = waypoints
            _ok, code, segment_times, coeffs, duration = diag_common.replay(call, grid)
            hits = len(diag_common.inflated_hits(grid, coeffs, segment_times)) if coeffs else -1
            name = path.split("/")[-1].replace(".json", "") + ":" + call["label"]
            rows.append((code, hits))
            print("ROW %s %s orig=%d code=%d hits=%d T0=%.2f dur=%.1f" % (
                tag, name, call["error_code"], code, hits, segment_times[0] if segment_times else -1,
                duration), flush=True)
    clean = sum(1 for code, hits in rows if code == 0 and hits == 0)
    print("SUMMARY %s shift=%.2f clean=%d/%d code0=%d" % (
        tag, shift, clean, len(rows), sum(1 for code, _ in rows if code == 0)), flush=True)


if __name__ == "__main__":
    main()
