"""Declutter checkpoints + results by moving DEAD (non-current) experiments into _archive/.

Keeps every experiment whose name starts with one of the --keep prefixes; moves the rest
into models/checkpoints/_archive/ and results/_archive/. Only experiment-named entries
(prefix "run" or "spin") are candidates, so control outputs like results/track_mpc and any
underscore-special dir are always protected. Dry-run by default.

    python tools/archive_experiments.py                      # show what WOULD move
    python tools/archive_experiments.py --apply               # actually move
    python tools/archive_experiments.py --keep run07,run08    # keep a different active set
"""
import argparse
import shutil
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())
CKPT = ROOT / "models" / "checkpoints"
RES  = ROOT / "results"

EXPERIMENT_PREFIXES = ("run", "spin")   # only these are archive candidates; track_mpc etc. are safe


def _archivable(name: str, keep: tuple) -> bool:
    if not name.startswith(EXPERIMENT_PREFIXES):   # not an experiment (e.g. track_mpc) -> protect
        return False
    return not name.startswith(keep)               # keep the active runs


def plan(keep: tuple):
    moves = []
    if CKPT.exists():
        for f in sorted(CKPT.glob("*.pt")):
            if _archivable(f.name, keep):
                moves.append((f, CKPT / "_archive" / f.name))
    if RES.exists():
        for d in sorted(RES.iterdir()):
            if d.is_dir() and _archivable(d.name, keep):
                moves.append((d, RES / "_archive" / d.name))
    return moves


def main():
    ap = argparse.ArgumentParser(description="Archive dead experiments (dry-run by default).")
    ap.add_argument("--keep", default="run07,run08,run09",
                    help="comma-separated name prefixes to KEEP active")
    ap.add_argument("--apply", action="store_true", help="actually move (default: dry run)")
    a = ap.parse_args()
    keep = tuple(s.strip() for s in a.keep.split(",") if s.strip())

    moves = plan(keep)
    if not moves:
        print(f"nothing to archive (keeping {keep})")
        return

    verb = "MOVING" if a.apply else "WOULD MOVE"
    print(f"{verb} {len(moves)} items  (keeping prefixes {keep}):")
    for src, dst in moves:
        print(f"  {src.relative_to(ROOT)}  ->  {dst.relative_to(ROOT)}")

    if not a.apply:
        print("\ndry run - re-run with --apply to move.")
        return

    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    print(f"\ndone: archived {len(moves)} items into _archive/.")


if __name__ == "__main__":
    main()
