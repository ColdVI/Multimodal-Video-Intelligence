"""Pencere sonuclarini surekli zaman araliklarina birlestirir."""
from collections import defaultdict


def merge_intervals(rows, gap_tol: float = 10.0, min_score=None):
    """rows: (video_id, t_start, t_end, dist) iterable (herhangi bir sirada).
    Donus: (video_id, t0, t1, score) listesi, skora gore azalan sirada."""
    by_vid = defaultdict(list)
    for vid, t0, t1, dist in rows:
        score = 1 - dist
        if min_score is not None and score < min_score:
            continue
        by_vid[vid].append((t0, t1, score))

    results = []
    for vid, wins in by_vid.items():
        wins.sort()
        cur_start, cur_end, cur_score = wins[0]
        for t0, t1, s in wins[1:]:
            if t0 - cur_end <= gap_tol:
                cur_end = max(cur_end, t1)
                cur_score = max(cur_score, s)
            else:
                results.append((vid, cur_start, cur_end, cur_score))
                cur_start, cur_end, cur_score = t0, t1, s
        results.append((vid, cur_start, cur_end, cur_score))

    results.sort(key=lambda r: -r[3])
    return results


def fmt(t: float) -> str:
    h, r = divmod(int(t), 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}"


def pretty(results, n: int = 10) -> str:
    lines = [f"{vid}  {fmt(a)}\u2013{fmt(b)}  (skor {s:.2f})"
             for vid, a, b, s in results[:n]]
    return "\n".join(lines)
