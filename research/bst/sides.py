"""Which labelled player (A/B) is on the Top (far) side of the court, per rally.

This is BST's own rule (repo/ShuttleSet/gen_my_dataset.py), reproduced exactly:
match.csv's `downcourt` flag says whether A starts at the top; the players change
ends after set 1, and in set 3 again after the rally in which one of them first
reaches 11 points.  A fixed "B is Top" rule is right only for one end of one set.

    top = top_letter_map(label_dir)      # {(set, rally): "A" | "B"}
    side = "Top" if row["player"] == top[(set, rally)] else "Bottom"
"""
import csv, os


def _downcourt(label_dir):
    match_csv = os.path.join(os.path.dirname(os.path.abspath(label_dir)), "match.csv")
    name = os.path.basename(os.path.normpath(label_dir))
    for r in csv.DictReader(open(match_csv, encoding="utf-8")):
        if r["video"] == name:
            return bool(int(float(r["downcourt"])))
    raise KeyError(f"{name} not in {match_csv}")


def _searchsorted_left(values, x):
    for i, v in enumerate(values):
        if v >= x:
            return i
    return len(values)


def top_letter_map(label_dir):
    a_top_first = _downcourt(label_dir)
    out = {}
    for f in sorted(os.listdir(label_dir)):
        if not (f.startswith("set") and f.endswith(".csv")):
            continue
        s = int(f[3:-4])
        rows = list(csv.DictReader(open(os.path.join(label_dir, f), encoding="utf-8")))
        rallies = [int(float(r["rally"])) for r in rows]
        if s in (1, 2):
            a_top = a_top_first ^ (s == 2)
            for rl in rallies:
                out[(s, rl)] = "A" if a_top else "B"
            continue
        # set 3: split after the rally in which someone first reaches 11
        sa = [int(float(r["roundscore_A"])) for r in rows]
        sb = [int(float(r["roundscore_B"])) for r in rows]
        i = _searchsorted_left(sa, 11)
        if i >= len(rows) or sa[i] <= sb[i]:
            i = _searchsorted_left(sb, 11)
        split_rally = rallies[min(i, len(rows) - 1)]
        for rl in rallies:
            first_half = rl <= split_rally
            a_top = a_top_first if first_half else not a_top_first
            out[(s, rl)] = "A" if a_top else "B"
    return out


def side_of(top_map, set_no, rally, player):
    return "Top" if player == top_map[(set_no, rally)] else "Bottom"
