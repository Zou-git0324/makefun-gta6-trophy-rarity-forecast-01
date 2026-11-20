"""Forecast GTA VI (PS5) trophy earn rates on Exophase at a target instant.

Method (stdlib only, no network, deterministic):
  * For each trophy, collect the distinct reads in the case history and fit
    a straight line to logit(rate) against ln(1 + hours since release).
    Cumulative earn rates are S-shaped in time, and logit-vs-log-time is
    roughly linear while a game is young, so this extrapolates growth
    instead of assuming rates stay put.
  * The per-trophy slope is shrunk toward the pooled median slope across
    trophies (and, when the history has few distinct reads, toward a weak
    prior slope measured from the baked Exophase snapshot in data/).
    Slopes are clipped so one noisy trophy cannot explode.
  * The median is anchored on the fitted line at the last read, then
    progression logic is enforced: story trophies are non-increasing
    along Chapter 1 -> 2 -> 3 -> 4 -> final mission, the 100%-completion
    gold cannot exceed the final-mission rate, and the platinum cannot
    exceed the 100%-completion rate. Long-grind trophies get a ceiling
    so a steep early slope cannot forecast an absurd week-one rate.
  * Quantile spread on the logit scale combines a random-walk term
    (0.5 * sqrt(horizon in days), as in the house baseline), the slope's
    standard error times the extrapolation distance in log-time, and a
    fixed refresh-shock term: Exophase's tracked-player pool jumps when
    it refreshes, which can move every rate at once, so honest bands
    must be wider than a pure random walk.
"""

import glob
import json
import math
import os
import sys

Z = (-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722)
RELEASE = "2026-11-19T00:00:00Z"

# Trophy ids in story order (must be non-increasing in earn rate).
STORY_CHAIN = (3, 4, 5, 6, 7)
# id -> ceiling (percent) for trophies whose grind makes a fast week-one
# rate implausible; only binds when extrapolation would otherwise explode.
CEILINGS = {1: 1.5, 14: 3.0, 15: 8.0, 17: 5.0, 18: 3.0, 19: 12.0,
            25: 3.0, 30: 2.5, 31: 2.5, 32: 8.0, 40: 3.0}


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp into epoch seconds (float)."""
    text = str(text).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    from datetime import datetime
    return datetime.fromisoformat(text).timestamp()


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit(pct):
    p = clamp(pct, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(y):
    if y >= 0.0:
        return 100.0 / (1.0 + math.exp(-y))
    e = math.exp(y)
    return 100.0 * e / (1.0 + e)


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, residual_std, n)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0, 0.0, n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(sum((y - (a + b * x)) ** 2
                              for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return a, b, resid, n


def load_prior():
    """Weak prior slopes from the baked Exophase snapshot, or {}."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "data", "exophase_snapshot_2026-11-20.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            snap = json.load(fh)
    except (OSError, ValueError):
        return {}
    try:
        t0 = parse_time(RELEASE)
        x_launch = math.log1p(max(0.0, (parse_time(snap["launch_at"]) - t0) / 3600.0))
        x_snap = math.log1p(max(0.0, (parse_time(snap["fetched_at"]) - t0) / 3600.0))
    except (KeyError, ValueError):
        return {}
    if x_snap - x_launch < 0.5:
        return {}
    prior = {}
    launch = snap.get("launch_rates") or []
    snap_rates = snap.get("snapshot_rates") or []
    for i, (r0, r1) in enumerate(zip(launch, snap_rates)):
        slope = (logit(r1) - logit(r0)) / (x_snap - x_launch)
        prior[i + 1] = clamp(slope, -2.0, 4.0)
    return prior


def distinct_points(history, release_ts, n_trophies):
    """Collapse history into per-trophy lists of (x, logit) change points.

    Consecutive identical reads carry no new information (Exophase only
    refreshes on its own schedule), so only points where a rate changed
    are kept, plus the final read. Returns (points, last_rates, last_x).
    """
    points = [[] for _ in range(n_trophies)]
    last_rates = [None] * n_trophies
    last_x = None
    prev_rates = None
    for read in history:
        try:
            t = parse_time(read["at"])
            rates = read["rates"]
        except (KeyError, TypeError, ValueError):
            continue
        if not isinstance(rates, list) or len(rates) < n_trophies:
            continue
        x = math.log1p(max(0.0, (t - release_ts) / 3600.0))
        last_x = x
        changed = prev_rates is None
        for i in range(n_trophies):
            r = rates[i]
            if r is None:
                continue
            last_rates[i] = float(r)
            if prev_rates is not None and prev_rates[i] != r:
                changed = True
        if changed:
            for i in range(n_trophies):
                r = rates[i]
                if r is None:
                    continue
                points[i].append((x, logit(float(r))))
        prev_rates = list(rates)
    # Keep the final read as a point only if its rate actually changed
    # from the last kept point: a flat tail is one plateau, not a trend.
    # (The last rate and its time are still tracked separately as the
    # anchor, so dropping a duplicate point loses nothing.)
    if history and last_x is not None:
        for i in range(n_trophies):
            if last_rates[i] is not None:
                if not points[i] or points[i][-1][1] != logit(last_rates[i]):
                    points[i].append((last_x, logit(last_rates[i])))
    return points, last_rates, last_x


def forecast_case(case, prior):
    trophies = case["trophies"]
    n = len(trophies)
    ids = [int(t["id"]) for t in trophies]
    try:
        release_ts = parse_time(case.get("game", {}).get("release_date", "2026-11-19")
                                + "T00:00:00Z")
    except ValueError:
        release_ts = parse_time(RELEASE)
    try:
        at_ts = parse_time(case["at"])
        horizon_min = float(case.get("horizon_minutes") or 0.0)
        target_ts = at_ts + horizon_min * 60.0
        if case.get("target"):
            target_ts = parse_time(case["target"])
    except (KeyError, TypeError, ValueError):
        at_ts = release_ts
        target_ts = release_ts
        horizon_min = 0.0
    x_target = math.log1p(max(0.0, (target_ts - release_ts) / 3600.0))
    horizon_days = max(0.0, (target_ts - at_ts) / 86400.0)

    points, last_rates, last_x = distinct_points(
        case.get("history") or [], release_ts, n)
    if last_x is None:
        last_x = math.log1p(max(0.0, (at_ts - release_ts) / 3600.0))

    # First pass: per-trophy OLS fits.
    fits = []
    slopes = []
    for i in range(n):
        pts = points[i]
        if len(pts) >= 2 and pts[-1][0] > pts[0][0]:
            a, b, resid, _ = ols([p[0] for p in pts], [p[1] for p in pts])
            fits.append((a, clamp(b, -2.0, 8.0), resid, len(pts), pts[-1][0]))
            slopes.append(clamp(b, -2.0, 8.0))
        else:
            fits.append(None)
    pooled = median(slopes) if slopes else 0.0
    pooled = clamp(pooled, -0.5, 3.0)

    rows = {}
    for idx, tid in enumerate(ids):
        last = last_rates[idx]
        if last is None:
            last = 0.01
        last = float(last)
        # The 100% trophy: everyone tracked has it; it can only drift
        # down as fresh players join, and only slowly.
        if last >= 99.99:
            rows[tid] = (99.7, 0.18)
            continue
        fit = fits[idx]
        if fit is not None:
            a, b_fit, resid, npts, _ = fit
            prior_b = prior.get(tid, pooled)
            pw = 1.0 if npts < 3 else 0.25
            b = (npts * b_fit + 2.0 * pooled + pw * prior_b) / (npts + 2.0 + pw)
            b = clamp(b, -1.5, 6.0)
            y_last_fit = a + b_fit * last_x
            y_anchor = 0.5 * y_last_fit + 0.5 * logit(last)
            dx = max(0.0, x_target - last_x)
            y_hat = y_anchor + b * dx
            # Slope standard error from residuals (fallback when tiny n).
            if npts > 2 and resid > 0.0:
                xs = [p[0] for p in points[idx]]
                mx = sum(xs) / len(xs)
                sxx = sum((x - mx) ** 2 for x in xs)
                se_b = resid / math.sqrt(sxx) if sxx > 1e-12 else 0.8
            else:
                se_b = 0.8
            se_b = clamp(se_b, 0.15, 1.5)
            sigma = math.sqrt((0.5 * math.sqrt(horizon_days)) ** 2
                              + (se_b * dx) ** 2 + 0.30 ** 2)
        else:
            # No usable trend: drift from the last rate with the pooled
            # slope, or the prior slope if there is one.
            b = prior.get(tid, pooled)
            b = clamp(b, -1.5, 6.0)
            dx = max(0.0, x_target - last_x)
            y_hat = logit(last) + b * dx
            sigma = math.sqrt((0.5 * math.sqrt(horizon_days)) ** 2
                              + (0.8 * dx) ** 2 + 0.30 ** 2)
        sigma = clamp(sigma, 0.35, 2.2)
        med = sigmoid(y_hat)
        ceil = CEILINGS.get(tid)
        if ceil is not None:
            med = min(med, ceil)
        med = clamp(med, 0.01, 99.99)
        rows[tid] = (med, sigma)

    # Progression logic: story chain non-increasing; platinum <= 100%
    # gold <= final-mission rate.
    for chain in (STORY_CHAIN,):
        prev = None
        for tid in chain:
            if tid in rows:
                med, sig = rows[tid]
                if prev is not None:
                    med = min(med, prev)
                rows[tid] = (med, sig)
                prev = med
    if 7 in rows and 14 in rows:
        rows[14] = (min(rows[14][0], rows[7][0]), rows[14][1])
    if 14 in rows and 1 in rows:
        rows[1] = (min(rows[1][0], rows[14][0]), rows[1][1])

    out = []
    for tid in ids:
        med, sigma = rows[tid]
        y_med = logit(med)
        qs = []
        for z in Z:
            q = sigmoid(y_med + z * sigma)
            qs.append(round(clamp(q, 0.0, 100.0), 4))
        # Guarantee non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        out.append({"id": tid, "q": qs})
    return {"trophies": out}


def fallback_case(case):
    """Last-rate forecast with a house-style spread; never crashes."""
    rows = []
    try:
        trophies = case["trophies"]
        history = case.get("history") or []
        last_rates = history[-1]["rates"] if history else []
        horizon_days = max(0.0, float(case.get("horizon_minutes") or 0.0) / 1440.0)
    except (KeyError, TypeError, ValueError, IndexError):
        return {"trophies": []}
    sigma = 0.5 * math.sqrt(horizon_days) if horizon_days > 0 else 0.5
    for t in trophies:
        tid = int(t["id"])
        r = 0.01
        try:
            if tid - 1 < len(last_rates) and last_rates[tid - 1] is not None:
                r = float(last_rates[tid - 1])
        except (TypeError, ValueError):
            r = 0.01
        y = logit(r)
        qs = [round(clamp(sigmoid(y + z * sigma), 0.0, 100.0), 4) for z in Z]
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": tid, "q": qs})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    prior = load_prior()
    for path in sorted(glob.glob(os.path.join(in_dir, "*.in"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path, "r", encoding="utf-8") as fh:
                case = json.load(fh)
            result = forecast_case(case, prior)
            if len(result.get("trophies", [])) != len(case.get("trophies", [])):
                raise ValueError("trophy count mismatch")
        except Exception:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    case = json.load(fh)
                result = fallback_case(case)
            except Exception:
                continue
        out_path = os.path.join(out_dir, stem + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(result, separators=(",", ":")))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
