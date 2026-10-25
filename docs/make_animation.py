"""Animated walk-through of the paper's running example. Two robots tidy a kitchen. Robot A puts away plates and
glasses, Robot B forks and spoons, and Robot B uses Team-Plan to decide whether to step in and how. The panel on the
right follows Team-Plan's steps at each decision.

The animation is a schematic with illustrative numbers, not a recording of the simulator.

    python docs/make_animation.py                       # docs/assets/team_plan_demo.gif and a still
    python docs/make_animation.py --mp4                 # also docs/assets/team_plan_demo.mp4 (needs imageio-ffmpeg)
    python docs/make_animation.py --font-dir DIR        # use the Open Sans files in DIR
    python docs/make_animation.py --stills 15,30        # only PNG stills at these times, for checking the layout
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager, transforms  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

# ------------------------------------------------------------------------------------------------ palette
BG = "#F4F6FA"
INK, MUTED, FAINT = "#1F2937", "#6B7280", "#C5CDD8"
A_COL, A_DARK, A_FACE = "#F59E0B", "#B45309", "#FFF3D6"          # Robot A, the partner
B_COL, B_DARK, B_FACE = "#2563EB", "#1D4ED8", "#DCE8FF"          # Robot B, which uses Team-Plan
NAVY, ACTIVE_BG = "#1E3A5F", "#E3EDFF"
CAUSE_COLS = ["#7C5CBF", "#0EA5A0", "#E0475B"]
GOOD, BAD, BAR_GREY, TRACK = "#16A34A", "#DC2626", "#A0AEC0", "#EEF1F5"
KITCHEN, DINING, HALL, WALL = "#FFF4E3", "#E6F0FF", "#EFEAF8", "#8B97A8"

DURATION = 57.0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


# ------------------------------------------------------------------------------------------------ timing helpers
def ease(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def ramp(t: float, t0: float, t1: float) -> float:
    return ease((t - t0) / (t1 - t0)) if t1 > t0 else float(t >= t0)


def window(t: float, t0: float, t1: float, fade: float = 0.35) -> float:
    return min(ramp(t, t0, t0 + fade), 1 - ramp(t, t1 - fade, t1))


def path(t: float, keys) -> np.ndarray:
    if t <= keys[0][0]:
        return np.array(keys[0][1], float)
    for (t0, p0), (t1, p1) in zip(keys, keys[1:]):
        if t <= t1:
            return np.array(p0, float) + ramp(t, t0, t1) * (np.array(p1, float) - np.array(p0, float))
    return np.array(keys[-1][1], float)


# ------------------------------------------------------------------------------------------------ the story
SLOTS = {"plate0": (1.25, 7.15), "plate1": (1.6, 7.15), "glass0": (2.2, 7.15), "glass1": (2.48, 7.15),
         "fork0": (3.05, 7.15), "fork1": (3.3, 7.15), "spoon0": (3.8, 7.15), "spoon1": (4.05, 7.15)}
CABINET = {"fork0": (7.72, 7.22), "fork1": (7.94, 7.22), "glass0": (8.28, 7.22), "glass1": (8.55, 7.22),
           "spoon0": (8.85, 7.22), "spoon1": (9.05, 7.22), "plate0": (8.0, 6.83), "plate1": (8.6, 6.83)}
PATH_B = [(4.0, (3.9, 4.6)), (5.0, (3.2, 6.42)), (5.3, (3.2, 6.42)), (6.8, (7.9, 6.32)), (7.1, (7.9, 6.32)),
          (8.4, (3.95, 6.42)), (8.7, (3.95, 6.42)), (12.4, (4.7, 5.2)), (23.6, (4.7, 5.2)), (26.8, (8.9, 6.32)),
          (40.4, (8.9, 6.32)), (44.0, (1.45, 6.42)), (44.4, (1.45, 6.42)), (46.5, (8.35, 6.32))]
PATH_A = [(4.0, (2.0, 4.8)), (5.2, (2.34, 6.42)), (5.5, (2.34, 6.42)), (7.6, (8.45, 6.32)), (7.9, (8.45, 6.32)),
          (9.6, (3.0, 4.75)), (25.6, (3.0, 4.75)), (27.4, (1.43, 6.42)), (42.0, (1.43, 6.42)), (43.2, (1.0, 4.75))]
CARRIES = [("fork0", "b", 5.3, 6.8), ("fork1", "b", 5.3, 6.8), ("glass0", "a", 5.5, 7.6), ("glass1", "a", 5.5, 7.6),
           ("spoon0", "b", 8.7, 26.8), ("spoon1", "b", 8.7, 26.8), ("plate0", "b", 44.4, 46.5),
           ("plate1", "b", 44.4, 46.5)]
DONE = {"plates": 46.6, "glasses": 7.7, "forks": 6.9, "spoons": 26.9}

STEPS = ["Is help needed?", "Why is Robot A stuck?", "Which help gains the most?", "Is the best choice clear?",
         "Act, then watch"]
# (start, end, step, card)
STEP_TIMELINE = [(7.2, 10.6, 0, "no"), (12.5, 14.5, 0, "maybe"), (14.5, 17.5, 1, "belief"), (17.5, 21.0, 2, "gain"),
                 (21.0, 23.6, 3, "clear1"), (23.6, 31.5, 4, "act1"), (31.5, 34.5, 1, "belief"), (34.5, 38.0, 2, "gain"),
                 (38.0, 40.3, 3, "clear2"), (40.3, 51.0, 4, "act2")]
TEXT_CARDS = {
    "no": ["Robot A is heading to its items.", "No help needed for now.", "Robot B continues its own work."],
    "maybe": ["The plates are still untouched", "halfway through the task.", "Help may be needed. But why?"],
    "clear1": ["Best: remind, +0.31", "Next: tell where, +0.16", "A clear lead, so act now."],
    "clear2": ["Best: take over, +0.52", "Next: remind, +0.02", "A clear lead, so act now."],
    "act1": ["Robot B sends a reminder", "and keeps watching Robot A.", ""],
    "act2": ["Robot B takes over the plates.", "Its own work is already done.", ""],
}
CLOSE_NOTE = "With a small lead, Team-Plan would\nfirst simulate the options."
CAUSES = ["forgot the plates", "wrong location", "weak grip"]
PRIOR, POSTERIOR = np.array([0.55, 0.30, 0.15]), np.array([0.08, 0.05, 0.87])
ACTIONS = ["remind", "tell where", "take over", "continue"]
GAIN1, GAIN2 = np.array([0.31, 0.16, 0.12, 0.0]), np.array([0.02, 0.01, 0.52, 0.0])

NARRATION = [(4.0, 7.2, "Robot A tidies plates and glasses, Robot B forks and spoons."),
             (7.2, 12.5, "At each decision, Team-Plan asks if Robot B should help."),
             (12.5, 14.5, "The plates stay untouched. Is help needed?"),
             (14.5, 17.5, "The cause is hidden, so Team-Plan weighs possible causes."),
             (17.5, 21.0, "It scores each kind of help by its gain for the whole task."),
             (21.0, 23.6, "A reminder is clearly best, so Robot B acts now."),
             (23.6, 27.4, "Robot B sends the reminder and keeps watching."),
             (27.4, 31.5, "Robot A tries, but cannot pick up the plates."),
             (31.5, 34.5, "The failed pick points to a weak grip."),
             (34.5, 40.3, "Now taking over gains the most for the whole task."),
             (40.3, 47.0, "Robot B takes over the plates."),
             (47.0, 51.4, "All four assignments are done.")]
BUBBLES = [("b", 23.8, 25.3, "Please put away the plates."), ("a", 25.3, 26.9, "On it!"),
           ("b", 40.5, 42.3, "I'll take over the plates.")]
THINKING = [(12.5, 23.6), (31.5, 40.3)]


# ------------------------------------------------------------------------------------------------ sprites
class Sprite:
    """Patches drawn around the origin and moved together."""

    def __init__(self, ax, patches, z: float):
        self.ax, self.patches = ax, patches
        for p in patches:
            p.set_zorder(z); ax.add_patch(p)

    def place(self, x: float, y: float) -> None:
        tr = transforms.Affine2D().translate(x, y) + self.ax.transData
        for p in self.patches:
            p.set_transform(tr)


def robot(ax, body: str, face: str) -> Sprite:
    s = 1.2
    parts = [Ellipse((0, -0.37 * s), 0.8 * s, 0.16 * s, fc=(0, 0, 0, 0.13), ec="none"),
             FancyBboxPatch((-0.3 * s, -0.28 * s), 0.6 * s, 0.54 * s, boxstyle=f"round,pad=0.02,rounding_size={0.13 * s}",
                            fc=body, ec="none"),
             FancyBboxPatch((-0.21 * s, -0.01 * s), 0.42 * s, 0.2 * s, boxstyle=f"round,pad=0.01,rounding_size={0.08 * s}",
                            fc=face, ec="none"),
             Circle((-0.09 * s, 0.09 * s), 0.046 * s, fc=INK, ec="none"), Circle((0.09 * s, 0.09 * s), 0.046 * s, fc=INK, ec="none"),
             Rectangle((-0.014 * s, 0.26 * s), 0.028 * s, 0.13 * s, fc=body, ec="none"),
             Circle((0, 0.41 * s), 0.055 * s, fc=face, ec=body, lw=1.8),
             Circle((-0.2 * s, -0.31 * s), 0.065 * s, fc="#4B5563", ec="none"), Circle((0.2 * s, -0.31 * s), 0.065 * s, fc="#4B5563", ec="none")]
    return Sprite(ax, parts, 6)


def item(ax, kind: str) -> Sprite:
    if kind == "plate":
        parts = [Circle((0, 0), 0.16, fc="white", ec="#3B82F6", lw=1.6), Circle((0, 0), 0.09, fc="none", ec="#93C5FD", lw=1.1)]
    elif kind == "glass":
        parts = [Polygon([(-0.085, 0.14), (0.085, 0.14), (0.06, -0.13), (-0.06, -0.13)], closed=True, fc="#BDE7F8",
                         ec="#0284C7", lw=1.4)]
    elif kind == "fork":
        parts = [Rectangle((-0.014, -0.16), 0.028, 0.21, fc="#4B5563", ec="none")]
        parts += [Rectangle((dx - 0.01, 0.04), 0.02, 0.11, fc="#4B5563", ec="none") for dx in (-0.048, 0, 0.048)]
        parts += [Rectangle((-0.058, 0.03), 0.116, 0.028, fc="#4B5563", ec="none")]
    else:
        parts = [Rectangle((-0.014, -0.16), 0.028, 0.18, fc="#4B5563", ec="none"),
                 Ellipse((0, 0.07), 0.09, 0.14, fc="#4B5563", ec="none")]
    return Sprite(ax, parts, 7)


def card(ax, x, y, w, h, fc="white", ec="#D9DEE7", lw=1.5, z=1, r=0.18):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z)
    ax.add_patch(p)
    return p


# ------------------------------------------------------------------------------------------------ static scene
def build(fig):
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")
    fig.patch.set_facecolor(BG)
    card(ax, 0.4, 3.45, 4.25, 4.45, fc=KITCHEN, ec=WALL, lw=2.2)
    card(ax, 4.75, 3.45, 4.7, 4.45, fc=DINING, ec=WALL, lw=2.2)
    card(ax, 0.4, 1.2, 9.05, 2.15, fc=HALL, ec=WALL, lw=2.2)
    for x, y, s in ((0.62, 7.6, "Kitchen"), (4.97, 7.6, "Dining room"), (0.62, 3.05, "Hallway")):
        ax.text(x, y, s, fontsize=17, color="#4A5568", fontweight="bold", va="center")
    card(ax, 0.9, 6.9, 3.45, 0.5, fc="#E2C9A6", ec="none", r=0.08)
    ax.text(0.95, 6.78, "counter", fontsize=12, color="#8A6D47", ha="left", va="top")
    card(ax, 7.45, 6.55, 1.85, 1.0, fc="#E9D8BE", ec="#A88B65", lw=1.6, r=0.08)
    ax.text(8.38, 7.68, "cabinet", fontsize=12, color="#8A6D47", ha="center")
    card(ax, 1.0, 1.55, 1.6, 0.65, fc="#E4DCF0", ec="#A99BC4", lw=1.4, r=0.08)
    ax.text(1.8, 1.87, "old drawer", fontsize=12.5, color="#6D5A8F", ha="center", va="center")
    card(ax, 0.4, 0.3, 9.05, 0.75, fc="white", ec="#D9DEE7", lw=1.4, r=0.16)          # narration bar
    # who does what, with check boxes that fill in
    boxes = {}
    for x0, name, col, dark, things in ((0.45, "Robot A", A_COL, A_DARK, ("plates", "glasses")),
                                        (5.0, "Robot B", B_COL, B_DARK, ("forks", "spoons"))):
        card(ax, x0, 8.18, 0.34, 0.34, fc=col, ec="none", r=0.08)
        ax.text(x0 + 0.48, 8.35, name, fontsize=16, fontweight="bold", color=dark, va="center")
        for k, th in enumerate(things):
            bx = x0 + 1.75 + 1.25 * k
            card(ax, bx, 8.2, 0.3, 0.3, fc="white", ec="#9AA5B1", lw=1.5, r=0.06)
            ax.text(bx + 0.42, 8.35, th, fontsize=15, color=INK, va="center")
            boxes[th] = ax.text(bx + 0.15, 8.36, "✓", fontsize=18, fontweight="bold", color=GOOD, ha="center",
                                va="center", alpha=0, zorder=5)
    # Team-Plan panel
    card(ax, 9.75, 0.3, 6.0, 8.45, fc="white", ec="#D9DEE7", lw=1.5, r=0.22)
    card(ax, 9.75, 8.0, 6.0, 0.75, fc=NAVY, ec="none", r=0.22)
    ax.add_patch(Rectangle((9.75, 8.0), 6.0, 0.3, fc=NAVY, ec="none", zorder=1))
    ax.text(10.0, 8.38, "Team-Plan, deciding for Robot B", fontsize=17, fontweight="bold", color="white", va="center", zorder=2)
    ax.text(10.0, 0.85, "Illustrative numbers.", fontsize=12, color=MUTED, va="center")
    ax.text(10.0, 0.55, "Team-Plan repeats these steps at every decision.", fontsize=12, color=MUTED, va="center")
    ax.plot([10.0, 15.5], [5.42, 5.42], color="#E5E9F0", lw=1.5, zorder=1)
    return ax, boxes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=int, default=10); ap.add_argument("--mp4", action="store_true")
    ap.add_argument("--font-dir", default=""); ap.add_argument("--out", default=OUT)
    ap.add_argument("--stills", default="", help="comma-separated times in seconds: save PNG stills and stop")
    a = ap.parse_args()
    for f in glob.glob(os.path.join(a.font_dir, "*.ttf")) if a.font_dir else []:
        font_manager.fontManager.addfont(f)
    plt.rcParams["font.family"] = ["Open Sans", "DejaVu Sans"]
    os.makedirs(a.out, exist_ok=True)
    clock = {"fps": a.fps}

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax, boxes = build(fig)
    rb, ra = robot(ax, B_COL, B_FACE), robot(ax, A_COL, A_FACE)
    lab_b = ax.text(0, 0, "Robot B", fontsize=14, fontweight="bold", color=B_DARK, ha="center", zorder=8)
    lab_a = ax.text(0, 0, "Robot A", fontsize=14, fontweight="bold", color=A_DARK, ha="center", zorder=8)
    items = {name: item(ax, name[:-1]) for name in SLOTS}
    dots = [Circle((0, 0), 0.07, fc=B_COL, ec="none", zorder=9, alpha=0) for _ in range(3)]
    for d in dots:
        ax.add_patch(d)

    # steps list
    step_rows = []
    for k, title in enumerate(STEPS):
        y = 7.6 - 0.47 * k
        bg = card(ax, 9.95, y - 0.21, 5.6, 0.42, fc=ACTIVE_BG, ec="none", z=1, r=0.1); bg.set_alpha(0)
        num = Circle((10.22, y), 0.17, fc=FAINT, ec="none", zorder=2); ax.add_patch(num)
        nt = ax.text(10.22, y - 0.01, str(k + 1), fontsize=13, fontweight="bold", color="white", ha="center", va="center", zorder=3)
        tt = ax.text(10.52, y, title, fontsize=15.5, color=MUTED, va="center", zorder=3)
        step_rows.append((bg, num, nt, tt))

    # cards: text lines, belief bars, gain bars
    text_lines = [ax.text(10.05, y, "", fontsize=16.5, color=INK, va="center", zorder=3) for y in (4.85, 4.3, 3.75)]
    note = ax.text(10.05, 2.95, "", fontsize=13, color=MUTED, va="center", zorder=3, style="italic", linespacing=1.4)
    bel_title = ax.text(10.05, 4.95, "Belief over hidden causes", fontsize=15.5, fontweight="bold", color=INK, va="center")
    bel_sub = ax.text(10.05, 4.55, "", fontsize=12.5, color=MUTED, va="center")
    bel = []
    for k, (lab, col) in enumerate(zip(CAUSES, CAUSE_COLS)):
        y = 3.95 - 0.62 * k
        t1 = ax.text(10.05, y, lab, fontsize=15, color=INK, va="center", zorder=3)
        tr = card(ax, 12.55, y - 0.15, 2.6, 0.3, fc=TRACK, ec="none", z=2, r=0.05)
        bar = Rectangle((12.55, y - 0.15), 0, 0.3, fc=col, ec="none", zorder=3); ax.add_patch(bar)
        v = ax.text(12.6, y, "", fontsize=14, color=INK, va="center", zorder=4, fontweight="bold")
        bel.append((t1, tr, bar, v))
    gain_title = ax.text(10.05, 4.95, "Expected gain for the whole task", fontsize=15.5, fontweight="bold", color=INK, va="center")
    gain_sub = ax.text(10.05, 4.55, "belief × benefit × importance − cost", fontsize=12.5, color=MUTED, va="center")
    gains = []
    for k, lab in enumerate(ACTIONS):
        y = 3.95 - 0.64 * k
        t1 = ax.text(10.05, y, lab, fontsize=15, color=INK, va="center", zorder=3)
        tr = card(ax, 11.75, y - 0.15, 3.4, 0.3, fc=TRACK, ec="none", z=2, r=0.05)
        bar = Rectangle((11.75, y - 0.15), 0, 0.3, fc=BAR_GREY, ec="none", zorder=3); ax.add_patch(bar)
        v = ax.text(11.8, y, "", fontsize=14, color=INK, va="center", zorder=4, fontweight="bold")
        gains.append((t1, tr, bar, v))
    gain_note = ax.text(11.75, 3.95 - 1.28 - 0.34, "", fontsize=12, color=MUTED, va="center", zorder=4)
    belief_group = [bel_title, bel_sub] + [x for row in bel for x in row]
    gain_group = [gain_title, gain_sub, gain_note] + [x for row in gains for x in row]

    narration = ax.text(4.92, 0.67, "", fontsize=16.5, color=INK, ha="center", va="center", zorder=9)
    fail = ax.text(2.2, 6.25, "✗ pick failed", fontsize=17, fontweight="bold", color=BAD, ha="left", va="center", zorder=10, alpha=0)
    bubble_box = FancyBboxPatch((0, 0), 1, 0.5, boxstyle="round,pad=0.08,rounding_size=0.16", fc="white", ec=INK, lw=2,
                                zorder=11, alpha=0)
    ax.add_patch(bubble_box)
    bubble_txt = ax.text(0, 0, "", fontsize=16, color=INK, ha="center", va="center", zorder=12, alpha=0)
    big = FancyBboxPatch((1.6, 2.2), 12.8, 4.6, boxstyle="round,pad=0.1,rounding_size=0.3", fc="white", ec="#CBD5E1",
                         lw=2, zorder=20, alpha=0)
    ax.add_patch(big)
    big_title = ax.text(8, 5.55, "", fontsize=32, fontweight="bold", color=NAVY, ha="center", va="center", zorder=21)
    big_sub = ax.text(8, 3.95, "", fontsize=18, color=INK, ha="center", va="center", zorder=21, linespacing=1.7)

    def where(name, t, pb, pa):
        for obj, who, t_pick, t_drop in CARRIES:
            if obj == name:
                if t < t_pick:
                    return SLOTS[name]
                if t < t_drop:
                    rp = pb if who == "b" else pa
                    return (rp[0] + (-0.1 if name[-1] == "0" else 0.1), rp[1] + 0.86)
                return CABINET[name]
        return SLOTS[name]

    def current_step(t):
        for t0, t1, k, c in STEP_TIMELINE:
            if t0 <= t < t1:
                return k, c, window(t, t0, t1, 0.3)
        return None, None, 0.0

    def frame(i):
        t = i / clock["fps"]
        pb, pa = path(t, PATH_B), path(t, PATH_A)
        if 9.6 <= t <= 25.6:
            pa = pa + np.array([0.06 * np.sin(2.2 * t), 0])
        if 27.4 <= t <= 29.2:
            pa = pa + np.array([0, 0.06 * np.sin(18 * t) * (1 - ramp(t, 28.6, 29.2))])
        rb.place(*pb); ra.place(*pa)
        lab_b.set_position((pb[0], pb[1] - 0.78)); lab_a.set_position((pa[0], pa[1] - 0.78))
        for name, spr in items.items():
            spr.place(*where(name, t, pb, pa))
        think = max([window(t, t0, t1, 0.3) for t0, t1 in THINKING] + [0])
        for j, d in enumerate(dots):                                # "thinking" dots beside Robot B
            d.center = (pb[0] - 0.95 + 0.22 * j, pb[1] + 0.15)
            d.set_alpha(think * (0.35 + 0.65 * (0.5 + 0.5 * np.sin(5 * t - 1.3 * j))))

        # steps list
        k, c, op = current_step(t)
        visited = {s for t0, t1, s, _ in STEP_TIMELINE if t >= t0}
        for j, (bg, num, nt, tt) in enumerate(step_rows):
            on = (j == k)
            bg.set_alpha(op if on else 0)
            num.set_facecolor(B_COL if on else (NAVY if j in visited else FAINT))
            tt.set_color(INK if (on or j in visited) else MUTED); tt.set_fontweight("bold" if on else "normal")

        # card contents
        for x in text_lines + [note] + belief_group + gain_group:
            x.set_alpha(0)
        if c in TEXT_CARDS:
            for x, s in zip(text_lines, TEXT_CARDS[c]):
                x.set_text(s); x.set_alpha(op)
                x.set_fontweight("bold" if s.startswith(("Best", "No help", "Help may")) else "normal")
            if c.startswith("clear"):
                note.set_text(CLOSE_NOTE); note.set_alpha(op)
        elif c == "belief":
            b = PRIOR * ramp(t, 14.6, 15.8) if t < 30 else PRIOR + (POSTERIOR - PRIOR) * ramp(t, 31.8, 33.4)
            bel_sub.set_text("from what Robot B has seen so far" if t < 30 else "after seeing a failed pick")
            for x in belief_group:
                x.set_alpha(op)
            for (_, _, bar, v), val in zip(bel, b):
                bar.set_width(2.6 * val); v.set_text(f"{val:.2f}")
                v.set_position((12.55 + 2.6 * val + 0.08, v.get_position()[1]))
        elif c == "gain":
            g = GAIN1 * ramp(t, 17.6, 18.8) if t < 30 else GAIN1 + (GAIN2 - GAIN1) * ramp(t, 34.7, 36.2)
            best = int(np.argmax(g))
            for x in gain_group:
                x.set_alpha(op)
            for j, ((_, _, bar, v), val) in enumerate(zip(gains, g)):
                bar.set_width(3.4 * val / 0.6); bar.set_facecolor(B_COL if j == best and val > 0.05 else BAR_GREY)
                v.set_text(f"{val:+.2f}"); v.set_position((11.75 + 3.4 * val / 0.6 + 0.08, v.get_position()[1]))
            gain_note.set_text("Robot B's own work is still open" if t < 30 else "Robot B's own work is done")

        for key, chk in boxes.items():
            chk.set_alpha(ramp(t, DONE[key], DONE[key] + 0.3))
        narration.set_alpha(0)
        for t0, t1, text in NARRATION:
            if t0 <= t < t1:
                narration.set_text(text); narration.set_alpha(window(t, t0, t1, 0.3))
        fail.set_alpha(window(t, 28.3, 31.6, 0.25))
        bubble_box.set_alpha(0); bubble_txt.set_alpha(0)
        for who, t0, t1, text in BUBBLES:
            if t0 <= t < t1:
                rp = path(t0, PATH_B if who == "b" else PATH_A)       # the bubble stays where it was spoken
                w = 0.15 * len(text) + 0.6
                x = min(max(rp[0], 0.75 + w / 2), 9.15 - w / 2)
                yb = rp[1] - 1.6 if rp[1] > 6.0 else rp[1] + 1.25
                bubble_box.set_bounds(x - w / 2, yb, w, 0.5)
                bubble_box.set_edgecolor(B_COL if who == "b" else A_COL)
                o = window(t, t0, t1, 0.2)
                bubble_box.set_alpha(o); bubble_txt.set_alpha(o)
                bubble_txt.set_text(text); bubble_txt.set_position((x, yb + 0.25))

        o_start, o_end = 1 - ramp(t, 3.4, 4.0), ramp(t, 51.4, 52.0)
        big.set_alpha(max(o_start, o_end))
        if t < 4.0:
            big_title.set_text("When should an AI agent step in?")
            big_sub.set_text("Two robots share a household tidying task.\n"
                             "Robot A puts away plates and glasses, Robot B forks and spoons.\n"
                             "Robot B uses Team-Plan to decide whether to help, and how.")
            big_title.set_alpha(o_start); big_sub.set_alpha(o_start)
        elif t >= 51.4:
            big_title.set_text("Team-Plan")
            big_sub.set_text("decides whether and how to step in\n"
                             "by reasoning about why the partner is stuck\n"
                             "github.com/HamedKhosravi99/team-plan")
            big_title.set_alpha(o_end); big_sub.set_alpha(o_end)
        else:
            big_title.set_alpha(0); big_sub.set_alpha(0)
        return []

    if a.stills:
        for ts in a.stills.split(","):
            clock["fps"] = 100; frame(int(round(float(ts) * 100)))
            fig.savefig(os.path.join(a.out, f"still_{float(ts):05.1f}.png"), dpi=75, facecolor=BG)
        return
    gif = os.path.join(a.out, "team_plan_demo.gif")
    save_gif(fig, frame, clock, a.fps, gif)
    print("wrote", gif)
    if a.mp4:
        import imageio_ffmpeg
        from matplotlib.animation import FFMpegWriter
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        clock["fps"] = 25
        mp4 = os.path.join(a.out, "team_plan_demo.mp4")
        FuncAnimation(fig, frame, frames=int(DURATION * 25), interval=40, blit=False).save(
            mp4, writer=FFMpegWriter(fps=25, bitrate=3000, extra_args=["-pix_fmt", "yuv420p"]), dpi=150)
        print("wrote", mp4)
    clock["fps"] = a.fps; frame(int(36.5 * a.fps))
    fig.savefig(os.path.join(a.out, "team_plan_demo_still.png"), dpi=100, facecolor=BG)


def save_gif(fig, frame, clock, fps: int, path_out: str, dpi: int = 75) -> None:
    """Write the GIF with one palette shared by all frames, so the light floor colours survive the 256-colour limit."""
    from PIL import Image

    def grab(t):
        frame(int(round(t * clock["fps"]))); fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        return Image.frombuffer("RGBA", (w, h), fig.canvas.buffer_rgba()).convert("RGB")

    old_dpi = fig.get_dpi(); fig.set_dpi(dpi); clock["fps"] = fps
    samples = [grab(t) for t in (2.0, 8.5, 16.5, 19.5, 22.5, 29.5, 33.0, 46.0, 54.0)]
    mosaic = Image.new("RGB", (samples[0].width * len(samples), samples[0].height))
    for k, im in enumerate(samples):
        mosaic.paste(im, (k * im.width, 0))
    palette = mosaic.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    frames = [grab(i / fps).quantize(palette=palette, dither=Image.Dither.NONE) for i in range(int(DURATION * fps))]
    frames[0].save(path_out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=False)
    fig.set_dpi(old_dpi)


if __name__ == "__main__":
    main()
