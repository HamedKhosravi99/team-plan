"""Animated illustration of the paper's running example: two robots tidy a kitchen, the partner leaves the plates
untouched, and Team-Plan updates its belief about why before choosing how to help.

The animation is a schematic with illustrative numbers, not a recording of the simulator.

    python docs/make_animation.py                       # docs/assets/team_plan_demo.gif
    python docs/make_animation.py --mp4                 # also docs/assets/team_plan_demo.mp4 (needs imageio-ffmpeg)
    python docs/make_animation.py --font-dir /path/to/fonts   # use the Open Sans files in that folder
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
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

TP, WN, GREY, INK, BG, RED, GREEN = "#1F4E79", "#E08E0B", "#9B9B9B", "#222222", "#FBFAF6", "#C0392B", "#2E7D32"
LIGHT_TP, LIGHT_WN = "#DCE6F1", "#FBE9CF"
DURATION = 25.0
ROWS_Y = [4.85, 4.25, 3.65, 3.05]   # rows of the intervention scores
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


# ------------------------------------------------------------------------------------------------ timing helpers
def ease(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def ramp(t: float, t0: float, t1: float) -> float:
    """0 before t0, 1 after t1, smooth in between."""
    return ease((t - t0) / (t1 - t0)) if t1 > t0 else float(t >= t0)


def window(t: float, t0: float, t1: float, fade: float = 0.35) -> float:
    """Opacity of an element shown between t0 and t1, with short fades."""
    return min(ramp(t, t0, t0 + fade), 1 - ramp(t, t1 - fade, t1))


def path(t: float, keys) -> np.ndarray:
    """Position along waypoints [(time, (x, y)), ...], holding still between consecutive equal points."""
    if t <= keys[0][0]:
        return np.array(keys[0][1], float)
    for (t0, p0), (t1, p1) in zip(keys, keys[1:]):
        if t <= t1:
            return np.array(p0, float) + ramp(t, t0, t1) * (np.array(p1, float) - np.array(p0, float))
    return np.array(keys[-1][1], float)


def lerp(a, b, u):
    return a + (b - a) * u


# ------------------------------------------------------------------------------------------------ the story
# counter positions of the objects, robots stop just below the counter (y = 6.05) to pick up
SLOTS = {"plate0": (1.30, 6.72), "plate1": (1.62, 6.72), "glass0": (2.22, 6.72), "glass1": (2.48, 6.72),
         "fork0": (3.05, 6.72), "fork1": (3.27, 6.72), "spoon0": (3.78, 6.72), "spoon1": (4.0, 6.72)}
CABINET = {"fork0": (7.78, 6.72), "fork1": (7.98, 6.72), "glass0": (8.28, 6.72), "glass1": (8.52, 6.72),
           "spoon0": (8.82, 6.72), "spoon1": (9.02, 6.72), "plate0": (8.1, 6.36), "plate1": (8.6, 6.36)}

TEAMPLAN_PATH = [(2.6, (3.9, 4.4)), (3.7, (3.15, 6.05)), (4.0, (3.15, 6.05)), (5.6, (7.9, 5.95)), (5.9, (7.9, 5.95)),
                 (7.1, (3.9, 6.05)), (7.4, (3.9, 6.05)), (8.3, (4.75, 5.0)), (10.6, (4.75, 5.0)),
                 (12.4, (8.9, 5.95)), (17.6, (8.9, 5.95)), (19.2, (1.55, 6.05)), (19.6, (1.55, 6.05)),
                 (21.3, (8.35, 5.95))]
PARTNER_PATH = [(2.6, (2.0, 4.6)), (3.6, (2.35, 6.05)), (3.9, (2.35, 6.05)), (5.8, (8.4, 5.95)), (6.1, (8.4, 5.95)),
                (7.6, (3.0, 4.55)), (11.6, (3.0, 4.55)), (13.0, (1.46, 6.05)), (17.4, (1.46, 6.05)),
                (18.6, (1.1, 4.7))]
# (object, robot, pick time, drop time)
CARRIES = [("fork0", "tp", 4.0, 5.6), ("fork1", "tp", 4.0, 5.6), ("glass0", "pa", 3.9, 5.8),
           ("glass1", "pa", 3.9, 5.8), ("spoon0", "tp", 7.4, 12.4), ("spoon1", "tp", 7.4, 12.4),
           ("plate0", "tp", 19.6, 21.3), ("plate1", "tp", 19.6, 21.3)]
DONE = {"glasses": 5.8, "forks": 5.6, "spoons": 12.4, "plates": 21.3}

CAUSES = ["forgot the plates", "wrong location (old drawer)", "weak grip"]
PRIOR, POSTERIOR = np.array([0.55, 0.30, 0.15]), np.array([0.08, 0.05, 0.87])
ACTIONS = ["remind", "tell where", "take over", "continue"]
SCORES_BEFORE, SCORES_AFTER = np.array([0.31, 0.16, 0.12, 0.0]), np.array([0.02, 0.01, 0.52, 0.0])
NOTES_BEFORE, NOTES_AFTER = ["", "", "own work still open", ""], ["", "", "own work done", ""]

NARRATION = [(2.6, 7.4, "Each robot puts away its own items."),
             (7.4, 10.0, "The plates stay untouched, and the cause is hidden."),
             (10.0, 13.2, "The partner most likely forgot, so a reminder is worth the most."),
             (13.2, 16.6, "The partner tries but cannot pick up the plates. A weak grip is now most likely."),
             (16.6, 21.4, "Now taking over gives the largest gain for the whole task."),
             (21.4, DURATION, "All four assignments are put away.")]
BUBBLES = [("tp", 10.0, 11.8, "Please put away the plates."), ("pa", 11.2, 12.8, "On it!"),
           ("tp", 16.6, 18.4, "I'll take over the plates.")]


# ------------------------------------------------------------------------------------------------ sprites
class Sprite:
    """A group of patches drawn around the origin and moved together."""

    def __init__(self, ax, patches, z: float):
        self.ax, self.patches = ax, patches
        for p in patches:
            p.set_zorder(z); ax.add_patch(p)

    def place(self, x: float, y: float) -> None:
        tr = transforms.Affine2D().translate(x, y) + self.ax.transData
        for p in self.patches:
            p.set_transform(tr)


def robot(ax, color: str, light: str) -> Sprite:
    parts = [Ellipse((0, -0.34), 0.72, 0.14, fc=(0, 0, 0, 0.12), ec="none"),
             FancyBboxPatch((-0.3, -0.27), 0.6, 0.52, boxstyle="round,pad=0.02,rounding_size=0.12", fc=color, ec="none"),
             FancyBboxPatch((-0.21, -0.02), 0.42, 0.2, boxstyle="round,pad=0.01,rounding_size=0.08", fc=light, ec="none"),
             Circle((-0.09, 0.08), 0.045, fc=INK, ec="none"), Circle((0.09, 0.08), 0.045, fc=INK, ec="none"),
             Rectangle((-0.012, 0.25), 0.024, 0.13, fc=color, ec="none"), Circle((0, 0.4), 0.05, fc=light, ec=color, lw=1.5),
             Circle((-0.2, -0.3), 0.06, fc="#555555", ec="none"), Circle((0.2, -0.3), 0.06, fc="#555555", ec="none")]
    return Sprite(ax, parts, 6)


def item(ax, kind: str) -> Sprite:
    if kind == "plate":
        parts = [Circle((0, 0), 0.15, fc="white", ec="#8A8A8A", lw=1.2), Circle((0, 0), 0.085, fc="none", ec="#BDBDBD", lw=0.9)]
    elif kind == "glass":
        parts = [Polygon([(-0.08, 0.13), (0.08, 0.13), (0.055, -0.12), (-0.055, -0.12)], closed=True, fc="#CFE8F6",
                         ec="#5B8DB8", lw=1.1)]
    elif kind == "fork":
        parts = [Rectangle((-0.012, -0.15), 0.024, 0.2, fc="#6D6D6D", ec="none")]
        parts += [Rectangle((dx - 0.009, 0.04), 0.018, 0.1, fc="#6D6D6D", ec="none") for dx in (-0.045, 0, 0.045)]
        parts += [Rectangle((-0.054, 0.03), 0.108, 0.025, fc="#6D6D6D", ec="none")]
    else:
        parts = [Rectangle((-0.012, -0.15), 0.024, 0.17, fc="#6D6D6D", ec="none"), Ellipse((0, 0.07), 0.08, 0.13, fc="#6D6D6D", ec="none")]
    return Sprite(ax, parts, 7)


# ------------------------------------------------------------------------------------------------ the scene
def build(fig):
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")
    fig.patch.set_facecolor(BG)
    room = dict(boxstyle="round,pad=0,rounding_size=0.18", lw=1.4, ec="#C9C4B8")
    ax.add_patch(FancyBboxPatch((0.5, 3.55), 4.15, 3.95, fc="#F3EFE6", **room))
    ax.add_patch(FancyBboxPatch((4.75, 3.55), 4.7, 3.95, fc="#EEF1EC", **room))
    ax.add_patch(FancyBboxPatch((0.5, 0.95), 8.95, 2.45, fc="#F1F0EC", **room))
    for x, y, s in ((0.7, 7.15, "Kitchen"), (4.95, 7.15, "Dining room"), (0.7, 3.05, "Hallway")):
        ax.text(x, y, s, fontsize=13, color="#8C8577", fontweight="bold", va="center")
    ax.add_patch(FancyBboxPatch((1.0, 6.5), 3.25, 0.45, boxstyle="round,pad=0,rounding_size=0.06", fc="#D9CFBF", ec="none"))
    ax.text(2.62, 6.38, "counter", fontsize=9, color="#8C8577", ha="center", va="top")
    ax.add_patch(FancyBboxPatch((7.5, 6.15), 1.75, 0.95, boxstyle="round,pad=0,rounding_size=0.06", fc="#E2D6C3", ec="#B9A98F", lw=1.2))
    ax.text(8.38, 7.24, "cabinet", fontsize=9, color="#8C8577", ha="center")
    ax.add_patch(FancyBboxPatch((1.0, 1.25), 1.3, 0.55, boxstyle="round,pad=0,rounding_size=0.06", fc="#E7E1D6", ec="#C9C4B8", lw=1.0))
    ax.text(1.65, 1.52, "old drawer", fontsize=9, color="#8C8577", ha="center", va="center")

    # right panel
    ax.add_patch(FancyBboxPatch((9.85, 0.35), 5.85, 8.3, boxstyle="round,pad=0,rounding_size=0.2", fc="white", ec="#E0DCD2", lw=1.2))
    ax.text(10.15, 8.25, "Team-Plan's view", fontsize=15, fontweight="bold", color=TP, va="center")
    ax.text(15.45, 8.25, "illustrative numbers", fontsize=9, color=GREY, ha="right", va="center")
    ax.text(10.15, 7.6, "Why are the plates untouched?", fontsize=11.5, fontweight="bold", color=INK, va="center")
    ax.text(10.15, 5.35, "Expected gain in full-task success", fontsize=11.5, fontweight="bold", color=INK, va="center")
    ax.text(10.15, 2.55, "Assignments", fontsize=11.5, fontweight="bold", color=INK, va="center")
    return ax


def panel_bars(ax, labels, ys, x0, width, color):
    bars, vals, notes = [], [], []
    for lab, y in zip(labels, ys):
        ax.text(10.2, y, lab, fontsize=10.5, color=INK, va="center")
        ax.add_patch(Rectangle((x0, y - 0.13), width, 0.26, fc="#F1F1EE", ec="none", zorder=2))
        b = Rectangle((x0, y - 0.13), 0, 0.26, fc=color, ec="none", zorder=3); ax.add_patch(b); bars.append(b)
        vals.append(ax.text(x0 + 0.05, y, "", fontsize=10, color=INK, va="center", zorder=4))
        notes.append(ax.text(x0, y - 0.34, "", fontsize=8.5, color=GREY, va="center", zorder=4))
    return bars, vals, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=int, default=12); ap.add_argument("--mp4", action="store_true")
    ap.add_argument("--font-dir", default=""); ap.add_argument("--out", default=OUT)
    ap.add_argument("--stills", default="", help="comma-separated times in seconds: save PNG stills and stop")
    a = ap.parse_args()
    clock = {"fps": a.fps}
    for f in glob.glob(os.path.join(a.font_dir, "*.ttf")) if a.font_dir else []:
        font_manager.fontManager.addfont(f)
    plt.rcParams["font.family"] = ["Open Sans", "DejaVu Sans"]
    os.makedirs(a.out, exist_ok=True)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = build(fig)
    tp, pa = robot(ax, TP, LIGHT_TP), robot(ax, WN, LIGHT_WN)
    tp_label = ax.text(0, 0, "Team-Plan", fontsize=9.5, fontweight="bold", color=TP, ha="center", zorder=8)
    pa_label = ax.text(0, 0, "Partner", fontsize=9.5, fontweight="bold", color=WN, ha="center", zorder=8)
    items = {name: item(ax, name[:-1]) for name in SLOTS}
    bel_bars, bel_vals, _ = panel_bars(ax, CAUSES, [7.1, 6.55, 6.0], 13.05, 2.3, TP)
    act_bars, act_vals, act_notes = panel_bars(ax, ACTIONS, ROWS_Y, 12.0, 3.3, GREY)
    best = FancyBboxPatch((0, 0), 1, 0.4, boxstyle="round,pad=0.02,rounding_size=0.08", fc="none", ec=TP, lw=2, zorder=5, alpha=0)
    ax.add_patch(best)
    rows = [("plates", "partner", WN), ("glasses", "partner", WN), ("forks", "Team-Plan", TP), ("spoons", "Team-Plan", TP)]
    checks = []
    for i, (what, who, col) in enumerate(rows):
        y = 2.05 - 0.42 * i
        ax.add_patch(FancyBboxPatch((10.22, y - 0.13), 0.26, 0.26, boxstyle="round,pad=0,rounding_size=0.05", fc="white", ec="#B8B8B8", lw=1.2, zorder=3))
        ax.text(10.65, y, what, fontsize=10.5, color=INK, va="center")
        ax.text(11.9, y, who, fontsize=10, color=col, va="center", fontweight="bold")
        checks.append(ax.text(10.35, y + 0.01, "✓", fontsize=14, color=GREEN, ha="center", va="center", zorder=4, alpha=0))
    narration = ax.text(5.0, 0.5, "", fontsize=12.5, color=INK, ha="center", va="center", zorder=9)
    fail = ax.text(1.46, 5.15, "✗ pick failed", fontsize=13, fontweight="bold", color=RED, ha="center", va="center", zorder=10, alpha=0)
    bubble_box = FancyBboxPatch((0, 0), 1, 0.5, boxstyle="round,pad=0.08,rounding_size=0.15", fc="white", ec=INK, lw=1.2, zorder=11, alpha=0)
    ax.add_patch(bubble_box)
    bubble_txt = ax.text(0, 0, "", fontsize=12.5, color=INK, ha="center", va="center", zorder=12, alpha=0)
    card = FancyBboxPatch((2.2, 2.7), 11.6, 3.6, boxstyle="round,pad=0.1,rounding_size=0.3", fc="white", ec="#D8D3C8", lw=1.5, zorder=20, alpha=0)
    ax.add_patch(card)
    card_title = ax.text(8, 5.2, "", fontsize=26, fontweight="bold", color=TP, ha="center", va="center", zorder=21)
    card_sub = ax.text(8, 4.05, "", fontsize=14, color=INK, ha="center", va="center", zorder=21, linespacing=1.6)

    def where(name: str, t: float, pos_tp, pos_pa):
        for obj, who, t_pick, t_drop in CARRIES:
            if obj == name:
                if t < t_pick:
                    return SLOTS[name]
                if t < t_drop:
                    rp = pos_tp if who == "tp" else pos_pa
                    dx = {"0": -0.09, "1": 0.09}[name[-1]]
                    return (rp[0] + dx, rp[1] + 0.8)
                return CABINET[name]
        return SLOTS[name]

    def frame(i: int):
        t = i / clock["fps"]
        p_tp, p_pa = path(t, TEAMPLAN_PATH), path(t, PARTNER_PATH)
        if 7.6 <= t <= 11.6:                                   # the stalled partner hesitates on the spot
            p_pa = p_pa + np.array([0.05 * np.sin(2.4 * t), 0])
        if 13.2 <= t <= 14.6:                                  # a failed grasp, the robot jerks
            p_pa = p_pa + np.array([0, 0.05 * np.sin(18 * t) * (1 - ramp(t, 14.2, 14.6))])
        tp.place(*p_tp); pa.place(*p_pa)
        tp_label.set_position((p_tp[0], p_tp[1] - 0.62)); pa_label.set_position((p_pa[0], p_pa[1] - 0.62))
        for name, spr in items.items():
            spr.place(*where(name, t, p_tp, p_pa))

        # belief and scores
        b = lerp(PRIOR, POSTERIOR, ramp(t, 14.6, 16.0)) * ramp(t, 8.0, 9.0)
        for bar, val, v in zip(bel_bars, bel_vals, b):
            bar.set_width(2.3 * v); val.set_position((13.05 + 2.3 * v + 0.07, val.get_position()[1]))
            val.set_text(f"{v:.2f}" if t >= 8.0 else "")
        s = lerp(SCORES_BEFORE, SCORES_AFTER, ramp(t, 15.6, 16.6)) * ramp(t, 9.0, 10.0)
        k = int(np.argmax(s))
        for j, (bar, val, note, v) in enumerate(zip(act_bars, act_vals, act_notes, s)):
            bar.set_width(3.3 * v / 0.6); bar.set_facecolor(TP if (j == k and t >= 9.6) else GREY)
            val.set_position((12.0 + 3.3 * v / 0.6 + 0.07, val.get_position()[1]))
            val.set_text(f"{v:+.2f}" if t >= 9.0 else "")
            note.set_text((NOTES_AFTER if t >= 16.0 else NOTES_BEFORE)[j] if t >= 9.0 else "")
        y = ROWS_Y[k]
        best.set_bounds(10.1, y - 0.2, 5.5, 0.4); best.set_alpha(0.9 * ramp(t, 9.6, 10.0))

        for chk, key in zip(checks, ["plates", "glasses", "forks", "spoons"]):
            chk.set_alpha(ramp(t, DONE[key], DONE[key] + 0.3))
        for t0, t1, text in NARRATION:
            if t0 <= t < t1:
                narration.set_text(text); narration.set_alpha(window(t, t0, t1, 0.3))
        fail.set_alpha(window(t, 13.8, 16.0, 0.25))
        bubble_box.set_alpha(0); bubble_txt.set_alpha(0)
        for who, t0, t1, text in BUBBLES:
            if t0 <= t < t1:
                rp = p_tp if who == "tp" else p_pa
                w = 0.125 * len(text) + 0.5
                x = min(max(rp[0], 0.7 + w / 2), 9.25 - w / 2)
                yb = rp[1] - 1.45 if rp[1] > 5.5 else rp[1] + 1.15        # below robots at the counter or cabinet
                bubble_box.set_bounds(x - w / 2, yb, w, 0.46)
                bubble_box.set_edgecolor(TP if who == "tp" else WN)
                op = window(t, t0, t1, 0.2)
                bubble_box.set_alpha(op); bubble_txt.set_alpha(op)
                bubble_txt.set_text(text); bubble_txt.set_position((x, yb + 0.23))

        op_start, op_end = 1 - ramp(t, 2.0, 2.6), ramp(t, 22.4, 23.0)
        card.set_alpha(max(op_start, op_end))
        if t < 2.6:
            card_title.set_text("When should an AI agent step in?")
            card_sub.set_text("Two robots share a household tidying task.\n"
                              "The partner puts away plates and glasses, Team-Plan's robot forks and spoons.")
            card_title.set_alpha(op_start); card_sub.set_alpha(op_start)
        elif t >= 22.4:
            card_title.set_text("Team-Plan")
            card_sub.set_text("Belief over the partner's problem, interventions scored by their effect on the whole task.\n"
                              "github.com/HamedKhosravi99/team-plan")
            card_title.set_alpha(op_end); card_sub.set_alpha(op_end)
        else:
            card_title.set_alpha(0); card_sub.set_alpha(0)
        return []

    if a.stills:
        for ts in a.stills.split(","):
            clock["fps"] = 100; frame(int(round(float(ts) * 100)))
            fig.savefig(os.path.join(a.out, f"still_{float(ts):05.1f}.png"), dpi=62.5, facecolor=BG)
        return
    frames = int(DURATION * a.fps)
    anim = FuncAnimation(fig, frame, frames=frames, interval=1000 / a.fps, blit=False)
    gif = os.path.join(a.out, "team_plan_demo.gif")
    anim.save(gif, writer=PillowWriter(fps=a.fps), dpi=62.5)
    print("wrote", gif)
    if a.mp4:
        import imageio_ffmpeg
        from matplotlib.animation import FFMpegWriter
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        clock["fps"] = 25
        mp4 = os.path.join(a.out, "team_plan_demo.mp4")
        FuncAnimation(fig, frame, frames=int(DURATION * 25), interval=40, blit=False).save(
            mp4, writer=FFMpegWriter(fps=25, bitrate=2400, extra_args=["-pix_fmt", "yuv420p"]), dpi=100)
        print("wrote", mp4)
    # a still from the decision moment, for pages that do not play GIFs
    clock["fps"] = a.fps; frame(int(16.8 * a.fps)); fig.savefig(os.path.join(a.out, "team_plan_demo_still.png"), dpi=100, facecolor=BG)


if __name__ == "__main__":
    main()
