import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("measure_delivery", ROOT / "scripts/measure_delivery.py")
md = importlib.util.module_from_spec(spec)
spec.loader.exec_module(md)

FPS = 15
ALERT = 2.0


def series(flash=True):
    """Gray levels of the phone's box, 0-8 s at 15 fps, as the l01 frames looked (11/3):
    a shadow over the phone just before the alert that recovers as a step, a slow rise
    while the person walks out of the frame, then (if flash) the two pulses of the flash
    notification 2.4 s after the alert."""
    times, levels = [], []
    for i in range(8 * FPS):
        t = i / FPS
        lv = 70.0
        if 1.6 <= t < ALERT:
            lv -= 40          # the shadow
        if ALERT <= t < 4.0:
            lv += (t - ALERT) * 15  # the slow rise, +30 over 2 s
        if flash and (4.4 <= t < 4.8 or 5.07 <= t < 5.47):
            lv += 42          # the two pulses
        times.append(t)
        levels.append(lv)
    return times, levels


class FirstLit(unittest.TestCase):
    def test_blink_finds_the_first_pulse_not_the_shadow_recovery(self):
        times, levels = series()
        i = md.first_lit(times, levels, ALERT, 25, blink=True)
        self.assertAlmostEqual(times[i], 4.4, places=2)

    def test_without_blink_the_recovery_step_counts(self):
        times, levels = series()
        i = md.first_lit(times, levels, ALERT, 25, blink=False)
        self.assertAlmostEqual(times[i], ALERT, places=2)

    def test_no_pulse_is_not_seen(self):
        times, levels = series(flash=False)
        self.assertIsNone(md.first_lit(times, levels, ALERT, 25, blink=True))

    def test_frames_before_the_alert_are_ignored(self):
        times, levels = series()
        self.assertIsNone(md.first_lit(times, levels, 6.0, 25, blink=True))


if __name__ == "__main__":
    unittest.main()
