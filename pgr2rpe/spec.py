"""Physical constants and unit conversions for the three chart encodings.

The whole conversion hinges on one choice: every quantity is stored in the
canonical model in **official-chart physical units** (seconds, screen-normalised
position, degrees, Y/s). RPE and official JSON are then both "projections" of
that model, and all factors below are exact ratios of two documented unit
definitions, not empirical fits.

Official chart units (docs.lchzh.net/learning/phigros, "概念")
-------------------------------------------------------------
W = screen width, H = screen height (reference frame 1920x1080).

    1 X = 0.05625 W = 108 px      horizontal unit, Note.positionX
    1 Y = 0.6     H = 648 px      vertical unit, Note.floorPosition, speed events
    1 T = 1.875/BPM s             time unit, 32 T = 1 beat (128th note)

RPE units (Re:PhiEdit 1.5/1.7 manual + prpr/src/parse/rpe.rs)
------------------------------------------------------------
    RPE_WIDTH  = 1350             x in [-675, 675]
    RPE_HEIGHT = 900              y in [-450, 450], +y is up
    1 RPESpeed = 120 RPE px / s   (manual: "速度为 1 时每秒下落 120 个 Y 单位")

Derived factors
---------------
Position, normalised to [-1, 1] over the full screen (the space Phira uses and
the space both formats ultimately fill: official x=0.5 is the screen centre,
RPE x=0 is the screen centre):

    official 1 X -> 2 * 0.05625      = 0.1125      normalised
    RPE      1 px -> 2 / 1350        = 0.00148148  normalised
    => note positionX * 75.9375 = RPE px
       (1350 * 0.05625; the 75.0 used by the legacy toolchain is 1.25 % short)

    official 1 Y -> 2 * 0.6          = 1.2         normalised
    RPE      1 px -> 2 / 900         = 0.00222222  normalised
    => 1 Y = 540 RPE px

Speed:

    official 1 Y/s = 540 RPE px/s = 540 / 120 RPESpeed = 4.5 RPESpeed
    => RPE speed = official speed * 4.5
       (independently confirmed by prpr: SPEED_RATIO = 10/45/HEIGHT_RATIO and the
        official parser dividing by HEIGHT_RATIO, ratio = 4.5 exactly)

Rotation, alpha
---------------
    official rotate is degrees, counter-clockwise positive (doc).
    prpr parses RPE rotate events with factor -1 and official rotate with
    factor +1, so for identical visuals: rpe_deg = -official_deg.
    official disappear is 0..1 (<=0 transparent, >=1 opaque);
    prpr multiplies RPE alpha by 1/255 -> rpe_alpha = official_alpha * 255.

Time
----
    official tick -> seconds : t * 60 / (32 * line.bpm)
    seconds -> RPE beat      : s * global_bpm / 60

The RPE beat axis is global, so a chart whose judge lines carry different
``bpm`` values is converted by mapping every tick to seconds on its own line
first and then to a beat on the global axis. That keeps ``bpmfactor`` at 1.0,
which matters because the field cannot be edited in Re:PhiEdit and Phira
ignores it entirely (``// TODO bpmfactor`` in prpr/src/parse/rpe.rs).
"""

from __future__ import annotations

from fractions import Fraction
import math

# --- official chart units -------------------------------------------------

TICKS_PER_BEAT = 32.0
"""32 T = 1 beat; 1 T = 1.875 / BPM seconds."""

OFFICIAL_X_UNIT = 0.05625
"""1 X = 0.05625 * screen width."""

OFFICIAL_Y_UNIT = 0.6
"""1 Y = 0.6 * screen height."""

X_NORM_PER_OFFICIAL_UNIT = 2.0 * OFFICIAL_X_UNIT          # 0.1125
Y_NORM_PER_OFFICIAL_UNIT = 2.0 * OFFICIAL_Y_UNIT          # 1.2

# --- RPE geometry ---------------------------------------------------------

RPE_WIDTH = 1350.0
RPE_HEIGHT = 900.0

X_PX_PER_NORM = RPE_WIDTH / 2.0                           # 675
Y_PX_PER_NORM = RPE_HEIGHT / 2.0                          # 450

X_PX_PER_OFFICIAL_UNIT = X_NORM_PER_OFFICIAL_UNIT * X_PX_PER_NORM   # 75.9375
Y_PX_PER_OFFICIAL_UNIT = Y_NORM_PER_OFFICIAL_UNIT * Y_PX_PER_NORM   # 540.0

# --- speed ----------------------------------------------------------------

RPESPEED_PX_PER_SECOND = 120.0
OFFICIAL_SPEED_TO_RPE = Y_PX_PER_OFFICIAL_UNIT / RPESPEED_PX_PER_SECOND   # 4.5

# --- rotation -------------------------------------------------------------

RPE_ROTATE_SIGN = -1.0
"""rpe_degrees = RPE_ROTATE_SIGN * official_degrees."""

# --- alpha ----------------------------------------------------------------

RPE_ALPHA_MAX = 255.0

# --- note types -----------------------------------------------------------

OFFICIAL_TYPE_TO_NAME = {1: 'tap', 2: 'drag', 3: 'hold', 4: 'flick'}
RPE_TYPE_TO_OFFICIAL = {1: 1, 2: 3, 3: 4, 4: 2}
"""RPE: 1 Tap, 2 Hold, 3 Flick, 4 Drag  ->  official: 1 Tap, 2 Drag, 3 Hold, 4 Flick."""

OFFICIAL_TYPE_TO_RPE = {v: k for k, v in RPE_TYPE_TO_OFFICIAL.items()}

# --- format version of the emitted file -----------------------------------

DEFAULT_RPE_VERSION = 170

# --- tolerances -----------------------------------------------------------

EPS = 1e-9

MAX_TRIPLE_DENOMINATOR = 100_000
"""RPE stores times as [measure, beat, division] with unsigned beat/division, so
a beat value must be re-expressed as a fraction. 1e-5 beat at 200 BPM is 3 us."""


class FormatError(ValueError):
    """Raised when an input chart cannot be interpreted unambiguously."""


def tick_to_seconds(tick: float, bpm: float) -> float:
    """Official tick -> seconds for a judge line with the given BPM."""
    return float(tick) * 60.0 / (TICKS_PER_BEAT * float(bpm))


def seconds_to_tick(seconds: float, bpm: float) -> float:
    return float(seconds) * TICKS_PER_BEAT * float(bpm) / 60.0


def seconds_to_beat(seconds: float, bpm: float) -> float:
    return float(seconds) * float(bpm) / 60.0


def beat_to_seconds(beat: float, bpm: float) -> float:
    return float(beat) * 60.0 / float(bpm)


def to_triple(beat: float, max_denominator: int = MAX_TRIPLE_DENOMINATOR) -> list[int]:
    """Encode a beat value as RPE's ``[measure, beat, division]``.

    RPE deserialises the triple as ``(i32, u32, u32)``: the integer part may be
    negative but the numerator and denominator must be non-negative, so a value
    of -0.5 beats has to be written as ``[-1, 1, 2]``, never ``[0, -1, 2]``.
    """
    if not math.isfinite(beat):
        raise FormatError(f'non-finite beat value: {beat!r}')
    # Fast path: official charts are tick quantised, so almost every beat value
    # is an exact multiple of 1/32 and needs no continued fraction.
    scaled = beat * TICKS_PER_BEAT
    nearest = round(scaled)
    if abs(scaled - nearest) < 1e-9:
        measure, remainder = divmod(int(nearest), int(TICKS_PER_BEAT))
        return [measure, remainder, int(TICKS_PER_BEAT)]
    frac = Fraction(beat).limit_denominator(max_denominator)
    measure = math.floor(frac)
    rest = frac - measure
    numerator = int(rest * frac.denominator)
    denominator = int(frac.denominator)
    if numerator >= denominator:  # pragma: no cover - floor() keeps rest < 1
        measure += 1
        numerator -= denominator
    return [measure, numerator, denominator]


def from_triple(value) -> float:
    """Decode ``[measure, beat, division]`` (or a bare number) into beats."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        raise FormatError(f'beat value must be [measure, beat, division], got {value!r}')
    measure, numerator, denominator = (float(value[0]), float(value[1]), float(value[2]))
    if denominator == 0:
        raise FormatError('beat division must be non-zero')
    return measure + numerator / denominator
