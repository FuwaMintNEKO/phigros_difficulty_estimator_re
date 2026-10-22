"""Conversions between Phigros chart encodings.

Supported paths
---------------
``official -> RPE``   :func:`load_official` + :func:`write_rpe`
``RPE -> official``   :func:`load_rpe` + :func:`write_official`
``PEC/PE -> RPE``     :func:`load_pec` + :func:`pec_to_rpe`
``*-*-*``             :mod:`pgr2rpe.oracle` re-reads all three formats with an
                      independent, documentation-driven implementation and
                      compares them in format-neutral units.

The converter itself works on one canonical model whose coordinates follow the
official chart's physical definitions (:mod:`pgr2rpe.model`). PEC is the one
exception: it uses 29 easing curves that the official model cannot hold, so it
is mapped straight onto RPE (whose easing numbering is identical).
"""

from .model import Chart, Line, Note, Seg, Track
from .official_in import load_official
from .official_out import write_official
from .pec_in import PecChart, load_pec, pec_to_rpe
from .rpe_in import load_rpe
from .rpe_out import write_rpe
from .verify import compare_charts, ConversionError

__all__ = [
    'Chart', 'Line', 'Note', 'Seg', 'Track', 'PecChart',
    'load_official', 'write_official', 'load_rpe', 'write_rpe',
    'load_pec', 'pec_to_rpe',
    'compare_charts', 'ConversionError',
]

__version__ = '1.1.0'
