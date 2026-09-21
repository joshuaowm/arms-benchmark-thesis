"""Where the third-party model repos live.

Set ARMS_THIRD_PARTY to the folder holding sam-hq/ and OSISeg/ (created by
third_party/setup_third_party.sh). The default is the cluster path the
experiments ran on. Other paths are still set inside the individual scripts.
"""
import os
from pathlib import Path

THIRD_PARTY = Path(os.environ.get("ARMS_THIRD_PARTY", "/share/castor/home/e2406747/axolotl/code"))
SAMHQ_ROOT  = THIRD_PARTY / "sam-hq"
SAM2_ROOT   = THIRD_PARTY / "sam-hq" / "sam-hq2"
OSISEG_ROOT = THIRD_PARTY / "OSISeg"


def add_third_party_to_path():
    """Put sam-hq (segment_anything), OSISeg (networks) and sam2 on sys.path."""
    import sys
    for p in (SAMHQ_ROOT, OSISEG_ROOT, SAM2_ROOT):
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
