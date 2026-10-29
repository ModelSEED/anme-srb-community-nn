"""Shared path resolution — every script locates the version's model/data/surrogate
directories relative to its own location, so the repo is runnable from any checkout."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))       # versions/vN/code
VDIR = os.path.dirname(HERE)                             # versions/vN
MODELDIR = os.path.join(VDIR, "model")
DATADIR = os.path.join(VDIR, "data")
SURRDIR = os.path.join(VDIR, "surrogate")
MODEL = os.path.join(MODELDIR, "community_ANME2a_SRB2_AOM_final_N2open.json")
