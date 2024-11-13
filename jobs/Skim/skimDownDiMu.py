import sys
import os
from pathlib import Path
import awkward as ak
import numpy as np
import uproot
from coffea import nanoevents
import coffea.processor as processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.nanoevents.methods import nanoaod, vector

def is_rootcompat(a):
    """Is it a flat or 1-d jagged array?"""
    t = ak.type(a)
    if isinstance(t, ak._ext.ArrayType):
        if isinstance(t.type, ak._ext.PrimitiveType):
            return True
        if isinstance(t.type, ak._ext.ListType) and isinstance(t.type.type, ak._ext.PrimitiveType):
            return True
    return False


def uproot_writeable(events):
    """Restrict to columns that uproot can write compactly"""
    out = {}
    for bname in events.fields:
        if events[bname].fields:
            out[bname] = ak.zip(
                {
                    n: ak.packed(ak.without_parameters(events[bname][n]))
                    for n in events[bname].fields
                    if is_rootcompat(events[bname][n])
                }
            )
        else:
            out[bname] = ak.packed(ak.without_parameters(events[bname]))
    return out

filename = 'filename'
index = filename.index("NANO_NANO")
filename = filename[index:]
numroot = filename[10:]
events = NanoEventsFactory.from_root(
    filename,
    treepath="/Events",
).events()
count = len(events)
hcount = np.ones(count)
hcount_hist = np.histogram(hcount, bins=1)
genWeights = ak.sum(events.genWeight)
sumGenw = np.histogram(genWeights, bins=1)


events = events[(ak.num(events.Muon) > 1)]
events = events[events.HLT.Mu50]
leading = events.Muon[(events.Muon.pt > 50) & (np.abs(events.Muon.eta) < 2.4)]
subleading = events.Muon[(events.Muon.pt > 10) & (np.abs(events.Muon.eta) < 2.4)]
events = events[(ak.num(leading, axis=-1) > 0) & (ak.num(subleading, axis=-1) > 0)]

pairs = ak.cartesian({'i0': events.Muon, 'i1': events.Muon}, axis=1, nested=False) 
dr = pairs['i0'].delta_r(pairs['i1'])

pairs = pairs[(dr > .1) & (dr < .8)]
events = events[(ak.num(pairs, axis=-1) > 0)]

with uproot.recreate('NANO_NANO_l.root') as fout:   
    fout["Events"] = uproot_writeable(events)
    fout["hcount"] = hcount_hist
    fout["sumGenw"] = sumGenw
