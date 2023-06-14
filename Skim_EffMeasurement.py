import awkward as ak
import uproot
import hist
from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.nanoevents.methods import vector


#read in file
fname = "./NANO_NANO_3.root"
events = NanoEventsFactory.from_root(
    fname,
    schemaclass=NanoAODSchema.v6,
).events()

#define shortcut
muons = events.Muon
boostedTaus = events.boostedTau


#mask cuts for all events
muon_mask = (ak.num(muons) > 0) & ak.all(muons.pt > 30, axis=1) & ak.all(abs(muons.eta) < 2.5, axis=1) 
boostedTau_mask = (ak.num(boostedTaus) > 0) & ak.all(boostedTaus.pt > 30, axis=1)


dr = events[(ak.num(muons) > 0) & (ak.num(boostedTaus) > 0)].Muon[:,0].delta_r(events[(ak.num(muons) > 0) & (ak.num(boostedTaus) > 0)].boostedTau[:,0])
dr_mask = ak.all(dr > .1, axis = 0)

#combine all cuts into one mask
mask = muon_mask & boostedTau_mask & dr_mask

selected_events = events[mask]

#make 4-vectors
muVec = ak.zip(
   {
    "pt": selected_events.Muon[:,0].pt,
    "eta": selected_events.Muon[:,0].eta,
    "phi": selected_events.Muon[:,0].phi,
    "mass": selected_events.Muon[:,0].mass,
   },
   with_name="PtEtaPhiMLorentzVector",
   behavior=vector.behavior,
)

boostedTauVec = ak.zip(
   {
    "pt": selected_events.boostedTau[:,0].pt,
    "eta": selected_events.boostedTau[:,0].eta,
    "phi": selected_events.boostedTau[:,0].phi,
    "mass": selected_events.boostedTau[:,0].mass,
   },
   with_name="PtEtaPhiMLorentzVector",
   behavior=vector.behavior,
)

#combine 4-vectors
muTauVec = muVec.add(boostedTauVec)



#plot and save
fig, ax = plt.subplots()
h = Hist(hist.axis.Regular(50,0,2,name="mass",label="GeV"))
h.fill(muTauVec.mass)
hep.histplot(h)
ax.set_title("Muon Vector + Boosted Tau Vector Mass")
ax.set_xlabel("Mass (GeV)")
fig.savefig("muTauVec_mass.png")
