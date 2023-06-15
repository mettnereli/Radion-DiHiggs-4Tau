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
event = NanoEventsFactory.from_root(
    fname,
    schemaclass=NanoAODSchema.v6,
).events()

#define shortcut
events = event[(ak.num(event.Muon) > 0) & (ak.num(event.boostedTau) > 0)]
muons = events.Muon
boostedTaus = events.boostedTau


#mask cuts for all events
muon_mask =  ak.all(muons.pt > 30, axis=1) & ak.all(abs(muons.eta) < 2.5, axis=1) 
boostedTau_mask = ak.all(boostedTaus.pt > 30, axis=1) & ak.all(boostedTaus.rawDeepTau2017v2p1VSmu > .9, axis=1)
charge_mask = ak.any((muons[:,0].charge == 1) & (boostedTaus[:,0].charge == -1), axis=0) & ak.any((muons[:,0].charge == -1) & (boostedTaus[:,0].charge == 1), axis=0)
met_mask = events.MET.pt > 30

dr = events.Muon[:,0].delta_r(events.boostedTau[:,0])
dr_mask = dr > .1

#combine all cuts into one mask
mask = muon_mask & boostedTau_mask & charge_mask & dr_mask & met_mask

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
h = Hist(hist.axis.Regular(50,0,150,name="mass",label="GeV"))
h.fill(muTauVec.mass)
hep.histplot(h, w2=None, histtype = 'fill')
ax.set_title("Muon Vector + Boosted Tau Vector Mass")
ax.set_xlabel("Mass (GeV)")
fig.savefig("muTauVec_mass.png")
