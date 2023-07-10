import sys
import math
import awkward as ak
import uproot
import boost_histogram as bh
import hist
from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep
from coffea import processor, nanoevents
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.nanoevents.methods import vector

class MyProcessor(processor.ProcessorABC):
   def __init__(self):
      pass

   def makeVector(self, particle):
      newVec = ak.zip(
        {
            "pt": particle.pt,
            "eta": particle.eta,
            "phi": particle.phi,
            "mass": particle.mass,
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
      )
      return newVec

   def process(self,events):
      dataset = events.metadata['dataset']

      #define shortcuts
      events = events[(ak.num(events.Muon) > 0) & (ak.num(events.boostedTau) > 0)]
      muons = events.Muon
      boostedTaus = events.boostedTau
      IsoLep1Value = muons.pfRelIso04_all/muons.pt
     
      #mask cuts for all events
      muon_mask =  (ak.all(muons.pt > 28, axis=1) 
                    & ak.all(abs(muons.eta) <= 2.5, axis=1) 
                    & ak.all(muons.pfRelIso04_all < 1, axis=1)
                    & ak.all(muons.dxy >= .0001, axis=1) 
                    & ak.all(muons.dz >= .0001, axis=1)
                    & ak.all(IsoLep1Value <= 30, axis=1))
      
      boostedTau_mask =  (ak.all(boostedTaus.pt > 20, axis=1) 
                         & ak.all(abs(boostedTaus.eta) <= 2.5, axis=1)
                         & ak.all(boostedTaus.rawIsodR03 > .7, axis=1))
      met_mask = events.MET.pt > 30
      dr = events.Muon[:,0].delta_r(events.boostedTau[:,0])
      dr_mask = ak.any(dr > .1, axis=0) & ak.any(dr < .8, axis=0)
      
      #combine all cuts into one mask
      mask = muon_mask & boostedTau_mask & met_mask  & dr_mask
      selected_events = events[mask]

      signs = (selected_events.Muon[:,0].charge) * (selected_events.boostedTau[:,0].charge)
      VVLooseNum = ak.all(selected_events.boostedTau.rawMVAnewDM2017v2 > .98, axis =1)
      
      print("Number of Leading Boosted Tau Before: ")
      print(ak.num(ak.flatten(events.boostedTau.pt, axis=None), axis=0))

      print("Number of Leading Boosted Tau After: ")
      print(ak.num(ak.flatten(selected_events.boostedTau.pt, axis=None), axis=0))

      print("Same sign: ", ak.num(ak.flatten(selected_events.boostedTau[signs == 1].pt, axis=None), axis=0))
      print("Opposite sign: ", ak.num(ak.flatten(selected_events.boostedTau[signs == -1].pt, axis=None), axis=0))

      muonVec = self.makeVector(selected_events.Muon[:,0])
      tauVec = self.makeVector(selected_events.boostedTau[:,0])
      muTauVec = muonVec.add(tauVec)

      vec_pt = Hist.new.Regular(100,300,800, name='vec_pt', label="$p_T$ (GeV)").Double()
      vec_pt.fill(vec_pt = muTauVec.pt)

      vec_mass = Hist.new.Regular(100,0,400, name='vec_mass', label="Mass(GeV)").Double()
      vec_mass.fill(vec_mass = muTauVec.mass)

      boosted_pt = Hist.new.Regular(10,0,500, name="pt", label ="$p_T$ (GeV)").StrCat(["opposite", "same"], name="sign", label = "Sign").StrCat(["denominator", "numerator"], name="fraction", label="Fraction").Double()
      boosted_pt.fill(sign="opposite", fraction="denominator", pt = ak.flatten(selected_events.boostedTau[signs == -1][:,0].pt, axis=None))
      boosted_pt.fill(sign="opposite", fraction="numerator", pt = ak.flatten(selected_events.boostedTau[signs == -1 & VVLooseNum][:,0].pt, axis=None))
      boosted_pt.fill(sign="same", fraction="denominator", pt = ak.flatten(selected_events.boostedTau[signs == 1][:,0].pt, axis=None))
      boosted_pt.fill(sign="same", fraction="numerator", pt = ak.flatten(selected_events.boostedTau[signs == 1 & VVLooseNum][:,0].pt, axis=None))

      boosted_pt_rate_os = boosted_pt[:,"opposite","numerator"].view() / (boosted_pt[:,"opposite","denominator"]).view()
      boosted_pt_rate_ss = boosted_pt[:,"same","numerator"].view() / (boosted_pt[:,"same","denominator"]).view()

      boosted_pt_fakeRate = Hist.new.Regular(10, 0, 500, name="fakerate", label="$p_T (GeV)").StrCat(["opposite", "same"], name="sign", label = "Sign").Double()
      boosted_pt_fakeRate.fill(sign="opposite", fakerate=boosted_pt_rate_os)
      boosted_pt_fakeRate.fill(sign="same", fakerate=boosted_pt_rate_ss)
   

      return {
         dataset: {
            "entries": len(events),
            "events": selected_events,
            "pT": boosted_pt,
            "fakeRate": boosted_pt_fakeRate,
            "vec_pt": vec_pt,
            "vec_mass": vec_mass,
         }
      }
   
   def postprocess(self, accumulator):
      pass
   

if __name__ == "__main__":
   dataset = sys.argv[1]
   #read in file
   fname = "./DY.root"
   events = NanoEventsFactory.from_root(
      fname,
      schemaclass=NanoAODSchema.v6,
      metadata={"dataset": dataset},
   ).events()

   p = MyProcessor()
   out = p.process(events)



   #plot and save

   fig, axs = plt.subplots(2, 2, figsize=(20, 20))
   out[dataset]["pT"][:,"opposite","denominator"].plot1d(ax=axs[0, 0])
   out[dataset]["pT"][:,"same","denominator"].plot1d(ax=axs[0, 1])
   out[dataset]["pT"][:,"opposite","numerator"].plot1d(ax=axs[1, 0])
   out[dataset]["pT"][:,"same","numerator"].plot1d(ax=axs[1, 1])

   axs[0, 0].set_title("OS_Denom_$p_T$")
   axs[0, 1].set_title("SS_Denom_$p_T$")
   axs[1, 0].set_title("OS_Numerator_$p_T$")
   axs[1, 1].set_title("SS_Numerator_$p_T$")

   axs[0, 0].set_xlabel("$p_T$ (GeV)")
   axs[0, 1].set_xlabel("$p_T$ (GeV)")
   axs[1, 0].set_xlabel("$p_T$ (GeV)")
   axs[1, 1].set_xlabel("$p_T$ (GeV)")


   fig.savefig("./plots/pT_Fake.png")

   fig, ax = plt.subplots(figsize=(10,10))
   out[dataset]["fakeRate"][:,"opposite"].plot1d()
   ax.set_title("OS $p_T$ Fake Rate")
   ax.set_xlabel("$p_T$ (GeV)")
   fig.savefig("./plots/pT_fakeRate_OS.png")


   fig, ax = plt.subplots(figsize=(10, 10))
   out[dataset]["fakeRate"][:,"same"].plot1d()
   ax.set_title("SS $p_T$ Fake Rate")
   ax.set_xlabel("$p_T$ (GeV)")
   fig.savefig("./plots/pT_fakeRate_SS.png")

   fig, ax = plt.subplots(figsize=(10, 10))
   out[dataset]["vec_pt"].plot1d()
   ax.set_title("Muon + Bosoted Tau Vector $p_T$")
   ax.set_xlabel("$p_T$ (GeV)")
   fig.savefig("./plots/muTau_vec_pt.png")

   fig, ax = plt.subplots(figsize=(10, 10))
   out[dataset]["vec_mass"].plot1d()
   ax.set_title("Muon + Bosoted Tau Vector Mass")
   ax.set_xlabel("Mass (GeV)")
   fig.savefig("./plots/muTau_vec_mass.png")


