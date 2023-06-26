import sys
import awkward as ak
import uproot
import boost_histogram as bh
import hist
from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep
from coffea import processor, nanoevents
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

class MyProcessor(processor.ProcessorABC):
   def __init__(self):
      pass

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
                         & ak.all(abs(boostedTaus.eta) <= 2.5, axis=1))
      met_mask = events.MET.pt > 30
      dr = events.Muon[:,0].delta_r(events.boostedTau[:,0])
      dr_mask = ak.any(dr > .1, axis=0) & ak.any(dr < .8, axis=0)
      
      #combine all cuts into one mask
      mask = muon_mask & boostedTau_mask & met_mask  & dr_mask
      selected_events = events[mask]

      signs = selected_events.Muon[:,0].charge * selected_events.boostedTau[:,0].charge
      VVLooseNum = ak.all(selected_events.boostedTau.rawMVAnewDM2017v2 > .95, axis =1)
      
      print("Number of Leading Boosted Tau Before: ")
      print(ak.num(ak.flatten(events.boostedTau.pt, axis=None), axis=0))

      print("Number of Leading Boosted Tau After: ")
      print(ak.num(ak.flatten(selected_events.boostedTau.pt, axis=None), axis=0))

      histogram = Hist.new.Regular(10,0,500, name="pt", label ="$p_T (GeV)").StrCat(["opposite", "same"], name="sign", label = "Sign").StrCat(["denominator", "numerator"], name="fraction", label="Fraction").Double()
      histogram.fill(sign="opposite", fraction="denominator", pt = ak.flatten(selected_events.boostedTau[signs == -1].pt, axis=None))
      histogram.fill(sign="opposite", fraction="numerator", pt = ak.flatten(selected_events.boostedTau[signs == -1 & VVLooseNum].pt, axis=None))
      histogram.fill(sign="same", fraction="denominator", pt = ak.flatten(selected_events.boostedTau[signs == 1].pt, axis=None))
      histogram.fill(sign="same", fraction="numerator", pt = ak.flatten(selected_events.boostedTau[signs == 1 & VVLooseNum].pt, axis=None))

      rate_os = (histogram[:,"opposite","denominator"]).view() / histogram[:,"opposite","numerator"].view()
      rate_ss = (histogram[:,"same","denominator"]).view() / histogram[:,"same","numerator"].view()

      fakeRate = Hist.new.Regular(10, 0, 500, name="fakerate", label="$p_T (GeV)").StrCat(["opposite", "same"], name="sign", label = "Sign").Double()
      fakeRate.fill(sign="opposite", fakerate=rate_os)
      fakeRate.fill(sign="same", fakerate=rate_ss)
   

      return {
         dataset: {
            "entries": len(events),
            "events": selected_events,
            "pT": histogram,
            "fakeRate": fakeRate
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


   fig.savefig("pT_Fake.png")

   fig, ax = plt.subplots(figsize=(5, 10))
   out[dataset]["fakeRate"][:,"opposite"].plot1d()
   ax.set_title("OS $p_T$ Fake Rate")
   ax.set_xlabel("$p_T$ (GeV)")
   fig.savefig("pT_fakeRate_OS.png")


   fig, ax = plt.subplots(figsize=(5, 10))
   out[dataset]["fakeRate"][:,"same"].plot1d()
   ax.set_title("SS $p_T$ Fake Rate")
   ax.set_xlabel("$p_T$ (GeV)")
   fig.savefig("pT_fakeRate_SS.png")
