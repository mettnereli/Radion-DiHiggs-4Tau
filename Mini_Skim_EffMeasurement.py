import sys
import os
from pathlib import Path
import math
import awkward as ak
import numpy as np
import uproot
import boost_histogram as bh
import hist
from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep
from coffea import processor, nanoevents
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea.nanoevents.methods import vector, candidate


class MyProcessor(processor.ProcessorABC):
   def __init__(self):
      pass

   def write(self, file, string):
      file = open(file, "a")
      file.write(str(string))
      file.write('\n')
      file.close()
      return
   
   def delta_r(self, candidate1, candidate2):
      return np.sqrt((candidate2.eta - candidate1.eta)**2 + (candidate2.phi - candidate1.phi)**2)

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

      #Define tau candidate
      tau = ak.zip( 
			{
				"pt": events.boostedTauPt,
				"E": events.boostedTauEnergy,
				"Px": events.boostedTauPx,
				"Py": events.boostedTauPy,
				"Pz": events.boostedTauPz,
				"mass": events.boostedTauMass,
				"eta": events.boostedTauEta,
				"phi": events.boostedTauPhi,
				"leadingIndx": events.leadtauIndex,
				"nBoostedTau": events.nBoostedTau,
				"charge": events.boostedTauCharge,
				"iso": events.boostedTauByLooseIsolationMVArun2v1DBoldDMwLTNew,
			},
			with_name="TauArray",
			behavior=candidate.behavior,
		)

      #Define muon candidate
      muon = ak.zip( 
			{
				"pt": events.muPt,
				"E": events.muEn,
				"eta": events.muEta,
				"phi": events.muPhi,
				"nMuon": events.nMu,
				"charge": events.muCharge,
			},
			with_name="MuonArray",
			behavior=candidate.behavior,
		)

      #Create log file for writing/debugging
      f = "./log_mini.txt"

      my_file = Path(f)
      if my_file.is_file():
         os.remove(f)
      self.write(f, "Inital Events: ")
      self.write(f, ak.num(events, axis=0))
      self.write(f, ak.sum(ak.num(muon.pt, axis=1)))
      self.write(f, ak.sum(ak.num(tau.pt, axis=1)) )

      #mask cuts for candidates
      muon_mask =  ((muon.pt > 28)
                    & (np.absolute(muon.eta) < 2.4))
      
      boostedTau_mask = ((tau.pt > 20)
                         & (np.absolute(tau.eta) <= 2.5))
                          #look into antiMu ID
      
      
      #Apply all masks
      tau = tau[boostedTau_mask]
      muon = muon[muon_mask]
      
      
      #2nd Log
      self.write(f, "After event cuts (muon and tau): ")
      self.write(f, ak.sum(ak.num(muon.pt, axis=1)))
      self.write(f, ak.sum(ak.num(tau.pt, axis=1)))
      
      #dr cut
      muon_boostedTau_pairs = ak.cartesian({'tau': tau, 'muons': muon}, nested=False)
      dr = self.delta_r(muon_boostedTau_pairs['tau'], muon_boostedTau_pairs['muons'])
      dr_cut = (dr > .1) & (dr < 1)
      muon_boostedTau_pairs = muon_boostedTau_pairs[dr_cut]
      
      #3rd Log
      self.write(f, "After dr cuts (boosted tau): ")
      self.write(f, ak.sum(ak.num(muon_boostedTau_pairs, axis=1)))


      #Separate based on charge
      OS_pairs = muon_boostedTau_pairs[(muon_boostedTau_pairs['tau'].charge + muon_boostedTau_pairs['muons'].charge == 0)]
      SS_pairs = muon_boostedTau_pairs[(muon_boostedTau_pairs['tau'].charge + muon_boostedTau_pairs['muons'].charge != 0)]


      #4th Log
      self.write(f, "OS pairs:")
      self.write(f, ak.sum(ak.num(OS_pairs, axis=1)))
      self.write(f, "SS pairs:")
      self.write(f, ak.sum(ak.num(SS_pairs, axis=1)))

      #Get back the muons and taus after all cuts have been applied
      tau_OS, mu_OS = ak.unzip(OS_pairs)
      tau_SS, mu_SS = ak.unzip(SS_pairs)

      #Boosted Tau pT Fake Rate values (OS, SS, for both num and denum)
      OS_flat_pt_denom = ak.flatten(tau_OS.pt, axis=1)
      OS_flat_pt_num = ak.flatten(tau_OS[tau_OS.iso].pt, axis=1)
      SS_flat_pt_denom = ak.flatten(tau_SS.pt, axis=1)
      SS_flat_pt_num = ak.flatten(tau_SS[tau_SS.iso].pt, axis=1)
      
      #5th Log
      self.write(f, "boosted pts before binning: ")
      self.write(f, ak.num(OS_flat_pt_denom, axis=0))
      self.write(f, ak.num(OS_flat_pt_num, axis=0))
      self.write(f, ak.num(SS_flat_pt_denom, axis=0))
      self.write(f, ak.num(SS_flat_pt_num, axis=0))

      #Create boostedPT histo and fill
      boosted_pt = Hist.new.Regular(10,0,500, name="pt", label ="$p_T$ (GeV)").StrCat(["opposite", "same"], name="sign", label = "Sign").StrCat(["denominator", "numerator"], name="fraction", label="Fraction").Double()
      boosted_pt.fill(sign="opposite", fraction="denominator", pt = OS_flat_pt_denom)
      boosted_pt.fill(sign="opposite", fraction="numerator", pt = OS_flat_pt_num)
      boosted_pt.fill(sign="same", fraction="denominator", pt = SS_flat_pt_denom)
      boosted_pt.fill(sign="same", fraction="numerator", pt = SS_flat_pt_num)

      #Create fake Rate histos for OS and SS
      boosted_pt_rate_os = boosted_pt[:,"opposite","numerator"] / (boosted_pt[:,"opposite","denominator"])
      boosted_pt_rate_ss = boosted_pt[:,"same","numerator"] / (boosted_pt[:,"same","denominator"])

      #Final Log
      self.write(f, "boosted pt's : (OSnum OSdenom, SSnum, SSdenom)")
      self.write(f, (boosted_pt[:,"opposite","numerator"].counts()))
      self.write(f, boosted_pt[:,"opposite","denominator"].counts())
      self.write(f, boosted_pt[:,"same","numerator"].counts())
      self.write(f, boosted_pt[:,"same","denominator"].counts())
      
      return {
         dataset: {
            "pT": boosted_pt,
            "OS_fakeRate": boosted_pt_rate_os,
            "SS_fakeRate": boosted_pt_rate_ss,
         }
      }
   
   def postprocess(self, accumulator):
      pass
   

if __name__ == "__main__":
   
   dataset = sys.argv[1]

   #read in file
   fname = "./GluGluToRadionToHHTo4T_M-1000.root"
   events = NanoEventsFactory.from_root(
      fname,
      treepath="/4tau_tree",
      schemaclass=BaseSchema,
      metadata={"dataset": dataset},
   ).events()

   #Run process
   p = MyProcessor()
   out = p.process(events)

   #plot and save

   #boosted Tau pt (all four values)
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


   fig.savefig("./mini_plots/pT_Fake.png")


   #OS boostedTau pT fakerate
   fig, ax = plt.subplots(figsize=(10,10))
   hep.histplot(out[dataset]["OS_fakeRate"], ax=ax)
   ax.set_title("OS $p_T$ Fake Rate")
   ax.set_xlabel("$p_T$ (GeV)")
   fig.savefig("./mini_plots/pT_fakeRate_OS.png")

   #SS bosotedTau pT fakeRate
   fig, ax = plt.subplots(figsize=(10, 10))
   hep.histplot(out[dataset]["SS_fakeRate"], ax=ax)
   ax.set_title("SS $p_T$ Fake Rate")
   ax.set_xlabel("$p_T$ (GeV)")
   fig.savefig("./mini_plots/pT_fakeRate_SS.png")

