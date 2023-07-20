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
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.nanoevents.methods import vector


class MyProcessor(processor.ProcessorABC):
   def __init__(self):
      pass

   def write(self, file, string):
      file = open(file, "a")
      file.write(str(string))
      file.write('\n')
      file.close()
      return

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

      #Create log file for writing/debugging
      f = "./log.txt"
      my_file = Path(f)
      if my_file.is_file():
         os.remove(f)
      self.write(f, "Inital Events: ")
      self.write(f, ak.num(events, axis=0))
      
      #define shortcuts
      events = events[(ak.num(events.Muon) > 0) & (ak.num(events.boostedTau) > 0)]
      muons = events.Muon
      boostedTaus = events.boostedTau

      self.write(f, "Events containing Muon and boosted Tau (+ muon and boosted Taus): ")
      self.write(f, ak.num(events, axis=0))
      self.write(f, ak.num(ak.flatten(events.Muon, axis=1), axis=0))
      self.write(f, ak.num(ak.flatten(events.boostedTau, axis=1), axis=0))


      #mask cuts for all events
      muon_mask =  (ak.any(muons.pt > 28, axis=1) 
                    & ak.all(abs(muons.eta) < 2.4, axis=1))
      
      boostedTau_mask =  (ak.all(boostedTaus.pt > 20, axis=1) 
                         & ak.all(abs(boostedTaus.eta) <= 2.5, axis=1)
                         & ak.all(boostedTaus.rawIsodR03 > .7, axis=1)
                         & ak.all(boostedTaus.idAntiMu == 3, axis = 1))
      met_mask = events.MET.pt > 30
      
      #Apply all masks
      mask = muon_mask & boostedTau_mask & met_mask
      events = events[mask]


      self.write(f, "After event cuts (muon and tau): ")
      self.write(f, ak.num(events, axis=0))
      self.write(f, ak.num(ak.flatten(events.Muon, axis=1), axis=0))
      self.write(f, ak.num(ak.flatten(events.boostedTau, axis=1), axis=0))


      #Pair up boostedTaus with a muon
      muon_boostedTau_pairs = ak.cartesian({'muon': events.Muon, 'boostedTau': events.boostedTau}, axis=1, nested=False)

      #dr cut
      dr = muon_boostedTau_pairs['muon'].delta_r(muon_boostedTau_pairs['boostedTau'])
      dr_mask = ak.all(dr > .1, axis=1) & ak.all(dr < .8, axis=1)
      muon_boostedTau_pairs = muon_boostedTau_pairs[dr_mask]

      self.write(f, "After dr cuts (muon, boosted tau): ")
      self.write(f, ak.num(muon_boostedTau_pairs, axis=0))
      self.write(f, ak.num(ak.flatten(muon_boostedTau_pairs['muon'], axis=1), axis=-1))
      self.write(f, ak.num(ak.flatten(muon_boostedTau_pairs['boostedTau'], axis=1), axis=-1))


      #flatten everything out
      muon_boostedTau_pairs = ak.flatten(muon_boostedTau_pairs, axis=-1)
      
      
      #Separate by charge signs
      signs = muon_boostedTau_pairs['muon'].charge * muon_boostedTau_pairs['boostedTau'].charge
      OS_pairs = muon_boostedTau_pairs[signs == -1]
      SS_pairs = muon_boostedTau_pairs[signs == 1]

      self.write(f, "OS pairs:")
      self.write(f, ak.num(OS_pairs, axis=0))
      self.write(f, ak.num(OS_pairs['muon'], axis=0))
      self.write(f, ak.num(OS_pairs['boostedTau'], axis=0))
      self.write(f, "SS pairs:")
      self.write(f, ak.num(SS_pairs, axis=0))
      self.write(f, ak.num(SS_pairs['muon'], axis=0))
      self.write(f, ak.num(SS_pairs['boostedTau'], axis=0))

      #Fake rate numerator cuts
      OS_VVLooseNum = OS_pairs['boostedTau'].rawMVAnewDM2017v2 > .5
      SS_VVLooseNum = SS_pairs['boostedTau'].rawMVAnewDM2017v2 > .5

      #Make combined vector
      muonVec = self.makeVector(muon_boostedTau_pairs['muon'])
      tauVec = self.makeVector(muon_boostedTau_pairs['boostedTau'])
      muTauVec = muonVec.add(tauVec)

      #Vector transverse momentum histo
      vec_pt = Hist.new.Regular(100,0,600, name='vec_pt', label="$p_T$ (GeV)").Double()
      vec_pt.fill(vec_pt = muTauVec.pt)

      #Vector mass histo
      vec_mass = Hist.new.Regular(100,0,400, name='vec_mass', label="Mass(GeV)").Double()
      vec_mass.fill(vec_mass = muTauVec.mass)

      #Boosted Tau pT Fake Rate values (OS, SS, for both num and denum)
      OS_flat_pt_denom = OS_pairs['boostedTau'].pt
      OS_flat_pt_num = OS_pairs['boostedTau'][OS_VVLooseNum].pt
      SS_flat_pt_denom = SS_pairs['boostedTau'].pt
      SS_flat_pt_num = SS_pairs['boostedTau'][SS_VVLooseNum].pt

      #Create boostedPT histo and fill
      boosted_pt = Hist.new.Regular(10,0,500, name="pt", label ="$p_T$ (GeV)").StrCat(["opposite", "same"], name="sign", label = "Sign").StrCat(["denominator", "numerator"], name="fraction", label="Fraction").Double()
      boosted_pt.fill(sign="opposite", fraction="denominator", pt = OS_flat_pt_denom)
      boosted_pt.fill(sign="opposite", fraction="numerator", pt = OS_flat_pt_num)
      boosted_pt.fill(sign="same", fraction="denominator", pt = SS_flat_pt_denom)
      boosted_pt.fill(sign="same", fraction="numerator", pt = SS_flat_pt_num)

      #Create fake Rate histos for OS and SS
      boosted_pt_rate_os = boosted_pt[:,"opposite","numerator"] / (boosted_pt[:,"opposite","denominator"])
      boosted_pt_rate_ss = boosted_pt[:,"same","numerator"] / (boosted_pt[:,"same","denominator"])   

      return {
         dataset: {
            "pT": boosted_pt,
            "OS_fakeRate": boosted_pt_rate_os,
            "SS_fakeRate": boosted_pt_rate_ss,
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


   fig.savefig("./plots/pT_Fake.png")


   #OS boostedTau pT fakerate
   fig, ax = plt.subplots(figsize=(10,10))
   out[dataset]["OS_fakeRate"].plot1d()
   ax.set_title("OS $p_T$ Fake Rate")
   ax.set_xlabel("$p_T$ (GeV)")
   fig.savefig("./plots/pT_fakeRate_OS.png")

   #SS bosotedTau pT fakeRate
   fig, ax = plt.subplots(figsize=(10, 10))
   out[dataset]["SS_fakeRate"].plot1d()
   ax.set_title("SS $p_T$ Fake Rate")
   ax.set_xlabel("$p_T$ (GeV)")
   fig.savefig("./plots/pT_fakeRate_SS.png")

   # Muon+BoostedTau vector pT
   fig, ax = plt.subplots(figsize=(10, 10))
   out[dataset]["vec_pt"].plot1d()
   ax.set_title("Muon + Boosted Tau Vector $p_T$")
   ax.set_xlabel("$p_T$ (GeV)")
   fig.savefig("./plots/muTau_vec_pt.png")

   # Muon+BoostedTau vector mass
   fig, ax = plt.subplots(figsize=(10, 10))
   out[dataset]["vec_mass"].plot1d()
   ax.set_title("Muon + Boosted Tau Vector Mass")
   ax.set_xlabel("Mass (GeV)")
   fig.savefig("./plots/muTau_vec_mass.png")