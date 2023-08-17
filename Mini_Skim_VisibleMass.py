import sys
import os
from pathlib import Path
import math
import awkward as ak
import numpy as np
import uproot
import boost_histogram as bh
import hist
from hist import Hist, intervals
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

   def makeVector(self, particle, name):
      if name == "muon": mass = 0.10565837
      if name == "tau": mass = particle.mass
      newVec = ak.zip(
        {
            "pt": particle.pt,
            "eta": particle.eta,
            "phi": particle.phi,
            "mass": mass,
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
      )
      return newVec

   def weightCalc(self, name):
      WScaleFactor = 1.21
      TT_FulLep_BR= 0.1061
      TT_SemiLep_BR= 0.4392
      TT_Had_BR= 0.4544

      if "SingleMuon" in name: return 1
      elif "DYJets" in name:
         if "50To100" in name: return 387.130778
         elif "100To250" in name: return 89.395097
         elif "250To400" in name: return 3.435181
         elif "400To650" in name: return 0.464024
         elif "650ToInf" in name: return 0.043602
      elif "WJets" in name:
         if "100To200" in name: return 1345* WScaleFactor
         elif "200To400" in name: return 359.7* WScaleFactor
         elif "400To600" in name: return 48.91* WScaleFactor
         elif "600To800" in name: return 12.05* WScaleFactor
         elif "800To1200" in name: return 5.501* WScaleFactor
         elif "1200To2500" in name: return 1.329* WScaleFactor
         elif "2500ToInf" in name: return 0.03216* WScaleFactor 
      elif "QCD" in name:
         if "300to500" in name: return 347700
         elif "500to700" in name: return 32100
         elif "700to1000" in name: return 6831
         elif "1000to1500" in name: return 1207
         elif "1500to2000" in name: return 119.9
         elif "2000toInf" in name: return 25.24
      elif "ggH125" in name: return 48.30* 0.0621
      elif "ggZHLL125" in name: return 0.1223 * 0.062 * 3*0.033658
      elif "ggZHNuNu125" in name: return 0.1223 * 0.062 * 0.2000
      elif "ggZHQQ125" in name: return 0.1223 * 0.062 * 0.6991
      elif "JJH0" in name:
         if "OneJet" in name: return 0.1383997884
         elif "TwoJet" in name: return 0.2270577971
         elif "ZeroJet" in name: return 0.3989964912
      elif "qqH125" in name: return 3.770 * 0.0621
      elif "Tbar-tchan" in name: return  26.23
      elif "Tbar-tW" in name: return 35.6
      elif "toptopH125" in name: return 0.5033 * 0.062
      elif "T-tchan.root" in name: return 44.07
      elif "T-tW" in name: return 35.6
      elif "TTT" in name:
         if "2L2Nu" in name: return 831.76*TT_FulLep_BR
         if "Hadronic" in name: return 831.76*TT_Had_BR
         if "SemiLeptonic" in name: return 831.76*TT_SemiLep_BR
      elif "VV2l2nu" in name: return 11.95
      elif "WMinus" in name: return 0.5272 * 0.0621
      elif "WPlus" in name: return 0.8331 * 0.0621
      elif "WZ" in name:
         if "nu2q" in name: return 10.71
         elif "1l3nu" in name: return 3.05
         elif "2l2q" in name: return 5.595
         elif "3l1nu" in name: return 4.708
      elif "ZH125" in name: return 0.7544 * 0.0621
      elif "ZZ2l2q" in name: return 3.22
      elif "ZZ41" in name: return 1.212
      else: print("Something's Wrong!")
      return

   def process(self,events, bkg, name):
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
				"nBoostedTau": events.nBoostedTau,
				"charge": events.boostedTauCharge,
				"iso": events.boostedTauByLooseIsolationMVArun2v1DBoldDMwLTNew,
            "antiMu": events.boostedTauByLooseMuonRejection3
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

      #mask cuts for candidates
      muon_mask =  ((muon.pt > 28)
                    & (np.absolute(muon.eta) < 2.4))
      
      boostedTau_mask = ((tau.pt > 20)
                        & (np.absolute(tau.eta) <= 2.5)
                         & tau.antiMu)
      
      
      #Apply all masks
      tau = tau[boostedTau_mask]
      muon = muon[muon_mask]

      #dr cut
      muon_boostedTau_pairs = ak.cartesian({'tau': tau, 'muons': muon}, nested=False)
      dr = self.delta_r(muon_boostedTau_pairs['tau'], muon_boostedTau_pairs['muons'])
      dr_cut = (dr > .1) & (dr < 1)
      muon_boostedTau_pairs = muon_boostedTau_pairs[dr_cut]

      #Separate based on charge
      OS_pairs = muon_boostedTau_pairs[(muon_boostedTau_pairs['tau'].charge + muon_boostedTau_pairs['muons'].charge == 0)]
      SS_pairs = muon_boostedTau_pairs[(muon_boostedTau_pairs['tau'].charge + muon_boostedTau_pairs['muons'].charge != 0)]

      #Get back the muons and taus after all cuts have been applied
      tau_OS, mu_OS = ak.unzip(OS_pairs)
      tau_SS, mu_SS = ak.unzip(SS_pairs)

      muVec = self.makeVector(mu_OS, "muon")
      tauVec = self.makeVector(tau_OS, "tau")
      ZVec_OS = tauVec.add(muVec)

      muVec = self.makeVector(mu_SS, "muon")
      tauVec = self.makeVector(tau_SS, "tau") 
      ZVec_SS = tauVec.add(muVec)
      
      weight = self.weightCalc(name)
      shape = np.shape(ak.flatten(ZVec_OS.mass, axis=1))
      mass_w = np.full(shape=shape, fill_value=weight, dtype=np.double)
      return {
         dataset: {
            "mass": ak.flatten(ZVec_OS.mass, axis=1),
            "mass_w": mass_w,
         }
      }
   
   def postprocess(self, accumulator):
      pass
   

if __name__ == "__main__":
   directory = 'samples'
   dataset = "all"
   fig, ax = plt.subplots()
   hep.style.use(hep.style.ROOT)
   p = MyProcessor()
   bkgs = ["Top", "SingleTop", "SMHiggs", "Diboson", "WJets", "DY", "QCD"]
   DY, DY_w = [], []
   WJets, WJets_w = [], []
   QCD, QCD_w = [], []
   Diboson, Diboson_w = [], []
   SMHiggs, SMHiggs_w = [], []
   SingleTop, SingleTop_w = [], []
   Top, Top_w = [], []


   #read in file
   for sample in os.listdir(directory):
      fname = os.path.join(directory, sample)
      events = NanoEventsFactory.from_root(
         fname,
         treepath="/mutau_tree",
         schemaclass=BaseSchema,
         metadata={"dataset": dataset},
      ).events()
      for b in bkgs:
         if b in fname:
            out = p.process(events, b, fname)
            if "DY" in fname:
               DY = np.append(DY, out[dataset]["mass"], axis=0)
               DY_w = np.append(DY_w, out[dataset]["mass_w"], axis=0)
            if "WJets" in fname:
               WJets = np.append(WJets, out[dataset]["mass"], axis=0)
               WJets_w = np.append(WJets_w, out[dataset]["mass_w"], axis=0)
            if "QCD" in fname:
               QCD = np.append(QCD, out[dataset]["mass"], axis=0)
               QCD_w = np.append(QCD_w, out[dataset]["mass_w"], axis=0)
            if "ZZ4l" or "VV" or "WZ" or "ZZ" in fname:
              Diboson = np.append(Diboson, out[dataset]["mass"], axis=0)
              Diboson_w = np.append(Diboson_w, out[dataset]["mass_w"], axis=0) 
            if "gg" or "qq" or "toptop" or "WMinus" or "WPlus" or "ZH" in fname:
               SMHiggs = np.append(SMHiggs, out[dataset]["mass"], axis=0)
               SMHiggs_w = np.append(SMHiggs_w, out[dataset]["mass_w"], axis=0) 
            if "Tbar" or "T-tchan" or "T-tW" in fname:
               SingleTop = np.append(SingleTop, out[dataset]["mass"], axis=0)
               SingleTop_w = np.append(SingleTop_w, out[dataset]["mass_w"], axis=0) 
            if "TTT" in fname:
               Top = np.append(Top, out[dataset]["mass"], axis=0)
               Top_w = np.append(Top_w, out[dataset]["mass_w"], axis=0) 
   
   mass = [Top, SingleTop, SMHiggs, Diboson, WJets, DY, QCD]
   mass_w = [Top_w, SingleTop_w, SMHiggs_w, Diboson_w, WJets_w, DY_w, QCD_w]

   range = (0,200)
   bins = 50

   plt.hist(mass, weights=mass_w,label=bkgs, histtype=("stepfilled"),stacked=True, bins=bins, range=range)
   plt.legend(loc = 'best')

   #OS boostedTau pT fakerate
   plt.title("Boosted Tau + Muon Visible Mass")
   ax.set_xlabel("Mass (GeV)")
   fig.savefig("./VISIBLE_MASS.png")