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
   
   def bit_mask(self, bit):
      mask = 0
      mask += (1 << bit)
      return mask
   
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
      elif "ggH125" in name: return 48.30 * 0.0621
      elif "ggZHLL125" in name: return 0.1223 * 0.062 * 3 * 0.033658
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
         if "2L2Nu" in name: return 831.76 * TT_FulLep_BR
         if "Hadronic" in name: return 831.76 * TT_Had_BR
         if "SemiLeptonic" in name: return 831.76 * TT_SemiLep_BR
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
      elif "ZZ4l" in name: return 1.212
      else: print("Something's Wrong!")
      return

   def process(self,events,name, num_events):
      dataset = events.metadata['dataset']
      events = events[ak.all(events.jetDeepCSVTags_b < .7527, axis=-1)]
      events = events[events.nMu > 0]
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
      trigger_mask_Mu27 = self.bit_mask(19)
      trigger_mask_Mu50 = self.bit_mask(21)

      TriggerEvents_Mu50 = events[(muon[:,0].pt > 55)]
      TriggerEvents_Mu50 = TriggerEvents_Mu50[(np.bitwise_and(TriggerEvents_Mu50.HLTJet, trigger_mask_Mu50) == trigger_mask_Mu50)]

      TriggerEvents_Mu27 = events[(muon[:,0].pt < 55)]
      TriggerEvents_Mu27 = TriggerEvents_Mu27[(np.bitwise_and(TriggerEvents_Mu27.HLTJet, trigger_mask_Mu27) == trigger_mask_Mu27)]
      RelIsoMu = (TriggerEvents_Mu27.muPFChIso + TriggerEvents_Mu27.muPFNeuIso + TriggerEvents_Mu27.muPFPhoIso - (0.5 * TriggerEvents_Mu27.muPFPUIso)) / TriggerEvents_Mu27.muPt

      #Define muon candidate
      muon27 = ak.zip( 
			{
				"pt": TriggerEvents_Mu27.muPt,
				"E": TriggerEvents_Mu27.muEn,
				"eta": TriggerEvents_Mu27.muEta,
				"phi": TriggerEvents_Mu27.muPhi,
				"nMuon": TriggerEvents_Mu27.nMu,
				"charge": TriggerEvents_Mu27.muCharge,
			},
			with_name="MuonArray",
			behavior=candidate.behavior,
		)
      muon50 = ak.zip( 
			{
				"pt": TriggerEvents_Mu50.muPt,
				"E": TriggerEvents_Mu50.muEn,
				"eta": TriggerEvents_Mu50.muEta,
				"phi": TriggerEvents_Mu50.muPhi,
				"nMuon": TriggerEvents_Mu50.nMu,
				"charge": TriggerEvents_Mu50.muCharge,
			},
			with_name="MuonArray",
			behavior=candidate.behavior,
		)
      muon = ak.zip( 
			{
				"pt": np.concatenate((muon27.pt, muon50.pt), axis=0),
				"E": np.concatenate((muon27.E, muon50.E), axis=0),
				"eta": np.concatenate((muon27.eta, muon50.eta), axis=0),
				"phi": np.concatenate((muon27.phi, muon50.phi), axis=0),
				"nMuon": np.concatenate((muon27.nMuon, muon50.nMuon), axis=0),
				"charge": np.concatenate((muon27.charge, muon50.charge), axis=0),
			},
			with_name="MuonArray",
			behavior=candidate.behavior,
		)
      muon_mask =  ((muon[:,0].pt > 30)
                     & (np.absolute(muon.eta) < 2.4))
      tau = ak.zip( 
			{
				"pt": np.concatenate((TriggerEvents_Mu27.boostedTauPt, TriggerEvents_Mu50.boostedTauPt), axis=0),
				"E": np.concatenate((TriggerEvents_Mu27.boostedTauEnergy, TriggerEvents_Mu50.boostedTauEnergy), axis=0),
				"Px": np.concatenate((TriggerEvents_Mu27.boostedTauPx, TriggerEvents_Mu50.boostedTauPx), axis=0),
				"Py": np.concatenate((TriggerEvents_Mu27.boostedTauPy, TriggerEvents_Mu50.boostedTauPy), axis=0),
				"Pz": np.concatenate((TriggerEvents_Mu27.boostedTauPz, TriggerEvents_Mu50.boostedTauPz), axis=0),
				"mass": np.concatenate((TriggerEvents_Mu27.boostedTauMass, TriggerEvents_Mu50.boostedTauMass), axis=0),
				"eta": np.concatenate((TriggerEvents_Mu27.boostedTauEta, TriggerEvents_Mu50.boostedTauEta), axis=0),
				"phi": np.concatenate((TriggerEvents_Mu27.boostedTauPhi, TriggerEvents_Mu50.boostedTauPhi), axis=0),
				"nBoostedTau": np.concatenate((TriggerEvents_Mu27.nBoostedTau, TriggerEvents_Mu50.nBoostedTau), axis=0),
				"charge": np.concatenate((TriggerEvents_Mu27.boostedTauCharge, TriggerEvents_Mu50.boostedTauCharge), axis=0),
				"iso": np.concatenate((TriggerEvents_Mu27.boostedTauByLooseIsolationMVArun2v1DBoldDMwLTNew, TriggerEvents_Mu50.boostedTauByLooseIsolationMVArun2v1DBoldDMwLTNew), axis=0),
            "antiMu": np.concatenate((TriggerEvents_Mu27.boostedTauByLooseMuonRejection3, TriggerEvents_Mu50.boostedTauByLooseMuonRejection3), axis=0),
			},
			with_name="TauArray",
			behavior=candidate.behavior,
		)
      boostedTau_mask = ((tau.pt > 20)
                        & (np.absolute(tau.eta) <= 2.5)
                         & (tau.antiMu == True)
                         & (tau.iso == True))
      
      #Apply all masks
      tau = tau[boostedTau_mask]
      muon = muon[muon_mask]

      #dr cut
      muon_boostedTau_pairs = ak.cartesian({'tau': tau, 'muons': muon}, nested=False)
      dr = self.delta_r(muon_boostedTau_pairs['tau'], muon_boostedTau_pairs['muons'])
      dr_cut = (dr > .1) & (dr < 1)
      muon_boostedTau_pairs = muon_boostedTau_pairs[dr_cut]

      #Separate based on charge
      OS_pairs = muon_boostedTau_pairs[(muon_boostedTau_pairs['tau'].charge * muon_boostedTau_pairs['muons'].charge < 0)]
      SS_pairs = muon_boostedTau_pairs[(muon_boostedTau_pairs['tau'].charge * muon_boostedTau_pairs['muons'].charge > 0)]

      #Get back the muons and taus after all cuts have been applied
      tau_OS, mu_OS = ak.unzip(OS_pairs)
      tau_SS, mu_SS = ak.unzip(SS_pairs)
      if ak.sum(mu_OS.pt) == 0:
         return {
            dataset: {
            "mass": np.zeros(0),
            "mass_w": np.zeros(0),
            "ss_mass": np.zeros(0),
            "ss_mass_w": np.zeros(0)
            }
         }
      muVec = self.makeVector(mu_OS, "muon")
      tauVec = self.makeVector(tau_OS, "tau")
      ZVec_OS = tauVec.add(muVec)
      ZVec_OS = ZVec_OS[ZVec_OS.pt > 250]
      muVec = self.makeVector(mu_SS, "muon")
      tauVec = self.makeVector(tau_SS, "tau") 
      ZVec_SS = tauVec.add(muVec)
      ZVec_SS = ZVec_SS[ZVec_SS.pt > 250] 

      XSection = self.weightCalc(name)
      if XSection != 1:
         luminosity = 59830.
         weight = (XSection * luminosity) / num_events
      else: weight = 1

      shape = np.shape(ak.flatten(ZVec_OS.mass, axis=1))
      mass_w = np.full(shape=shape, fill_value=weight, dtype=np.double)
      shape = np.shape(ak.flatten(ZVec_SS.mass, axis=1))
      ss_mass_w = np.full(shape=shape, fill_value=weight, dtype=np.double)
      return {
         dataset: {
            "mass": ak.flatten(ZVec_OS.mass, axis=1),
            "mass_w": mass_w,
            "ss_mass": ak.flatten(ZVec_SS.mass, axis=1),
            "ss_mass_w": ss_mass_w
         }
      }
   
   def postprocess(self, accumulator):
      pass
   

if __name__ == "__main__":
   directory = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedH3/2018/mt/v2_fast_Hadd"
   dataset = "SingleMuon"
   fig, ax = plt.subplots()
   hep.style.use(hep.style.ROOT)
   p = MyProcessor()
   fileList = ["DYJetsToLL_Pt-100To250.root",    "Tbar-tW.root",
      'DYJetsToLL_Pt-250To400.root',            'toptopH125.root',
      'DYJetsToLL_Pt-400To650.root',            'T-tchan.root',
      'DYJetsToLL_Pt-50To100.root',             'TTTo2L2Nu.root',
      'DYJetsToLL_Pt-650ToInf.root',            'TTToHadronic.root',
      'TTToSemiLeptonic.root',                  'Tbar-tchan.root',
      'ggH125.root',                            'T-tW.root',
      'ggZHLL125.root',                         'VV2l2nu.root',
      'ggZHNuNu125.root',                       'WJetsToLNu_HT-100To200.root',
      'ggZHQQ125.root',                         'WJetsToLNu_HT-1200To2500.root',
      'JJH0PMToTauTauPlusOneJets.root',         'WJetsToLNu_HT-200To400.root',
      'JJH0PMToTauTauPlusTwoJets.root',         'WJetsToLNu_HT-2500ToInf.root',
      'JJH0PMToTauTauPlusZeroJets.root',        'WJetsToLNu_HT-400To600.root',
                                                'WJetsToLNu_HT-600To800.root',
                                                'WJetsToLNu_HT-800To1200.root',
                                                'WMinusH125.root',
                                                'WPlusH125.root',
                                                'WZ1l1nu2q.root',
                                                'WZ1l3nu.root',
      'qqH125.root',                            'WZ2l2q.root',
      'SingleMuon_Run2018A-17Sep2018-v2.root',  'WZ3l1nu.root',
      'SingleMuon_Run2018B-17Sep2018-v1.root',  'ZH125.root',
      'SingleMuon_Run2018C-17Sep2018-v1.root',  'ZZ2l2q.root',
      'SingleMuon_Run2018D-22Jan2019-v2.root',  'ZZ4l.root']

   datasets = ["WJets", "DY", "Top", "SingleTop", "SMHiggs", "Diboson", "QCD", "SingleMuon"]
   DY, DY_w = [], []
   WJets, WJets_w = [], []
   Diboson, Diboson_w = [], []
   SMHiggs, SMHiggs_w = [], []
   SingleTop, SingleTop_w = [], []
   Top, Top_w = [], []
   Data, Data_w = [], []
   Data_SS, DY_SS, Top_SS, WJets_SS = [], [], [], []
   DY_SS_w, Top_SS_w, WJets_SS_w = [], [], []
   bins=np.linspace(0, 150, 15)

   #read in file
   for sample in fileList:
      fname = os.path.join(directory, sample)
      file = uproot.open(fname)
      events = NanoEventsFactory.from_root(
         file,
         treepath="/mutau_tree",
         schemaclass=BaseSchema,
         metadata={"dataset": dataset},
      ).events()
      string = str(sample)
      num_events = file['hEvents'].member('fEntries')
      out = p.process(events, fname, num_events)
      if "DY" in string:
            print("DY ", fname)
            DY = np.append(DY, out[dataset]["mass"], axis=0)
            DY_w = np.append(DY_w, out[dataset]["mass_w"], axis=0)
            DY_SS = np.append(DY_SS, out[dataset]["ss_mass"], axis=0)
            DY_SS_w = np.append(DY_SS_w, out[dataset]["ss_mass_w"], axis=0)
      if "WJets" in string:
            print("WJets ", fname)
            WJets = np.append(WJets, out[dataset]["mass"], axis=0)
            WJets_w = np.append(WJets_w, out[dataset]["mass_w"], axis=0)
            WJets_SS = np.append(WJets_SS, out[dataset]["ss_mass"], axis=0)
            WJets_SS_w = np.append(WJets_SS_w, out[dataset]["ss_mass_w"], axis=0)
      matches = ["WZ", "VV2l2nu", "ZZ4l", "ZZ2l2q"]
      if any([x in fname for x in matches]):
            print("Diboson ", fname)
            Diboson = np.append(Diboson, out[dataset]["mass"], axis=0)
            Diboson_w = np.append(Diboson_w, out[dataset]["mass_w"], axis=0)
      matches = ["gg", "qqH125", "toptop", "WMinus", "WPlus", "ZH125", "TauTau"] 
      if any([x in fname for x in matches]):
            print("SMHiggs ", fname)
            SMHiggs = np.append(SMHiggs, out[dataset]["mass"], axis=0)
            SMHiggs_w = np.append(SMHiggs_w, out[dataset]["mass_w"], axis=0) 
      matches = ["Tbar", "T-tchan", "tW"]
      if any([x in fname for x in matches]):
            print("SingleTop ", fname)
            SingleTop = np.append(SingleTop, out[dataset]["mass"], axis=0)
            SingleTop_w = np.append(SingleTop_w, out[dataset]["mass_w"], axis=0) 
      if "TTTo" in string:
            print("Top ", fname)
            Top = np.append(Top, out[dataset]["mass"], axis=0)
            Top_w = np.append(Top_w, out[dataset]["mass_w"], axis=0) 
            Top_SS = np.append(Top_SS, out[dataset]["ss_mass"], axis=0)
            Top_SS_w = np.append(Top_SS_w, out[dataset]["ss_mass_w"], axis=0)
      if "SingleMuon" in string:
            print("SingleMuon ", fname)
            Data = np.append(Data, out[dataset]["mass"], axis=0)
            Data_w = np.append(Data_w, out[dataset]["mass_w"], axis=0)
            Data_SS = np.append(Data_SS, out[dataset]["ss_mass"], axis=0)

   NonQCD = np.append(np.append(DY_SS, WJets_SS, axis=0), Top_SS, axis=0)
   NonQCD_w = np.append(np.append(DY_SS_w, WJets_SS_w, axis=0), Top_SS_w, axis=0) 


   Data_h, Data_bins = np.histogram(Data, bins=bins)
   Data_SS_h, Data_bins = np.histogram(Data_SS, bins=bins)
   NonQCD_h, NonQCD_bins = np.histogram(NonQCD, bins=bins, weights=NonQCD_w)
   QCD_h = np.subtract(Data_SS_h, NonQCD_h, dtype=object)
   for i in range(QCD_h.size):
      if QCD_h[i] < 0.0:
         QCD_h[i] = 0.0
   QCDScaleFactor = 1.6996559936491136
   QCD_h = QCD_h * QCDScaleFactor
   QCD_w = np.full(shape=QCD_h.shape, fill_value=1, dtype=np.double) 

   DY_h, DY_bins = np.histogram(DY, bins=bins, weights=DY_w)
   WJets_h, WJets_bins = np.histogram(WJets, bins=bins, weights=WJets_w) 
   Top_h, Top_bins = np.histogram(Top, bins=bins, weights= Top_w) 
   SingleTop_h, SingleTop_bins = np.histogram(SingleTop, bins=bins, weights=SingleTop_w) 
   Diboson_h, Diboson_bins = np.histogram(Diboson, bins=bins, weights=Diboson_w)
   SMHiggs_h, SMHiggs_bins = np.histogram(SMHiggs, bins=bins, weights=SMHiggs_w)   

   mass =   [SMHiggs_h,   Diboson_h,   SingleTop_h,   DY_h,   Top_h,   WJets_h, QCD_h]
   mass_w = [SMHiggs_w, Diboson_w, SingleTop_w, DY_w, Top_w, WJets_w, QCD_w]
   labels = ["SMHiggs", "Diboson", "SingleTop", "DY", "Top", "WJets", "QCD"]
   hep.histplot(mass, label=labels, histtype=("fill"), bins=bins, stack=True)
   hep.histplot(Data_h, label="Data", histtype=("errorbar"), bins=bins)
   plt.legend(loc = 'upper right', ncols = 2, fontsize = 8)

   #OS boostedTau pT fakerate
   plt.title("Boosted Tau + Muon Visible Mass", fontsize= 'small')
   ax.set_xlabel("Mass (GeV)")
   fig.savefig("./SingleMuon_VISIBLE_MASS.png")
