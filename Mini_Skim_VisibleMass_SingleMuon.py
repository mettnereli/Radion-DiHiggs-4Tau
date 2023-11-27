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
from coffea import processor, nanoevents, lookup_tools
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea.nanoevents.methods import vector, candidate
from coffea.lookup_tools import extractor, evaluator



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

   def process(self, events,name, num_events):
      dataset = events.metadata['dataset']
      #At least on muon
      events = events[ak.num(events.muPt) > 0]
      #Bjet veto
      events = events[ak.all(events.jetDeepCSVTags_b < .7527, axis=-1)]
      #Extra electron veto
      eleEvents = events[ak.num(events.elePt) > 0]
      RelIsoEle = (eleEvents.elePFChIso + eleEvents.elePFNeuIso + eleEvents.elePFPhoIso - (0.5 * eleEvents.elePFPUIso)) / eleEvents.elePt
      eleCut = ak.all((np.abs(RelIsoEle) < 0.8) & (eleEvents.elePt[:,0] > 10) & (np.bitwise_and(eleEvents.eleIDbit, self.bit_mask(2)) == self.bit_mask(2)), axis=-1)
      events = events[eleCut == False]
      #Extra muon veto
      extraMuEvents = events[ak.num(events.muPt) > 1]
      RelIsoMu = (extraMuEvents.muPFChIso + extraMuEvents.muPFNeuIso + extraMuEvents.muPFPhoIso - (0.5 * extraMuEvents.muPFPUIso)) / extraMuEvents.muPt
      muCut = ak.all((np.abs(RelIsoMu) < 0.8) & (extraMuEvents.muPt[:,1] > 10) & (np.bitwise_and(extraMuEvents.muIDbit, self.bit_mask(2)) == self.bit_mask(2)), axis=-1)
      events = events[muCut == False]

      #Trigger cut
      trigger_mask_Mu27 = self.bit_mask(19)
      trigger_mask_Mu50 = self.bit_mask(21) 

      Mu50 = events[(events.muPt[:,0] > 55)]
      Mu50 = Mu50[(np.bitwise_and(Mu50.HLTEleMuX, trigger_mask_Mu50) == trigger_mask_Mu50)]
      Mu27 = events[(events.muPt[:,0] < 55) & (events.muPt[:,0] > 30)]
      Mu27 = Mu27[(np.bitwise_and(Mu27.HLTEleMuX, trigger_mask_Mu27) == trigger_mask_Mu27)]

      #LepCorr = np.append(Lep27Corr, Lep50Corr, axis=0)
      muon = ak.zip( 
			{
				"pt": np.concatenate((Mu27.muPt, Mu50.muPt), axis=0),
				"energy": np.concatenate((Mu27.muEn, Mu50.muEn), axis=0),
				"eta": np.concatenate((Mu27.muEta, Mu50.muEta), axis=0),
				"phi": np.concatenate((Mu27.muPhi, Mu50.muPhi), axis=0),
				"nMuon": np.concatenate((Mu27.nMu, Mu50.nMu), axis=0),
				"charge": np.concatenate((Mu27.muCharge, Mu50.muCharge), axis=0),
   		   "D0": np.concatenate(((Mu27.muD0, Mu50.muD0)), axis=0),
           	"Dz": np.concatenate((Mu27.muDz, Mu50.muDz), axis=0),
			},
			with_name="MuonArray",
			behavior=candidate.behavior,
		)

      tau = ak.zip( 
			{
				"pt": np.concatenate((Mu27.boostedTauPt, Mu50.boostedTauPt), axis=0),
				"E": np.concatenate((Mu27.boostedTauEnergy, Mu50.boostedTauEnergy), axis=0),
				"Px": np.concatenate((Mu27.boostedTauPx, Mu50.boostedTauPx), axis=0),
				"Py": np.concatenate((Mu27.boostedTauPy, Mu50.boostedTauPy), axis=0),
				"Pz": np.concatenate((Mu27.boostedTauPz, Mu50.boostedTauPz), axis=0),
				"mass": np.concatenate((Mu27.boostedTauMass, Mu50.boostedTauMass), axis=0),
				"eta": np.concatenate((Mu27.boostedTauEta, Mu50.boostedTauEta), axis=0),
				"phi": np.concatenate((Mu27.boostedTauPhi, Mu50.boostedTauPhi), axis=0),
				"nBoostedTau": np.concatenate((Mu27.nBoostedTau, Mu50.nBoostedTau), axis=0),
				"charge": np.concatenate((Mu27.boostedTauCharge, Mu50.boostedTauCharge), axis=0),
				"iso": np.concatenate((Mu27.boostedTauByLooseIsolationMVArun2v1DBoldDMwLTNew, Mu50.boostedTauByLooseIsolationMVArun2v1DBoldDMwLTNew), axis=0),
            "antiMu": np.concatenate((Mu27.boostedTauByLooseMuonRejection3, Mu50.boostedTauByLooseMuonRejection3), axis=0),
			},
			with_name="TauArray",
			behavior=candidate.behavior,
		)
      
      #Apply cut
      pairs = ak.cartesian({'tau': tau, 'muon': muon[:,0]}, nested=False)
      dr = self.delta_r(pairs['tau'], pairs['muon'])
      muTau_mask = ak.all((pairs['muon'].eta < 2.4)
                  & (pairs['muon'].D0 < 0.045)
                  & (pairs['muon'].Dz < 0.2)
                  & (pairs['tau'].pt > 20)
                  & (np.absolute(pairs['tau'].eta) <= 2.5)
                  & (pairs['tau'].antiMu == True)
                  & (pairs['tau'].iso == True)
                  & (dr > .1) 
                  & (dr < .8), axis=-1)
      pairs = pairs[muTau_mask]

      #Separate based on charge
      OS_pairs = pairs[(pairs['tau'].charge * pairs['muon'].charge < 0)]
      SS_pairs = pairs[(pairs['tau'].charge * pairs['muon'].charge > 0)]

      #If everything cut return 0
      if ak.sum(OS_pairs['tau'].pt) == 0:
         return {
            dataset: {
            "mass": np.zeros(0),
            "mass_w": np.zeros(0),
            "ss_mass": np.zeros(0),
            "ss_mass_w": np.zeros(0)
            }
         }

      #Define two vectors
      #OS
      muVec = self.makeVector(OS_pairs['muon'], "muon")
      tauVec = self.makeVector(OS_pairs['tau'], "tau")
      muVec50 = muVec[muVec.pt > 55]
      muVec27 = muVec[(muVec.pt < 55) & (muVec.pt > 30)]
      tauVec50 = tauVec[muVec.pt > 55]
      tauVec27 = tauVec[(muVec.pt < 55) & (muVec.pt > 30)]
      #SS
      muVec_SS = self.makeVector(SS_pairs['muon'], "muon")
      tauVec_SS = self.makeVector(SS_pairs['tau'], "tau")
      muVec50_SS = muVec_SS[muVec_SS.pt > 55]
      muVec27_SS = muVec_SS[(muVec_SS.pt < 55) & (muVec_SS.pt > 30)]
      tauVec50_SS = tauVec_SS[muVec_SS.pt > 55]
      tauVec27_SS = tauVec_SS[(muVec_SS.pt < 55) & (muVec_SS.pt > 30)]

      #Make Z Vector
      #OS
      ZVec50 = tauVec50.add(muVec50)
      ZVec27 = tauVec27.add(muVec27)
      ZVec50 = ZVec50[ZVec50.pt > 200]
      ZVec27 = ZVec27[ZVec27.pt > 200]

      #SS
      ZVec50_SS = tauVec50_SS.add(muVec50_SS)
      ZVec27_SS = tauVec27_SS.add(muVec27_SS)
      ZVec50_SS = ZVec50_SS[ZVec50_SS.pt > 200]
      ZVec27_SS = ZVec27_SS[ZVec27_SS.pt > 200]


      #Add weight sets
      ext = extractor()
      ext.add_weight_sets(["IDCorr NUM_LooseID_DEN_genTracks_pt_abseta ./RunBCDEF_SF_ID.root", "Trg50Corr Mu50_OR_TkMu50_PtEtaBins/pt_abseta_ratio ./Trigger_EfficienciesAndSF_RunBtoF.root", "Trg27Corr IsoMu24_OR_IsoTkMu24_PtEtaBins/pt_abseta_ratio ./Trigger_EfficienciesAndSF_RunBtoF.root", "IsoCorr NUM_LooseRelIso_DEN_MediumID_pt_abseta ./RunBCDEF_SF_ISO.root", "pTCorr Ratio2D ./zmm_2d_2018.root"])
      ext.finalize()
      evaluator = ext.make_evaluator()

      #OS
      Mu50IsoCorr = evaluator["IsoCorr"](ZVec50.pt, ZVec50.eta)   
      Mu50TrgCorr = evaluator["Trg50Corr"](ZVec50.pt, ZVec50.eta) 
      Mu50IDCorr = evaluator["IDCorr"](ZVec50.pt, ZVec50.eta) 
      Mu27IsoCorr = evaluator["IsoCorr"](ZVec27.pt, ZVec27.eta) 
      Mu27TrgCorr = evaluator["Trg27Corr"](ZVec27.pt, ZVec27.eta) 
      Mu27IDCorr = evaluator["IDCorr"](ZVec27.pt, ZVec27.eta) 

      Lep50Corr = Mu50IsoCorr * Mu50TrgCorr * Mu50IDCorr
      Lep27Corr = Mu27IsoCorr * Mu27TrgCorr * Mu27IDCorr

      #SS
      Mu50IsoCorr_SS = evaluator["IsoCorr"](ZVec50_SS.pt, ZVec50_SS.eta)   
      Mu50TrgCorr_SS = evaluator["Trg50Corr"](ZVec50_SS.pt, ZVec50_SS.eta)   
      Mu50IDCorr_SS = evaluator["IDCorr"](ZVec50_SS.pt, ZVec50_SS.eta)  
      Mu27IsoCorr_SS = evaluator["IsoCorr"](ZVec27_SS.pt, ZVec27_SS.eta) 
      Mu27TrgCorr_SS = evaluator["Trg27Corr"](ZVec27_SS.pt, ZVec27_SS.eta)  
      Mu27IDCorr_SS = evaluator["IDCorr"](ZVec27_SS.pt, ZVec27_SS.eta) 

      Lep50Corr_SS = Mu50IsoCorr_SS * Mu50TrgCorr_SS * Mu50IDCorr_SS
      Lep27Corr_SS = Mu27IsoCorr_SS * Mu27TrgCorr_SS * Mu27IDCorr_SS


      XSection = self.weightCalc(name)

      shape = np.shape(np.append(ak.flatten(ZVec27.mass, axis=1), ak.flatten(ZVec50.mass, axis=1)))
      SS_shape = np.shape(np.append(ak.flatten(ZVec27_SS.mass, axis=1), ak.flatten(ZVec50_SS.mass, axis=1)))

      #If not data, calculate all weight corrections
      if XSection != 1:
         luminosity = 59830.
         lumiWeight = (XSection * luminosity) / num_events
         
         LepCorrection = np.append(ak.flatten(Lep27Corr, axis=-1), ak.flatten(Lep50Corr, axis=-1), axis=-1) 
         #OS
         if ("DYJets" in name) or ("WJets" in name):
            pTCorrection27 = evaluator["pTCorr"](ZVec27.mass, ZVec27.pt)
            pTCorrection50 = evaluator["pTCorr"](ZVec50.mass, ZVec50.pt)
            Lep27Corr = Lep27Corr * pTCorrection27
            Lep50Corr = Lep50Corr * pTCorrection50
            LepCorrection = np.append(ak.flatten(Lep27Corr, axis=-1), ak.flatten(Lep50Corr, axis=-1), axis=-1)
         
         #SS
         SS_LepCorrection = np.append(ak.flatten(Lep27Corr_SS, axis=-1), ak.flatten(Lep50Corr_SS, axis=-1), axis=-1) 

         mass_w = np.full(shape=shape, fill_value=lumiWeight, dtype=np.double)
         mass_w = np.multiply(mass_w, ak.flatten(LepCorrection, axis=-1))
         SS_mass_w = np.full(shape=SS_shape, fill_value=lumiWeight, dtype=np.double) 
         SS_mass_w = np.multiply(SS_mass_w, ak.flatten(SS_LepCorrection, axis=-1))
      else:
         mass_w = np.full(shape=shape, fill_value=1, dtype=np.double)
         SS_mass_w = np.full(shape=SS_shape, fill_value=1, dtype=np.double) 

      return {
         dataset: {
            "mass": np.append(ak.flatten(ZVec27.mass, axis=-1), ak.flatten(ZVec50.mass, axis=-1), axis=-1),
            "mass_w": mass_w,
            "ss_mass": np.append(ak.flatten(ZVec27_SS.mass, axis=-1), ak.flatten(ZVec50_SS.mass, axis=-1), axis=-1),
            "ss_mass_w": SS_mass_w
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
   bins=np.linspace(0, 150, 30)

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
      num_events = file['hEvents'].member('fEntries') / 2
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

   QCDScaleFactor = 1.6996559936491136
   Data_h, Data_bins = np.histogram(Data, bins=bins)
   Data_SS_h, Data_SS_bins = np.histogram(Data_SS, bins=bins)
   NonQCD_h, NonQCD_bins = np.histogram(NonQCD, bins=bins)
   QCD_h = np.subtract(Data_SS_h, NonQCD_h, dtype=object, out=None)
   for i in range(QCD_h.size):
      if QCD_h[i] < 0.0:
         QCD_h[i] = 0.0
   QCD_h = QCD_h * QCDScaleFactor
   QCD_w = np.full(shape=QCD_h.shape, fill_value=1, dtype=np.double)
   QCD_hist = (QCD_h, Data_SS_bins)

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
   hep.histplot(Data_h, label="Data", histtype=("errorbar"), bins=bins, color='k')
   plt.legend(loc = 'upper right', ncols = 2, fontsize = 8)

   #OS boostedTau pT fakerate
   plt.title("Boosted Tau + Muon Visible Mass", fontsize= 'small')
   ax.set_xlabel("Mass (GeV)")
   fig.savefig("./singlemuon_plots/SingleMuon_VISIBLE_MASS.png")

   outFile = uproot.recreate("boostedHTT_mt_2018.input.root")
   DY_h = np.histogram(DY, bins=bins, weights=DY_w)
   TT_h = np.histogram(Top, bins=bins, weights=Top_w)
   VV_h = np.histogram(np.append(Diboson, SingleTop), bins=bins, weights=np.append(Diboson_w, SingleTop_w))
   WJets_h = np.histogram(WJets, bins=bins, weights=WJets_w)
   Data_h = np.histogram(Data, bins=bins)
   outFile["DY_Jets_mt_1_13TeV/DYJets125"] = DY_h
   outFile["DY_Jets_mt_1_13TeV/TT"] = TT_h
   outFile["DY_Jets_mt_1_13TeV/VV"] = VV_h
   outFile["DY_Jets_mt_1_13TeV/W"] = WJets_h
   outFile["DY_Jets_mt_1_13TeV/QCD"] = QCD_hist
   outFile["DY_Jets_mt_1_13TeV/data_obs"] = Data_h
