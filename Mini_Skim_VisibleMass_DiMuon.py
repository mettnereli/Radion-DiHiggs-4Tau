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
from coffea import processor, lookup_tools, nanoevents
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
   
   def delta_r(self, candidate1, candidate2):
      return np.sqrt((candidate2.eta - candidate1.eta)**2 + (candidate2.phi - candidate1.phi)**2)

   def mass(self, part1, part2):
      return np.sqrt((part1.E + part2.E)**2 - (part1.Px + part2.Px)**2 - (part1.Py + part2.Py)**2 - (part1.Pz + part2.Pz)**2)

   def bit_mask(self, bit):
      mask = 0
      mask += (1 << bit)
      return mask

   def makeVector(self, particle, name, XSection):
      mass = 0.10565837
      if XSection != 1:
         newVec = ak.zip(
        {
            "pt": particle.pt,
            "eta": particle.eta,
            "phi": particle.phi,
            "mass": mass,
            "puTrue": particle.puTrue, 
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
         )
      else:
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
      if "Data" in name: return 1
      elif "DYJets" in name:
         if "50To100" in name: return 387.130778
         elif "100To250" in name: return 89.395097
         elif "250To400" in name: return 3.435181
         elif "400To650" in name: return 0.464024
         elif "650ToInf" in name: return 0.043602
      elif "WJets" in name:
         if "50To100" in name: return 3298.37
         elif "100To250" in name: return 689.75
         elif "250To400" in name: return 24.507
         elif "400To600" in name: return 3.1101
         elif "600ToInf" in name: return 0.46832
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
         if "OneJet" in name: return 0.2270577971
         elif "TwoJet" in name: return 0.1383997884
         elif "ZeroJet" in name: return 0.3989964912
      elif "qqH125" in name: return 3.770 * 0.0621
      elif "Tbar-tchan.root" in name: return  26.23
      elif "Tbar-tW.root" in name: return 35.6
      elif "toptopH125" in name: return 0.5033 * 0.062
      elif "T-tchan.root" in name: return 44.07
      elif "T-tW.root" in name: return 35.6
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
         elif "1l1nu2q" in name: return 10.71
      elif "ZH125" in name: return 0.7544 * 0.0621
      elif "ZZ2l2q" in name: return 3.22
      elif "ZZ4l" in name: return 1.212
      else: print("Something's Wrong!")
      return

   def process(self,events,name, num_events, PUWeight):
      dataset = events.metadata['dataset']
      #print(events.fields)


      #Find weighting factor and identify if Data
      XSection = self.weightCalc(name)
      print(fname, " ", XSection)

      #At least two muons
      events = events[ak.num(events.muPt) > 1]

      # bjet veto - may need to fix
      events = events[ak.num(events.jetPt) > 0]
      events = events[ak.all((events.jetDeepCSVTags_b > .7527) & (events.jetPFLooseId > .5) & (events.jetPt > 20) & (np.abs(events.jetEta) < 2.4), axis=-1) == False]

      # Extra electron veto 
      eleEvents = events[ak.num(events.elePt) > 0]
      RelIsoEle = (eleEvents.elePFChIso + eleEvents.elePFNeuIso + eleEvents.elePFPhoIso - (0.5 * eleEvents.elePFPUIso)) / eleEvents.elePt
      MVAId_low = (np.abs(eleEvents.eleSCEta) <= 0.8) & (eleEvents.eleIDMVANoIso > 0.837)
      MVAId_mid = (np.abs(eleEvents.eleSCEta) > 0.8) & (np.abs(eleEvents.eleSCEta) <= 1.5) & (eleEvents.eleIDMVANoIso > 0.715) 
      MVAId_high = (np.abs(eleEvents.eleSCEta) >= 1.5) & (eleEvents.eleIDMVANoIso > 0.357) 
      MVAId = (MVAId_low) | (MVAId_mid) | (MVAId_high)
      eleCut = ak.all((np.abs(RelIsoEle) < 0.3) & (eleEvents.elePt[:,0] > 10) & (MVAId == True)) | ak.all((eleEvents.elePt[:,0] > 120) & (MVAId == True), axis=-1)
      events = events[eleCut == False]

      #Extra muon veto
      extraMuEvents = events[ak.num(events.muPt) > 2]
      RelIsoMu = (extraMuEvents.muPFChIso + extraMuEvents.muPFNeuIso + extraMuEvents.muPFPhoIso - (0.5 * extraMuEvents.muPFPUIso)) / extraMuEvents.muPt
      muCut = ak.all((np.abs(RelIsoMu) < 0.3) & (extraMuEvents.muPt[:,2] > 10) & (np.bitwise_and(extraMuEvents.muIDbit, self.bit_mask(2)) == self.bit_mask(2)), axis=-1)
      events = events[muCut == False]

      #HT Cut
      jets = ak.zip( 
		{
				"pt": events.jetPt,
				"eta": events.jetEta,
				"phi": events.jetPhi,
            "E": events.jetEn,
            "jetPFLooseId": events.jetPFLooseId,
		},
			   with_name="JetArray",
		      behavior=candidate.behavior,
		) 
      jets = jets[(jets.jetPFLooseId > 0.5) & (jets.pt > 30) & (np.abs(jets.eta) < 3.0)]
      HT = ak.sum(jets.pt, axis=-1) 
      events = events[HT > 200]

      #Apply triggermask
      trigger_mask_Mu50 = self.bit_mask(21)
      Mu50 = events[(events.muPt[:,0] > 55)]
      Mu50 = Mu50[(np.bitwise_and(Mu50.HLTEleMuX, trigger_mask_Mu50) == trigger_mask_Mu50)]

      #Define muon candidate
      #Need to separate with IF statement since Data doesn't have puTrue
      if XSection != 1:
         muon = ak.zip( 
			{
				"pt": Mu50.muPt,
				"energy": Mu50.muEn,
				"eta": Mu50.muEta,
				"phi": Mu50.muPhi,
				"nMuon": Mu50.nMu,
				"charge": Mu50.muCharge,
            "ID": Mu50.muIDbit,
            "muD0": Mu50.muD0,
            "muDz": Mu50.muDz,
            "puTrue": Mu50.puTrue[:,0],
            "muPFNeuIso": Mu50.muPFNeuIso,
            "muPFPhoIso": Mu50.muPFPhoIso,
            "muPFPUIso": Mu50.muPFPUIso,
            "muPFChIso": Mu50.muPFChIso,
            "Met": Mu50.pfMetNoRecoil,
            "Metphi": Mu50.pfMetPhiNoRecoil,
			},
			with_name="MuonArray",
			behavior=candidate.behavior,
		   )
      else:
         muon = ak.zip( 
			{
				"pt": Mu50.muPt,
				"energy": Mu50.muEn,
				"eta": Mu50.muEta,
				"phi": Mu50.muPhi,
				"nMuon": Mu50.nMu,
				"charge": Mu50.muCharge,
            "ID": Mu50.muIDbit,
            "muD0": Mu50.muD0,
            "muDz": Mu50.muDz,
            "muPFNeuIso": Mu50.muPFNeuIso,
            "muPFPhoIso": Mu50.muPFPhoIso,
            "muPFPUIso": Mu50.muPFPUIso,
            "muPFChIso": Mu50.muPFChIso,
            "Met": Mu50.pfMetNoRecoil,
            "Metphi": Mu50.pfMetPhiNoRecoil,
			},
			with_name="MuonArray",
			behavior=candidate.behavior,
		   )

      #Get all mumu combinations
      dimuon = ak.combinations(muon, 2, fields=['i0', 'i1'])

      #Check Isolation
      IsoCheck = ak.any(((dimuon['i0'].muPFNeuIso + dimuon['i0'].muPFPhoIso - 0.5 * dimuon['i0'].muPFPUIso) > 0.0), axis=-1)
      IsoLep1Val = np.divide(dimuon['i0'].muPFChIso, dimuon['i0'].pt)
      IsoLep1Val = np.where(IsoCheck, (dimuon['i0'].muPFChIso + dimuon['i0'].muPFNeuIso + dimuon['i0'].muPFPhoIso  - 0.5 * dimuon['i0'].muPFPUIso) / dimuon['i0'].pt, IsoLep1Val)

      Sub_IsoCheck = ak.any(((dimuon['i1'].muPFNeuIso + dimuon['i1'].muPFPhoIso - 0.5 * dimuon['i1'].muPFPUIso) > 0.0), axis=-1)
      Sub_IsoLep1Val = np.divide(dimuon['i1'].muPFChIso, dimuon['i1'].pt)
      Sub_IsoLep1Val = np.where(Sub_IsoCheck, (dimuon['i1'].muPFChIso + dimuon['i1'].muPFNeuIso + dimuon['i1'].muPFPhoIso  - 0.5 * dimuon['i1'].muPFPUIso) / dimuon['i1'].pt, Sub_IsoLep1Val)

      #Check ID
      IDMask = self.bit_mask(2)
      MuID = ((np.bitwise_and(dimuon['i0'].ID, IDMask) == IDMask) & (np.abs(dimuon['i0'].muD0) < 0.045) & (np.abs(dimuon['i0'].muDz) < .2))
      Sub_MuID = ((np.bitwise_and(dimuon['i1'].ID, IDMask) == IDMask) & (np.abs(dimuon['i1'].muD0) < 0.045) & (np.abs(dimuon['i1'].muDz) < .2)) 
      #Check Delta_R
      dr = self.delta_r(dimuon['i0'], dimuon['i1'])

      #Create mask for cuts
      dimuon_mask =  ((dimuon['i0'].pt > 53)
                     & (dimuon['i1'].pt > 10)
                     & (np.absolute(dimuon['i0'].eta) < 2.4)
                     & (np.absolute(dimuon['i1'].eta) < 2.4)
                     & (dr > .1)
                     & (dr < .8)
                     & (MuID)
                     & (Sub_MuID)
                     & (IsoLep1Val < .3)
                     & (Sub_IsoLep1Val < .3))
      dimuon = dimuon[dimuon_mask]

      #Corrections
      ext = extractor()
      ext.add_weight_sets(["IDCorr NUM_LooseID_DEN_genTracks_pt_abseta ./RunBCDEF_SF_ID.root", "TrgCorr Mu50_OR_TkMu50_PtEtaBins/pt_abseta_ratio ./Trigger_EfficienciesAndSF_RunBtoF.root", "pTCorr Ratio2D ./zmm_2d_2018.root"])
      ext.finalize()
      evaluator = ext.make_evaluator()

      #Separate based on charge
      OS_pairs = dimuon[(dimuon['i0'].charge + dimuon['i1'].charge == 0)]
      SS_pairs = dimuon[(dimuon['i0'].charge + dimuon['i1'].charge != 0)]


      #Get back the muons and taus after all cuts have been applied
      mu1_OS, mu2_OS = ak.unzip(OS_pairs)
      mu1_SS, mu2_SS = ak.unzip(SS_pairs)

      #in case every item is cut out
      if (ak.sum(mu1_OS.pt) == 0) or (ak.sum(mu1_SS.pt) == 0):
         return {
            dataset: {
            "mass": np.zeros(0),
            "mass_w": np.zeros(0),
            "ss_mass": np.zeros(0),
            "ss_mass_w": np.zeros(0),
            "pT": np.zeros(0),
            "eta": np.zeros(0),
            }
         }

      #MET Vector
      MetVec =  ak.zip(
      {
         "pt": OS_pairs['i0'].Met,
         "eta": 0,
         "phi": OS_pairs['i0'].Metphi,
         "mass": 0,
         },
      with_name="PtEtaPhiMLorentzVector",
      behavior=vector.behavior,
      )  

      MetVec_SS =  ak.zip(
      {
         "pt": SS_pairs['i0'].Met,
         "eta": 0,
         "phi": SS_pairs['i0'].Metphi,
         "mass": 0,
         },
      with_name="PtEtaPhiMLorentzVector",
      behavior=vector.behavior,
      ) 

      # make into vectors
      mu1Vec = self.makeVector(OS_pairs['i0'], "muon1", XSection)
      mu2Vec = self.makeVector(OS_pairs['i1'], "muon2", XSection)

      mu1Vec_SS = self.makeVector(SS_pairs['i0'], "muon1", XSection)
      mu2Vec_SS = self.makeVector(SS_pairs['i1'], "muon2", XSection)

      #TMass cut
      tmass = np.sqrt(np.square(mu1Vec.pt + OS_pairs['i0'].Met) - np.square(mu1Vec.px + OS_pairs['i0'].Met * np.cos(OS_pairs['i0'].Metphi)) - np.square(mu1Vec.py + OS_pairs['i0'].Met * np.sin(OS_pairs['i0'].Metphi)))
      mu1Vec = mu1Vec[tmass < 80]
      mu2Vec = mu2Vec[tmass < 80]

      #Combine two vectors
      diMuVec_OS = mu1Vec.add(mu2Vec)
      diMuVec_SS = mu1Vec_SS.add(mu2Vec_SS)

      #Get weighting array for plotting
      shape = np.shape(ak.flatten(diMuVec_OS.pt, axis=1))
      SS_shape = np.shape(ak.flatten(diMuVec_SS.pt, axis=1))

      #If not data, calculate all weight corrections
      if XSection != 1:

         #Luminosity (2018)
         luminosity = 59830.
         lumiWeight = (XSection * luminosity) / num_events
         
         #Pileup
         puTrue = np.array(np.rint(ak.flatten(mu1Vec.puTrue, axis=-1)), dtype=np.int8)
         SS_puTrue = np.array(np.rint(ak.flatten(mu1Vec_SS.puTrue, axis=-1)), dtype=np.int8)

         #OS
         LeadMuIDCorrection = evaluator["IDCorr"](ak.flatten(mu1Vec.pt, axis=-1), ak.flatten(mu1Vec.eta, axis=-1)) #ID
         SubMuIDCorrection = evaluator["IDCorr"](ak.flatten(mu2Vec.pt, axis=-1), ak.flatten(mu2Vec.eta, axis=-1))
         TrgCorrection = evaluator["TrgCorr"](ak.flatten(mu1Vec.pt, axis=-1), ak.flatten(mu1Vec.eta, axis=-1)) #Trigger
         PUCorrection = PUWeight[puTrue] #Pileup

         LepCorrection = LeadMuIDCorrection * SubMuIDCorrection * TrgCorrection * PUCorrection #Combine

         if ("DYJets" in name):
            #Z_Pt correction
            pTCorrection = evaluator["pTCorr"](ak.flatten(diMuVec_OS.mass, axis=-1), ak.flatten(diMuVec_OS.pt, axis=-1))
            LepCorrection = LeadMuIDCorrection * SubMuIDCorrection * TrgCorrection * PUCorrection * pTCorrection

         #SS
         SS_LeadMuIDCorrection = evaluator["IDCorr"](ak.flatten(mu1Vec_SS.pt, axis=-1), ak.flatten(mu1Vec_SS.eta, axis=-1))
         SS_SubMuIDCorrection = evaluator["IDCorr"](ak.flatten(mu2Vec_SS.pt, axis=-1), ak.flatten(mu2Vec_SS.eta, axis=-1))
         SS_TrgCorrection = evaluator["TrgCorr"](ak.flatten(mu1Vec_SS.pt, axis=-1), ak.flatten(mu1Vec_SS.eta, axis=-1))
         SS_PUCorrection = PUWeight[SS_puTrue]
         SS_LepCorrection = SS_LeadMuIDCorrection * SS_SubMuIDCorrection * SS_TrgCorrection * SS_PUCorrection

         mass_w = np.full(shape=shape, fill_value=lumiWeight, dtype=np.double)
         mass_w = np.multiply(mass_w, ak.flatten(LepCorrection, axis=-1))
         SS_mass_w = np.full(shape=SS_shape, fill_value=lumiWeight, dtype=np.double) 
         SS_mass_w = np.multiply(SS_mass_w, ak.flatten(SS_LepCorrection, axis=-1))
      else:
         mass_w = np.full(shape=shape, fill_value=1, dtype=np.double)
         SS_mass_w = np.full(shape=SS_shape, fill_value=1, dtype=np.double) 


      #Assign each weight to each value in the plot for easy access
      mass_h = np.column_stack((ak.flatten(diMuVec_OS.mass, axis=1), mass_w))
      SS_mass_h = np.column_stack((ak.flatten(diMuVec_SS.mass, axis=1), SS_mass_w))

      pt_h = np.column_stack((ak.flatten(diMuVec_OS.pt, axis=1), mass_w))
      SS_pt_h = np.column_stack((ak.flatten(diMuVec_SS.pt, axis=1), SS_mass_w))

      eta_h = np.column_stack((ak.flatten(diMuVec_OS.eta, axis=1), mass_w))
      
      #200 GeV pT cut, 60-120 mass cut
      mass_h, pt_h, eta_h = mass_h[pt_h[:,0] > 200], pt_h[pt_h[:,0] > 200], eta_h[pt_h[:,0] > 200]
      mass_h, pt_h, eta_h = mass_h[(mass_h[:,0] > 60) & (mass_h[:,0] < 120)], pt_h[(mass_h[:,0] > 60) & (mass_h[:,0] < 120)], eta_h[(mass_h[:,0] > 60) & (mass_h[:,0] < 120)]

      SS_mass_h = SS_mass_h[SS_pt_h[:,0] > 200]
      SS_mass_h = SS_mass_h[(SS_mass_h[:,0] > 60) & (SS_mass_h[:,0] < 120)]

      return {
         dataset: {
            "mass": mass_h[:,0],
            "mass_w": mass_h[:,1],
            "pT": pt_h[:,0],
            "eta": eta_h[:,0],
            "ss_mass": SS_mass_h[:,0],
            "ss_mass_w": SS_mass_h[:,1],
         }
      }
   
   
   def postprocess(self, accumulator):
      pass
   

if __name__ == "__main__":
   directory = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedH2/2018/mm/v2_Hadd/"
   dataset = "DiMuon"
   fig, ax = plt.subplots()
   hep.style.use(hep.style.ROOT)
   plt.style.use('seaborn-v0_8-colorblind')
   p = MyProcessor()
   fileList = ["DYJetsToLL_Pt-100To250.root",    "Tbar-tW.root",
      'DYJetsToLL_Pt-250To400.root',            'DYJetsToLL_Pt-400To650.root',            
      'TTTo2L2Nu.root',                         'DYJetsToLL_Pt-650ToInf.root',            
      'TTToHadronic.root',                      'TTToSemiLeptonic.root',                  
      'Tbar-tchan.root',                        'ggH125.root',                            
      'T-tW.root',                              'VV2l2nu.root',          
      'JJH0PMToTauTauPlusOneJets.root',         'WJetsToLNu_Pt-50To100.root',
      'JJH0PMToTauTauPlusTwoJets.root',         'WJetsToLNu_Pt-100To250.root',
      'JJH0PMToTauTauPlusZeroJets.root',        'WJetsToLNu_Pt-250To400.root',
      'WMinusH125.root',                        'WJetsToLNu_Pt-400To600.root',
      'WPlusH125.root',                         'WJetsToLNu_Pt-600ToInf.root',
      'WZ1l1nu2q.root',                         'WZ2l2q.root',
      'WZ3l1nu.root',                           'ZH125.root',
      'ZZ2l2q.root',                            'ZZ4l.root',
      'Data.root',                              'qqH125.root']

   datasets = ["WJets", "DY", "Top", "SingleTop", "SMHiggs", "Diboson", "Data"]

   #Create empty arrays for fillings
   DY, DY_w, DY_pt, DY_eta = [], [], [], []
   WJets, WJets_w, WJets_pt, WJets_eta = [], [], [], []
   QCD, QCD_w, QCD_pt, QCD_eta = [], [], [], []
   Diboson, Diboson_w, Diboson_pt, Diboson_eta = [], [], [], []
   SMHiggs, SMHiggs_w, SMHiggs_pt, SMHiggs_eta =[], [], [], []
   SingleTop, SingleTop_w, SingleTop_pt, SingleTop_eta =[], [], [], []
   Top, Top_w, Top_pt, Top_eta = [], [], [], []
   Data, Data_w, Data_pt, Data_eta = [], [], [], []
   Data_SS, DY_SS, Top_SS, WJets_SS = [], [], [], []
   DY_SS_w, Top_SS_w, WJets_SS_w = [], [], []

   #Get pileup reweighting factors
   with uproot.open("pu_distributions_mc_2018.root") as f1:
      with uproot.open("pu_distributions_data_2018.root") as f2:
         mc = f1["pileup"].values()
         data = f2["pileup"].values()
         HistoPUMC = np.divide(mc, ak.sum(mc))
         HistoPUData = np.divide(data, ak.sum(data))
         PUWeight = np.divide(HistoPUData, HistoPUMC)


   #read in file
   for sample in fileList:
      fname = os.path.join(directory, sample)
      file = uproot.open(fname)
      events = NanoEventsFactory.from_root(
         file,
         treepath="/EventTree",
         schemaclass=BaseSchema,
         metadata={"dataset": dataset},
      ).events()
      string = str(sample)
      num_events = file['hEvents'].member('fEntries') / 2

      #Read out and sort
      out = p.process(events, fname, num_events, PUWeight)
      if "DY" in string:
            DY = np.append(DY, out[dataset]["mass"], axis=0)
            DY_w = np.append(DY_w, out[dataset]["mass_w"], axis=0)
            DY_pt = np.append(DY_pt, out[dataset]["pT"], axis=0)
            DY_eta = np.append(DY_eta, out[dataset]["eta"], axis=0)
            DY_SS = np.append(DY_SS, out[dataset]["ss_mass"], axis=0)
            DY_SS_w = np.append(DY_SS_w, out[dataset]["ss_mass_w"], axis=0)
      if "WJets" in string:
            WJets = np.append(WJets, out[dataset]["mass"], axis=0)
            WJets_w = np.append(WJets_w, out[dataset]["mass_w"], axis=0)
            WJets_pt = np.append(WJets_pt, out[dataset]["pT"], axis=0)
            WJets_eta = np.append(WJets_eta, out[dataset]["eta"], axis=0)
            WJets_SS = np.append(WJets_SS, out[dataset]["ss_mass"], axis=0)
            WJets_SS_w = np.append(WJets_SS_w, out[dataset]["ss_mass_w"], axis=0)
      matches = ["WZ", "VV2l2nu", "ZZ4l", "ZZ2l2q"]
      if any([x in fname for x in matches]):
            Diboson = np.append(Diboson, out[dataset]["mass"], axis=0)
            Diboson_w = np.append(Diboson_w, out[dataset]["mass_w"], axis=0)
            Diboson_pt = np.append(Diboson_pt, out[dataset]["pT"], axis=0)
            Diboson_eta = np.append(Diboson_eta, out[dataset]["eta"], axis=0)
      matches = ["gg", "qqH125", "toptop", "WMinus", "WPlus", "ZH125", "TauTau"] 
      if any([x in fname for x in matches]):
            SMHiggs = np.append(SMHiggs, out[dataset]["mass"], axis=0)
            SMHiggs_w = np.append(SMHiggs_w, out[dataset]["mass_w"], axis=0) 
            SMHiggs_pt = np.append(SMHiggs_pt, out[dataset]["pT"], axis=0)
            SMHiggs_eta = np.append(SMHiggs_eta, out[dataset]["eta"], axis=0)
      matches = ["Tbar", "T-tchan", "tW"]
      if any([x in fname for x in matches]):
            SingleTop = np.append(SingleTop, out[dataset]["mass"], axis=0)
            SingleTop_w = np.append(SingleTop_w, out[dataset]["mass_w"], axis=0) 
            SingleTop_pt = np.append(SingleTop_pt, out[dataset]["pT"], axis=0)
            SingleTop_eta = np.append(SingleTop_eta, out[dataset]["eta"], axis=0)
      if "TTTo" in string:
            Top = np.append(Top, out[dataset]["mass"], axis=0)
            Top_w = np.append(Top_w, out[dataset]["mass_w"], axis=0) 
            Top_pt = np.append(Top_pt, out[dataset]["pT"], axis=0)
            Top_eta = np.append(Top_eta, out[dataset]["eta"], axis=0)
            Top_SS = np.append(Top_SS, out[dataset]["ss_mass"], axis=0)
            Top_SS_w = np.append(Top_SS_w, out[dataset]["ss_mass_w"], axis=0)
      if "Data" in string:
            Data = np.append(Data, out[dataset]["mass"], axis=0)
            Data_w = np.append(Data_w, out[dataset]["mass_w"], axis=0)
            Data_pt = np.append(Data_pt, out[dataset]["pT"], axis=0)
            Data_eta = np.append(Data_eta, out[dataset]["eta"], axis=0)
            Data_SS = np.append(Data_SS, out[dataset]["ss_mass"], axis=0)
   

   bins = np.linspace(60, 120, 60)

   QCDScaleFactor = 1.6996559936491136


   #Create histograms
   Data_h, Data_bins = np.histogram(Data, bins=bins)     
   DY_h, DY_bins = np.histogram(DY, bins=bins, weights=DY_w)
   WJets_h, WJets_bins = np.histogram(WJets, bins=bins, weights=WJets_w) 
   Top_h, Top_bins = np.histogram(Top, bins=bins, weights= Top_w) 
   SingleTop_h, SingleTop_bins = np.histogram(SingleTop, bins=bins, weights=SingleTop_w) 
   Diboson_h, Diboson_bins = np.histogram(Diboson, bins=bins, weights=Diboson_w)
   SMHiggs_h, SMHiggs_bins = np.histogram(SMHiggs, bins=bins, weights=SMHiggs_w)   


   #QCD Estimation
   #NonQCD = np.append(np.append(DY_SS, WJets_SS, axis=0), Top_SS, axis=0)
   #NonQCD_w = np.append(np.append(DY_SS_w, WJets_SS_w, axis=0), Top_SS_w, axis=0) 
   Data_SS_h, Data_SS_bins = np.histogram(Data_SS, bins=bins)
   DY_SS_h, DY_SS_bins = np.histogram(DY_SS, bins=bins, weights=DY_SS_w)
   WJets_SS_h, WJets_SS_bins = np.histogram(WJets_SS, bins=bins, weights=WJets_SS_w)
   Top_SS_h, Top_SS_bins = np.histogram(Top_SS, bins=bins, weights=Top_SS_w)
   #NonQCD_h, NonQCD_bins = np.histogram(NonQCD, bins=bins)

   QCD_h = np.subtract(np.subtract(np.subtract(Data_SS_h, DY_SS_h, dtype=object, out=None), WJets_SS_h, dtype=object, out=None), Top_SS_h, dtype=object, out=None)
   print("QCD: ", QCD_h)
   for i in range(QCD_h.size):
      if QCD_h[i] < 0.0:
         QCD_h[i] = 0.0
   QCD_w = np.full(shape=QCD_h.shape, fill_value=QCDScaleFactor, dtype=np.double)
   QCD_hist = (QCD_h, Data_SS_bins)


   #Mass and labels for plotting
   mass =   [SMHiggs_h,   Diboson_h,  SingleTop_h,   DY_h,   Top_h, WJets_h, QCD_h]
   labels = ["SMHiggs", "Diboson", "SingleTop", "DY", "Top", "WJets", "QCD"]

   #Plot MuMu Visible Mass
   hep.histplot(mass, label=labels, histtype=("fill"), bins=bins, stack=True)
   hep.histplot(Data_h, label="Data", histtype=("errorbar"), bins=bins, color="k")
   plt.legend(loc = 'upper right', ncols = 2, fontsize = 8)
   #plt.yscale("log")
   #OS boostedTau visible mass
   plt.title("DiMuon Visible Mass", fontsize= 'small')
   ax.set_xlabel("Mass (GeV)")
   ax.set_ylim(bottom=0)
   fig.savefig("./mumu_plots/DIMUON_VISIBLE_MASS.png")
   plt.clf()


   #Plot MuMu SS Visible Mass
   ss_mass = [DY_SS_h, Top_SS_h, WJets_SS_h]
   ss_mass_labels = ["DY", "Top", "WJets"]

   hep.histplot(ss_mass, label=ss_mass_labels, histtype=("fill"), bins=bins, stack=True)
   hep.histplot(Data_SS_h, label="Data", histtype=("errorbar"), bins=bins, color="k")
   plt.legend(loc = 'upper right', ncols = 2, fontsize = 8)

   plt.title("DiMuon Visible Mass (SS Region)", fontsize= 'small')
   ax.set_xlabel("Mass (GeV)")
   ax.set_ylim(bottom=0)
   fig.savefig("./mumu_plots/SS_DIMUON_VISIBLE_MASS.png")
   plt.clf()


   #Write to file
   oneBin = np.linspace(60, 120, num=2)
   outFile = uproot.recreate("boostedHTT_mm_2018.input.root")
   DY_hist, DY_bins = np.histogram(DY, bins=oneBin, weights=DY_w)
   print("DY: ", DY_bins)
   QCD_h = np.array([np.sum(QCD_h)])
   QCD_hist = (QCD_h, DY_bins)
   print("QCD: ", QCD_hist)
   TT_h = np.histogram(Top, bins=oneBin, weights=Top_w)
   print("TT: ", TT_h)
   VV_h = np.histogram(np.append(Diboson, SingleTop), bins=oneBin, weights=np.append(Diboson_w, SingleTop_w))
   WJets_h = np.histogram(WJets, bins=oneBin, weights=WJets_w)
   Data_h = np.histogram(Data, bins=oneBin)
   print("Data: ", Data_h)
   outFile["DY_Jets_mm_1_13TeV/DYJets125"] = (DY_hist, DY_bins)
   outFile["DY_Jets_mm_1_13TeV/TT"] = TT_h
   outFile["DY_Jets_mm_1_13TeV/VV"] = VV_h
   outFile["DY_Jets_mm_1_13TeV/W"] = WJets_h
   outFile["DY_Jets_mm_1_13TeV/QCD"] = QCD_hist
   outFile["DY_Jets_mm_1_13TeV/data_obs"] = Data_h


   #Plot visible mass with Abdollah's specifications
   newMass = [VV_h, TT_h, WJets_h, (DY_hist, DY_bins), QCD_hist]
   labels = ["VV", "TT", "WJets", "DY", "QCD"]
   hep.histplot(newMass, label=labels, histtype=("fill"), stack=True)
   hep.histplot(Data_h, label="Data", histtype=("errorbar"), color="k")
   plt.legend(loc = 'upper right', ncols = 2, fontsize = 8)
   #plt.yscale("log")
   #OS boostedTau visible mass
   plt.title("DiMuon Visible Mass", fontsize= 'small')
   ax.set_xlabel("Mass (GeV)")
   ax.set_ylim(bottom=0)
   fig.savefig("./mumu_plots/mumu_VISIBLE_MASS.png")
   plt.clf()




   #More histos
   bins = np.linspace(200, 1000, 80)
   Data_h, Data_bins = np.histogram(Data_pt, bins=bins)
   DY_h, DY_bins = np.histogram(DY_pt, bins=bins, weights=DY_w)
   WJets_h, WJets_bins = np.histogram(WJets_pt, bins=bins, weights=WJets_w) 
   Top_h, Top_bins = np.histogram(Top_pt, bins=bins, weights= Top_w) 
   SingleTop_h, SingleTop_bins = np.histogram(SingleTop_pt, bins=bins, weights=SingleTop_w) 
   Diboson_h, Diboson_bins = np.histogram(Diboson_pt, bins=bins, weights=Diboson_w)
   SMHiggs_h, SMHiggs_bins = np.histogram(SMHiggs_pt, bins=bins, weights=SMHiggs_w)   

   #PT
   pT =   [SMHiggs_h,   Diboson_h,   SingleTop_h,   DY_h,   Top_h,   WJets_h]
   labels = ["SMHiggs", "Diboson", "SingleTop", "DY", "Top", "WJets"]
   hep.histplot(pT, label=labels, histtype=("fill"), bins=bins, stack=True)
   hep.histplot(Data_h, label="Data", histtype=("errorbar"), bins=bins, color='k')
   plt.yscale("log")
   plt.legend(loc = 'upper right', ncols = 2, fontsize = 8)
   ax.set_ylim(bottom=0)
   #OS boostedTau pT fakerate
   plt.title("DiMuon pT", fontsize= 'small')
   ax.set_xlabel("pT (GeV)")
   fig.savefig("./mumu_plots/DIMUON_pT_log.png")
   plt.clf()

   #Eta
   bins = np.linspace(-3, 3, 20)
   Data_h, Data_bins = np.histogram(Data_eta, bins=bins)
   DY_h, DY_bins = np.histogram(DY_eta, bins=bins, weights=DY_w)
   WJets_h, WJets_bins = np.histogram(WJets_eta, bins=bins, weights=WJets_w) 
   Top_h, Top_bins = np.histogram(Top_eta, bins=bins, weights= Top_w) 
   SingleTop_h, SingleTop_bins = np.histogram(SingleTop_eta, bins=bins, weights=SingleTop_w) 
   Diboson_h, Diboson_bins = np.histogram(Diboson_eta, bins=bins, weights=Diboson_w)
   SMHiggs_h, SMHiggs_bins = np.histogram(SMHiggs_eta, bins=bins, weights=SMHiggs_w)   

   eta =   [SMHiggs_h,   Diboson_h,   SingleTop_h,   DY_h,   Top_h,   WJets_h]
   hep.histplot(eta, label=labels, histtype=("fill"), bins=bins, stack=True)
   hep.histplot(Data_h, label="Data", histtype=("errorbar"), bins=bins, color='k')
   plt.legend(loc = 'upper right', ncols = 2, fontsize = 8)
   ax.set_ylim(bottom=0)
   #eta
   plt.yscale("log")
   plt.title("DiMuon eta", fontsize= 'small')
   ax.set_xlabel("Radians")
   fig.savefig("./mumu_plots/DIMUON_eta_log.png")

