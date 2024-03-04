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
from matplotlib import cycler
import mplhep as hep
from coffea import processor, nanoevents, lookup_tools
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea.nanoevents.methods import vector, candidate
from coffea.nanoevents.methods.vector import PtEtaPhiELorentzVector
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
      return np.sqrt((np.subtract(candidate2.eta,candidate1.eta))**2 + (np.subtract(candidate2.phi, candidate1.phi))**2)

   def makeVector(self, particle, name, XSection):
      if name == "tau": mass = particle.mass
      if name == "muon": mass = 0.10565837
      if name == "jet": mass = particle.E
      if XSection != 1:
         newVec = ak.zip(
            {
            "pt": particle.pt,
            "eta": particle.eta,
            "phi": particle.phi,
            "mass": mass,
            "puTrue": particle.puTrue
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

   def process(self, events, name, num_events, PUWeight, en_var):
      dataset = events.metadata['dataset']
      #Call to find weight factor / XS
      XSection = self.weightCalc(name)
      print(name, " ", XSection)
      #print(events.fields)

      electron = ak.zip( 
		{
				"pt": events.elePt,
				"eta": events.eleEta,
				"phi": events.elePhi,
            "energy": events.eleEn,
            "eleSCEta": events.eleSCEta,
            "eleIDMVAIso": events.eleIDMVAIso,
		},
			   with_name="electronArray",
		      behavior=candidate.behavior,
		) 
      ele_cut = (electron.pt >= 15) & (np.abs(electron.eta) <= 2.5)
      lowMVAele = electron[(np.abs(electron.eleSCEta) <= 0.8) & (electron.eleIDMVAIso > -0.83) & ele_cut]
      midMVAele = electron[(np.abs(electron.eleSCEta) > 0.8) & (np.abs(electron.eleSCEta) <= 1.5) & (electron.eleIDMVAIso > -0.77) & ele_cut] 
      highMVAele = electron[(np.abs(electron.eleSCEta) >= 1.5) & (electron.eleIDMVAIso > -0.69) & ele_cut] 
      events = events[(ak.num(lowMVAele) == 0) & (ak.num(midMVAele) == 0) & (ak.num(highMVAele) == 0)]
      #Extra muon veto
      muon = ak.zip( 
		{
				"pt": events.muPt,
				"eta": events.muEta,
				"phi": events.muPhi,
            "energy": events.muEn,
            "muPFChIso": events.muPFChIso,
            "muPFNeuIso": events.muPFNeuIso,
            "muPFPhoIso": events.muPFPhoIso,
            "muPFPUIso": events.muPFPUIso,
            "muIDbit": events.muIDbit,
		},
			   with_name="electronArray",
		      behavior=candidate.behavior,
		)
      muon = muon[ak.num(muon.pt) > 1]
      RelIsoMu = (muon.muPFChIso + muon.muPFNeuIso + muon.muPFPhoIso - (0.5 * muon.muPFPUIso)) / muon.pt
      badMuon = muon[(np.abs(RelIsoMu) > 0.3) & (muon.pt[:,1] > 10) & (np.bitwise_and(muon.muIDbit, self.bit_mask(2)) == self.bit_mask(2))]
      events = events[(ak.num(muon) > 0) & (ak.num(badMuon) == 0)]

      #HT Cut
      jets = ak.zip( 
		{
				"pt": events.jetPt,
				"eta": events.jetEta,
				"phi": events.jetPhi,
            "energy": events.jetEn,
            "jetPFLooseId": events.jetPFLooseId,
            "jetCSV2BJetTags": events.jetCSV2BJetTags,
		},
			   with_name="JetArray",
		      behavior=candidate.behavior,
		)
      goodJets= jets[(jets.jetPFLooseId > 0.5) & (jets.pt > 30) & (np.abs(jets.eta) < 3.0)]
      HT = ak.sum(goodJets.pt, axis=-1)

      #BJet Veto
      bJets = jets[(jets.jetCSV2BJetTags > .7527) & (jets.jetPFLooseId > .5) & (jets.pt > 30) & (np.abs(jets.eta) < 2.4)]
      events = events[(ak.num(jets) > 0) & (HT > 200) & (ak.num(bJets) == 0)]

      #Create muon and tau candidates (recombine trigger separations)
      #If statement exists because Data does not have puTrue but Bkg does, need to define candidate without it
      if XSection != 1:
         muonC = ak.zip( 
			   {
				"nMuon": events.nMu,
				"charge": events.muCharge,
   		   "D0": events.muD0,
           	"Dz": events.muDz,
            "puTrue": events.puTrue[:,0],
            "muPFNeuIso": events.muPFNeuIso,
            "muPFPhoIso": events.muPFPhoIso,
            "muPFPUIso": events.muPFPUIso,
            "muPFChIso": events.muPFChIso,
            "muIDbit": events.muIDbit,
            "Trigger": events.HLTEleMuX,
            "met": events.pfMetNoRecoil,
			   },
			   with_name="muCandidate",
			   behavior=candidate.behavior,
		   )
         tauC = ak.zip( 
			   {
				"nBoostedTau": events.nBoostedTau,
				"charge": events.boostedTauCharge,
				"iso": events.boostedTaupfTausDiscriminationByDecayModeFinding,
            "antiMu": events.boostedTauByLooseMuonRejection3,
            "puTrue": events.puTrue[:,0],
            "Met": events.pfMetNoRecoil,
            "Metphi": events.pfMetPhiNoRecoil,
			   },
			   with_name="muCandidate",
			   behavior=candidate.behavior,
		   )
      else:
         muonC = ak.zip( 
			   {
				"nMuon": events.nMu,
				"charge": events.muCharge,
   		   "D0": events.muD0,
           	"Dz": events.muDz,
            "muPFNeuIso": events.muPFNeuIso,
            "muPFPhoIso": events.muPFPhoIso,
            "muPFPUIso": events.muPFPUIso,
            "muPFChIso": events.muPFChIso,
            "muIDbit": events.muIDbit,
            "Trigger": events.HLTEleMuX,
            "met": events.pfMetNoRecoil,
			   },
			   with_name="muCandidate",
			   behavior=candidate.behavior,
		   )
         tauC = ak.zip( 
			   {
				"nBoostedTau": events.nBoostedTau,
				"charge": events.boostedTauCharge,
				"iso": events.boostedTaupfTausDiscriminationByDecayModeFinding,
            "antiMu": events.boostedTauByLooseMuonRejection3,
            "Met": events.pfMetNoRecoil,
            "Metphi": events.pfMetPhiNoRecoil,
			   },
			   with_name="tauCandidate",
			   behavior=candidate.behavior,
		   )

      muon = ak.zip( 
			{
			"pt": events.muPt,
         "eta": events.muEta,
			"phi": events.muPhi,
			"energy": events.muEn,
		   },
		   with_name="PtEtaPhiELorentzVector",
         behavior=vector.behavior,
		)
      tau = ak.zip( 
			{
			"pt": events.boostedTauPt,
         "eta": events.boostedTauEta,
         "phi": events.boostedTauPhi,
			"energy": events.boostedTauEnergy,
		   },
		   with_name="PtEtaPhiELorentzVector",
		   behavior=vector.behavior,
		)

      if XSection != 1:
         #Gen-tau matching for systematics - boosted Tau Energy
         genTauExist = ak.any(events.mcPID == 15, axis=-1)
         genTauIndex = events.mcPID == 15
         allEta= ak.cartesian({'tau': events.boostedTauEta, 'mc': events.mcEta[genTauIndex]}, axis=1, nested=True)
         allPhi= ak.cartesian({'tau': events.boostedTauPhi, 'mc': events.mcPhi[genTauIndex]}, axis=1, nested=True)
         deltaR = np.sqrt((np.subtract(allEta['tau'], allEta['mc']))**2 + (np.subtract(allPhi['tau'], allPhi['mc']))**2)
         genTauCut = ak.min(deltaR, axis=-1) < .1
         newtau = ak.where(genTauExist & genTauCut, tau.multiply(events[f"{en_var}"]), tau)
         tau = ak.where(genTauExist, newtau, tau)





      #Split into pairs
      pairs = ak.cartesian({'tau': tau, 'muon': muon}, nested=False)
      pairsC = ak.cartesian({'tau': tauC, 'muon': muonC}, nested=False) 
      #Trigger Cut
      trigger_mask_Mu27 = self.bit_mask(19)
      trigger_mask_Mu50 = self.bit_mask(21)

      IsoCheck = ak.any(((pairsC['muon'].muPFNeuIso + pairsC['muon'].muPFPhoIso - 0.5 * pairsC['muon'].muPFPUIso) > 0.0), axis=-1)
      IsoLep1Val = np.divide(pairsC['muon'].muPFChIso, pairs['muon'].pt)
      IsoLep1Val = np.where(IsoCheck, (pairsC['muon'].muPFChIso + pairsC['muon'].muPFNeuIso + pairsC['muon'].muPFPhoIso  - 0.5 * pairsC['muon'].muPFPUIso) / pairs['muon'].pt, IsoLep1Val)
      #Delta r cut
      dr = pairs['tau'].delta_r(pairs['muon'])
      #ID Cut
      MuID = ((np.bitwise_and(pairsC['muon'].muIDbit, self.bit_mask(1)) == self.bit_mask(1)) & (np.abs(pairsC['muon'].Dz) < 0.2) & (np.abs(pairsC['muon'].D0) < 0.045))
      #Apply everything at once
      muTau_mask27 = ((np.abs(pairs['muon'].eta) < 2.4)
                  & (MuID)
                  & (pairs['muon'].pt >= 27)
                  & (pairs['muon'].pt < 52)
                  & (IsoLep1Val < .3)
                  & (pairsC['muon'].met > 30)
                  & (pairs['tau'].pt > 30)
                  & (np.bitwise_and(pairsC['muon'].Trigger, trigger_mask_Mu27) == trigger_mask_Mu27)
                  & (np.absolute(pairs['tau'].eta) < 2.3)
                  & (pairsC['tau'].antiMu >= 0.5)
                  & (pairsC['tau'].iso >= 0.5)
                  & (dr > .1) 
                  & (dr < .8))
      muTau_mask50 = ((np.abs(pairs['muon'].eta) < 2.4)
                  & (MuID)
                  & (pairs['muon'].pt >= 52)
                  & ((np.bitwise_and(pairsC['muon'].Trigger, trigger_mask_Mu50) == trigger_mask_Mu50))
                  & (pairs['tau'].pt > 30)
                  & (np.absolute(pairs['tau'].eta) < 2.3)
                  & (pairsC['tau'].antiMu >= 0.5)
                  & (pairsC['tau'].iso >= 0.5)
                  & (dr > .1) 
                  & (dr < .8))

      #If everything cut return 0 (to avoid pointer errors)
      if not ak.any(muTau_mask27, axis=-1) and not ak.any(muTau_mask50, axis=-1):
         print("0")
         return {
            dataset: {
            "mass": np.zeros(0),
            "mass_w": np.zeros(0),
            "ss_mass": np.zeros(0),
            "ss_mass_w": np.zeros(0),
            "Mu50IsoCorr": np.zeros(0),
            "Mu50IDCorr": np.zeros(0),
            "Mu50TrgCorr": np.zeros(0),
            "PUCorrection50": np.zeros(0),
            "Mu27IsoCorr": np.zeros(0),
            "Mu27IDCorr": np.zeros(0),
            "Mu27TrgCorr": np.zeros(0),
            "PUCorrection27": np.zeros(0),
            "pTCorrection50": np.zeros(0),
            "pTCorrection27": np.zeros(0),
            }
         }

      
      pairs = pairs[(muTau_mask27) | (muTau_mask50)]
      pairsC = pairsC[(muTau_mask27) | (muTau_mask50)]
      print(ak.count(pairs['muon'].pt))
      #Separate based on charge
      OS_pairs = pairs[pairsC['tau'].charge + pairsC['muon'].charge == 0]
      SS_pairs = pairs[pairsC['tau'].charge + pairsC['muon'].charge != 0]
      OS_pairsC = pairsC[pairsC['tau'].charge + pairsC['muon'].charge == 0]
      SS_pairsC = pairsC[pairsC['tau'].charge + pairsC['muon'].charge != 0]

      #Separate based on trigger again
      OS_50 = OS_pairs[OS_pairs['muon'].pt >= 52]
      OS_27 = OS_pairs[OS_pairs['muon'].pt < 52]
      SS_50 = SS_pairs[SS_pairs['muon'].pt >= 52]
      SS_27 = SS_pairs[SS_pairs['muon'].pt < 52]

      OS_50C = OS_pairsC[OS_pairs['muon'].pt >= 52]
      OS_27C = OS_pairsC[OS_pairs['muon'].pt < 52]
      SS_50C = SS_pairsC[SS_pairs['muon'].pt >= 52]
      SS_27C = SS_pairsC[SS_pairs['muon'].pt < 52]

      #If everything cut return 0 (to avoid pointer errors)
      if ak.sum(OS_pairs['tau'].pt) == 0 and ak.sum(SS_pairs['tau'].pt) == 0:
         return {
            dataset: {
            "mass": np.zeros(0),
            "mass_w": np.zeros(0),
            "ss_mass": np.zeros(0),
            "ss_mass_w": np.zeros(0),
            "Mu50IsoCorr": np.zeros(0),
            "Mu50IDCorr": np.zeros(0),
            "Mu50TrgCorr": np.zeros(0),
            "PUCorrection50": np.zeros(0),
            "Mu27IsoCorr": np.zeros(0),
            "Mu27IDCorr": np.zeros(0),
            "Mu27TrgCorr": np.zeros(0),
            "PUCorrection27": np.zeros(0),
            "pTCorrection50": np.zeros(0),
            "pTCorrection27": np.zeros(0),
            }
         }

      #Create vectors
      #OS
      muVec50 = OS_50['muon']
      tauVec50 = OS_50['tau']
      muVec27 = OS_27['muon']
      tauVec27 = OS_27['tau']

      #TMass Cut
      tmass50 = np.sqrt(np.square(OS_50['muon'].pt + OS_50C['tau'].Met) - np.square(OS_50['muon'].px + OS_50C['tau'].Met * np.cos(OS_50C['tau'].Metphi)) - np.square(OS_50['muon'].py + OS_50C['tau'].Met * np.sin(OS_50C['tau'].Metphi)))
      tmass27 = np.sqrt(np.square(OS_27['muon'].pt + OS_27C['tau'].Met) - np.square(OS_27['muon'].px + OS_27C['tau'].Met * np.cos(OS_27C['tau'].Metphi)) - np.square(OS_27['muon'].py + OS_27C['tau'].Met * np.sin(OS_27C['tau'].Metphi)))
      OS_50 = OS_50[tmass50 < 80]
      OS_27 = OS_27[tmass27 < 80]
      OS_50C = OS_50C[tmass50 < 80]
      OS_27C = OS_27C[tmass27 < 80]

      tmass50_SS = np.sqrt(np.square(SS_50['muon'].pt + SS_50C['tau'].Met) - np.square(SS_50['muon'].px + SS_50C['tau'].Met * np.cos(SS_50C['tau'].Metphi)) - np.square(SS_50['muon'].py + SS_50C['tau'].Met * np.sin(SS_50C['tau'].Metphi)))
      tmass27_SS = np.sqrt(np.square(SS_27['muon'].pt + SS_27C['tau'].Met) - np.square(SS_27['muon'].px + SS_27C['tau'].Met * np.cos(SS_27C['tau'].Metphi)) - np.square(SS_27['muon'].py + SS_27C['tau'].Met * np.sin(SS_27C['tau'].Metphi)))

      SS_50 = SS_50[tmass50_SS < 80]
      SS_27 = SS_27[tmass27_SS < 80]
      SS_50C = SS_50C[tmass50_SS < 80]
      SS_27C = SS_27C[tmass27_SS < 80]

      #MET Vector
      MetVec27 =  ak.zip(
      {
         "pt": OS_27C['tau'].Met,
         "eta": 0,
         "phi": OS_27C['tau'].Metphi,
         "mass": 0,
         },
      with_name="PtEtaPhiMLorentzVector",
      behavior=vector.behavior,
      ) 
      MetVec50 =  ak.zip(
      {
         "pt": OS_50C['tau'].Met,
         "eta": 0,
         "phi": OS_50C['tau'].Metphi,
         "mass": 0,
         },
      with_name="PtEtaPhiMLorentzVector",
      behavior=vector.behavior,
      )   
      MetVec27_SS =  ak.zip(
      {
         "pt": SS_27C['tau'].Met,
         "eta": 0,
         "phi": SS_27C['tau'].Metphi,
         "mass": 0,
         },
      with_name="PtEtaPhiMLorentzVector",
      behavior=vector.behavior,
      )  
      MetVec50_SS =  ak.zip(
      {
         "pt": SS_50C['tau'].Met,
         "eta": 0,
         "phi": SS_50C['tau'].Metphi,
         "mass": 0,
         },
      with_name="PtEtaPhiMLorentzVector",
      behavior=vector.behavior,
      ) 

      #Make Z Vector
      #OS
      ZVec50 = OS_50['tau'].add(OS_50['muon'])
      ZVec27 = OS_27['tau'].add(OS_27['muon'])
      #Make Higgs Vector
      Higgs50 = ZVec50.add(MetVec50)
      Higgs27 = ZVec27.add(MetVec27)


      #SS
      ZVec50_SS = SS_50['tau'].add(SS_50['muon'])
      ZVec27_SS = SS_27['tau'].add(SS_27['muon'])
      #Make Higgs Vector
      Higgs50_SS = ZVec50_SS.add(MetVec50_SS)
      Higgs27_SS = ZVec27_SS.add(MetVec27_SS)


      #Final ZVec Cuts
      ZVec50 = ZVec50[(ZVec50.pt > 200) & (Higgs50.pt > 250)]
      ZVec27 = ZVec27[(ZVec27.pt > 200) & (Higgs27.pt > 250)]
      ZVec50_SS = ZVec50_SS[(ZVec50_SS.pt > 200) & (Higgs50_SS.pt > 250)]
      ZVec27_SS = ZVec27_SS[(ZVec27_SS.pt > 200) & (Higgs27_SS.pt > 250)]



      #Add weight sets
      ext = extractor()
      ext.add_weight_sets(["IDCorr NUM_LooseID_DEN_genTracks_pt_abseta ./RunBCDEF_SF_ID.root", "Trg50Corr Mu50_OR_TkMu50_PtEtaBins/pt_abseta_ratio ./Trigger_EfficienciesAndSF_RunBtoF.root", "Trg27Corr IsoMu24_OR_IsoTkMu24_PtEtaBins/pt_abseta_ratio ./Trigger_EfficienciesAndSF_RunBtoF.root", "IsoCorr NUM_LooseRelIso_DEN_LooseID_pt_abseta ./RunBCDEF_SF_ISO.root", "pTCorr Ratio2D ./zmm_2d_2018.root"])
      ext.finalize()
      evaluator = ext.make_evaluator()

      #Get Shape
      shape = np.shape(np.append(ak.flatten(ZVec27.mass, axis=1), ak.flatten(ZVec50.mass, axis=1)))
      SS_shape = np.shape(np.append(ak.flatten(ZVec27_SS.mass, axis=1), ak.flatten(ZVec50_SS.mass, axis=1)))

      Mu50IsoCorr, Mu50IDCorr, Mu50TrgCorr, PUCorrection50, pTCorrection50 = [], [], [], [], []
      Mu27IsoCorr, Mu27IDCorr, Mu27TrgCorr, PUCorrection27, pTCorrection27 = [], [], [], [], []

      #If not data, calculate all weight corrections
      if XSection != 1:

         #Shortcut names to make things easier
         ZVec50pt = ak.flatten(ZVec50.pt, axis=-1)
         ZVec50eta = ak.flatten(ZVec50.eta, axis=-1)
         ZVec50_SSpt = ak.flatten(ZVec50_SS.pt, axis=-1)
         ZVec50_SSeta = ak.flatten(ZVec50_SS.eta, axis=-1)
         ZVec27pt = ak.flatten(ZVec27.pt, axis=-1)
         ZVec27eta = ak.flatten(ZVec27.eta, axis=-1)
         ZVec27_SSpt = ak.flatten(ZVec27_SS.pt, axis=-1)
         ZVec27_SSeta = ak.flatten(ZVec27_SS.eta, axis=-1)

         #OS
         Mu50IsoCorr = evaluator["IsoCorr"](ZVec50pt, ZVec50eta)   #Iso
         Mu50TrgCorr = evaluator["Trg50Corr"](ZVec50pt, ZVec50eta) #Trigger
         Mu50IDCorr = evaluator["IDCorr"](ZVec50pt, ZVec50eta) #ID
         #Mu27IsoCorr = evaluator["IsoCorr"](ZVec27pt, ZVec27eta) 
         Mu27TrgCorr = evaluator["Trg27Corr"](ZVec27pt, ZVec27eta) 
         Mu27IDCorr = evaluator["IDCorr"](ZVec27pt, ZVec27eta) 

         puTrue50 = np.array(np.rint(ak.flatten(OS_50C['muon'][ZVec50.pt > 200].puTrue, axis=-1)), dtype=np.int8) #Pileup
         puTrue27 = np.array(np.rint(ak.flatten(OS_27C['muon'][ZVec27.pt > 200].puTrue, axis=-1)), dtype=np.int8)
         PUCorrection50 = PUWeight[puTrue50]
         PUCorrection27 = PUWeight[puTrue27]

         Lep50Corr = Mu50IsoCorr * Mu50TrgCorr * Mu50IDCorr * PUCorrection50 #Combine
         Lep27Corr = Mu27TrgCorr * Mu27IDCorr * PUCorrection27

         #SS
         Mu50IsoCorr_SS = evaluator["IsoCorr"](ZVec50_SSpt, ZVec50_SSeta)   
         Mu50TrgCorr_SS = evaluator["Trg50Corr"](ZVec50_SSpt, ZVec50_SSeta)   
         Mu50IDCorr_SS = evaluator["IDCorr"](ZVec50_SSpt, ZVec50_SSeta)  
         Mu27IsoCorr_SS = evaluator["IsoCorr"](ZVec27_SSpt, ZVec27_SSeta) 
         Mu27TrgCorr_SS = evaluator["Trg27Corr"](ZVec27_SSpt, ZVec27_SSeta)  
         Mu27IDCorr_SS = evaluator["IDCorr"](ZVec27_SSpt, ZVec27_SSeta) 

         puTrue50_SS = np.array(np.rint(ak.flatten(SS_50C['muon'][ZVec50_SS.pt > 200].puTrue, axis=-1)), dtype=np.int8)
         puTrue27_SS = np.array(np.rint(ak.flatten(SS_27C['muon'][ZVec27_SS.pt > 200].puTrue, axis=-1)), dtype=np.int8)
         PUCorrection50_SS = PUWeight[puTrue50_SS]
         PUCorrection27_SS = PUWeight[puTrue27_SS]

         Lep50Corr_SS = Mu50IsoCorr_SS * Mu50TrgCorr_SS * Mu50IDCorr_SS * PUCorrection50_SS
         Lep27Corr_SS = Mu27IsoCorr_SS * Mu27TrgCorr_SS * Mu27IDCorr_SS * PUCorrection27_SS

         #Luminosity (2018)
         luminosity = 59830.
         lumiWeight = (XSection * luminosity) / num_events
         
         LepCorrection = np.append(ak.flatten(Lep27Corr, axis=-1), ak.flatten(Lep50Corr, axis=-1), axis=-1) 
         #OS
         if ("DYJets" in name): 
            #Zpt correction
            pTCorrection27 = evaluator["pTCorr"](ak.flatten(ZVec27.mass, axis=-1), ZVec27pt)
            pTCorrection50 = evaluator["pTCorr"](ak.flatten(ZVec50.mass, axis=-1), ZVec50pt)
            Lep27Corr = Lep27Corr * pTCorrection27
            Lep50Corr = Lep50Corr * pTCorrection50
            LepCorrection = np.append(ak.flatten(Lep27Corr, axis=-1), ak.flatten(Lep50Corr, axis=-1), axis=-1)
         
         #SS
         SS_LepCorrection = np.append(ak.flatten(Lep27Corr_SS, axis=-1), ak.flatten(Lep50Corr_SS, axis=-1), axis=-1) 

         #Get shape
         mass_w = np.full(shape=shape, fill_value=lumiWeight, dtype=np.double)
         mass_w = np.multiply(mass_w, ak.flatten(LepCorrection, axis=-1))
         SS_mass_w = np.full(shape=SS_shape, fill_value=lumiWeight, dtype=np.double) 
         SS_mass_w = np.multiply(SS_mass_w, ak.flatten(SS_LepCorrection, axis=-1))
      else:
         mass_w = np.full(shape=shape, fill_value=1, dtype=np.double)
         SS_mass_w = np.full(shape=SS_shape, fill_value=1, dtype=np.double) 


      print(ak.count(ZVec27.mass), " ", ak.count(ZVec50.mass))

      #Return
      return {
         dataset: {
            "mass": np.append(ak.flatten(ZVec27.mass, axis=-1), ak.flatten(ZVec50.mass, axis=-1), axis=-1),
            "mass_w": mass_w,
            "ss_mass": np.append(ak.flatten(ZVec27_SS.mass, axis=-1), ak.flatten(ZVec50_SS.mass, axis=-1), axis=-1),
            "ss_mass_w": SS_mass_w,
            "Mu50IsoCorr": Mu50IsoCorr,
            "Mu50IDCorr": Mu50IDCorr,
            "Mu50TrgCorr": Mu50TrgCorr,
            "PUCorrection50": PUCorrection50,
            "Mu27IsoCorr": Mu27IsoCorr,
            "Mu27IDCorr": Mu27IDCorr,
            "Mu27TrgCorr": Mu27TrgCorr,
            "PUCorrection27": PUCorrection27,
            "pTCorrection50": pTCorrection50,
            "pTCorrection27": pTCorrection27,
         }
      }
   
   def postprocess(self, accumulator):
      pass
   

if __name__ == "__main__":
   directory = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedH3/2018/mt/v2_fast_Hadd"
   dataset = "SingleMuon"
   fig, ax = plt.subplots()
   #hep.style.use(hep.style.ROOT)
   plt.style.use('seaborn-v0_8-colorblind')
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
      'qqH125.root',                            'WJetsToLNu_HT-600To800.root',
      'SingleMuon_Run2018A-17Sep2018-v2.root',  'WJetsToLNu_HT-800To1200.root',
      'SingleMuon_Run2018B-17Sep2018-v1.root',  'WMinusH125.root',
      'SingleMuon_Run2018C-17Sep2018-v1.root',  'WPlusH125.root',
      'SingleMuon_Run2018D-22Jan2019-v2.root',  'WZ1l1nu2q.root',
      'WZ1l3nu.root',                           'WZ2l2q.root',
      'WZ3l1nu.root',                           'ZH125.root',
      'ZZ2l2q.root',                            'ZZ4l.root']

   #Create empty arrays for data
   DY, DY_w = [], []
   WJets, WJets_w = [], []
   Diboson, Diboson_w = [], []
   SMHiggs, SMHiggs_w = [], []
   SingleTop, SingleTop_w = [], []
   Top, Top_w = [], []
   Data, Data_w = [], []
   Data_SS, DY_SS, Top_SS, WJets_SS = [], [], [], []
   DY_SS_w, Top_SS_w, WJets_SS_w = [], [], []
   Mu50IsoCorr, Mu50IDCorr, Mu50TrgCorr, PUCorrection50, pTCorrection50 = [], [], [], [], []
   Mu27IsoCorr, Mu27IDCorr, Mu27TrgCorr, PUCorrection27, pTCorrection27 = [], [], [], [], []
   bins=np.linspace(0, 150, 30)

   #Get Pileup Weight
   with uproot.open("pu_distributions_mc_2018.root") as f1:
      with uproot.open("pu_distributions_data_2018.root") as f2:
         mc = f1["pileup"].values()
         data = f2["pileup"].values()
         HistoPUMC = np.divide(mc, ak.sum(mc))
         HistoPUData = np.divide(data, ak.sum(data))
         PUWeight = np.divide(HistoPUData, HistoPUMC)

   en_variations = ["en_nominal", "en_scale_up", "en_scale_down"]


   d= {}
   for en_var in en_variations:
      print("Tau Energy Scale: ", en_var)
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

         events["en_scale_down"] = 0.97
         events["en_nominal"] = 1.00
         events["en_scale_up"] = 1.03

         out = p.process(events, fname, num_events, PUWeight, en_var)


         d["Mu50IsoCorr" + en_var] = np.append(Mu50IsoCorr, out[dataset]["Mu50IsoCorr"], axis=0)
         d["Mu50IDCorr" + en_var] = np.append(Mu50IDCorr, out[dataset]["Mu50IDCorr"], axis=0)
         d["Mu50TrgCorr" + en_var] = np.append(Mu50TrgCorr, out[dataset]["Mu50TrgCorr"], axis=0)
         d["PUCorrection50" + en_var] = np.append(PUCorrection50, out[dataset]["PUCorrection50"], axis=0)
         d["Mu27IsoCorr" + en_var] = np.append(Mu27IsoCorr, out[dataset]["Mu27IsoCorr"], axis=0)
         d["Mu27IDCorr" + en_var]  = np.append(Mu27IDCorr, out[dataset]["Mu27IDCorr"], axis=0)
         d["Mu27TrgCorr" + en_var] = np.append(Mu27TrgCorr, out[dataset]["Mu27TrgCorr"], axis=0)
         d["PUCorrection27" + en_var] = np.append(PUCorrection27, out[dataset]["PUCorrection27"], axis=0)
         d["ptCorrection50" + en_var] = np.append(pTCorrection50, out[dataset]["pTCorrection50"], axis=0)
         d["ptCorrection27" + en_var]= np.append(pTCorrection27, out[dataset]["pTCorrection27"], axis=0)

         #Sort files in their respective arrays
         if "DY" in string:
               d["DY" + en_var] = np.append(DY, out[dataset]["mass"], axis=0)
               d["DY_w" + en_var] = np.append(DY_w, out[dataset]["mass_w"], axis=0)
               d["DY_SS" + en_var] = np.append(DY_SS, out[dataset]["ss_mass"], axis=0)
               d["DY_SS_w" + en_var] = np.append(DY_SS_w, out[dataset]["ss_mass_w"], axis=0)
         if "WJets" in string:
               d["WJets" + en_var] = np.append(WJets, out[dataset]["mass"], axis=0)
               d["WJets_w" + en_var] = np.append(WJets_w, out[dataset]["mass_w"], axis=0)
               d["WJets_SS" + en_var] = np.append(WJets_SS, out[dataset]["ss_mass"], axis=0)
               d["WJets_SS_w" + en_var] = np.append(WJets_SS_w, out[dataset]["ss_mass_w"], axis=0)
         matches = ["WZ", "VV2l2nu", "ZZ4l", "ZZ2l2q"]
         if any([x in fname for x in matches]):
               d["Diboson" + en_var] = np.append(Diboson, out[dataset]["mass"], axis=0)
               d["Diboson_w" + en_var] = np.append(Diboson_w, out[dataset]["mass_w"], axis=0)
         matches = ["gg", "qqH125", "toptop", "WMinus", "WPlus", "ZH125", "TauTau"] 
         if any([x in fname for x in matches]):
               d["SMHiggs" + en_var] = np.append(SMHiggs, out[dataset]["mass"], axis=0)
               d["SMHiggs_w" + en_var]= np.append(SMHiggs_w, out[dataset]["mass_w"], axis=0) 
         matches = ["Tbar", "T-tchan", "tW"]
         if any([x in fname for x in matches]):
               d["SingleTop" + en_var] = np.append(SingleTop, out[dataset]["mass"], axis=0)
               d["SingleTop_w" + en_var] = np.append(SingleTop_w, out[dataset]["mass_w"], axis=0) 
         if "TTTo" in string:
               d["Top" + en_var] = np.append(Top, out[dataset]["mass"], axis=0)
               d["Top_w" + en_var] = np.append(Top_w, out[dataset]["mass_w"], axis=0) 
               d["Top_SS" + en_var] = np.append(Top_SS, out[dataset]["ss_mass"], axis=0)
               d["Top_SS_w" + en_var] = np.append(Top_SS_w, out[dataset]["ss_mass_w"], axis=0)
         if "SingleMuon" in string:
               d["Data" + en_var] = np.append(Data, out[dataset]["mass"], axis=0)
               d["Data_w" + en_var] = np.append(Data_w, out[dataset]["mass_w"], axis=0)
               d["Data_SS" + en_var] = np.append(Data_SS, out[dataset]["ss_mass"], axis=0)

   QCDScaleFactor = 1.6996559936491136


   #Turn into histograms
   labels1 = ["Data", "DY", "WJets", "Top", "SingleTop", "Diboson", "SMHiggs"]
   for i in labels1:
      d[i + "_h_" + en_var], d[i + "_bins_" + en_var] = np.histogram(d[i + "_" + en_var], bins=bins, weights=d[i + "_w_" + en_var])  
   
   Data_h, Data_bins = np.histogram(d["Data" + en_var], bins=bins)     
   DY_h, DY_bins = np.histogram(DY, bins=bins, weights=DY_w)
   WJets_h, WJets_bins = np.histogram(WJets, bins=bins, weights=WJets_w) 
   Top_h, Top_bins = np.histogram(Top, bins=bins, weights= Top_w) 
   SingleTop_h, SingleTop_bins = np.histogram(SingleTop, bins=bins, weights=SingleTop_w) 
   Diboson_h, Diboson_bins = np.histogram(Diboson, bins=bins, weights=Diboson_w)
   SMHiggs_h, SMHiggs_bins = np.histogram(SMHiggs, bins=bins, weights=SMHiggs_w)   


   #Data-driven QCD Estimation
   Data_SS_h, Data_SS_bins = np.histogram(Data_SS, bins=bins)
   DY_SS_h, DY_SS_bins = np.histogram(DY_SS, bins=bins, weights=DY_SS_w)
   WJets_SS_h, WJets_SS_bins = np.histogram(WJets_SS, bins=bins, weights=WJets_SS_w)
   Top_SS_h, Top_SS_bins = np.histogram(Top_SS, bins=bins, weights=Top_SS_w)
   QCD_h = np.subtract(np.subtract(np.subtract(Data_SS_h, DY_SS_h, dtype=object, out=None), WJets_SS_h, dtype=object, out=None), Top_SS_h, dtype=object, out=None)
   print("QCD: ", QCD_h)
   for i in range(QCD_h.size):
      if QCD_h[i] < 0.0:
         QCD_h[i] = 0.0
   QCD_w = np.full(shape=QCD_h.shape, fill_value=QCDScaleFactor, dtype=np.double)
   QCD_hist = (QCD_h, Data_SS_bins)


   #Plot and labels
   mass =   [SMHiggs_h,   Diboson_h,   SingleTop_h,  Top_h, WJets_h, DY_h, QCD_h]
   labels = ["SMHiggs", "Diboson", "SingleTop", "Top", "WJets", "DY", "QCD"]
   ss_mass = [Top_SS_h, WJets_SS_h, DY_SS_h]
   ss_mass_labels=["Top", "WJets", "DY"]

   #Plot OS boostedTau Visible Mass
   hep.histplot(mass, label=labels, histtype=("fill"), bins=bins, stack=True)
   hep.histplot(Data_h, label="Data", histtype=("errorbar"), bins=bins, color='k')
   plt.legend(loc = 'upper right', ncols = 2, fontsize = 8)
   plt.title("Boosted Tau + Muon Visible Mass", fontsize= 'small')
   ax.set_xlabel("Mass (GeV)")
   fig.savefig("./mutau_plots/SingleMuon_VISIBLE_MASS.png")
   plt.clf()

   #Plot SS boostedTau Visible Mass
   hep.histplot(ss_mass, label=ss_mass_labels, histtype=("fill"), bins=bins, stack=True)
   hep.histplot(Data_SS_h, label="Data", histtype=("errorbar"), bins=bins, color='k')
   plt.legend(loc = 'upper right', ncols = 2, fontsize = 8)
   plt.title("Boosted Tau + Muon Visible Mass (SS Region)", fontsize= 'small')
   ax.set_xlabel("Mass (GeV)")
   fig.savefig("./mutau_plots/SS_SingleMuon_VISIBLE_MASS.png")
   plt.clf()

   corrBins1= np.linspace(.9, 1.1, 80)
   corrBins2= np.linspace(0, 2, 80)
   plt.hist([Mu50IsoCorr, Mu27IsoCorr], bins=corrBins1)
   plt.title("MuIsoCorr")
   fig.savefig("./mutau_plots/correction_plots/MuIsoCorr.png")
   plt.clf()


   plt.hist([Mu50IDCorr, Mu27IDCorr], bins=corrBins1)
   plt.title("MuIDCorr")
   fig.savefig("./mutau_plots/correction_plots/MuIDCorr.png")
   plt.clf()


   plt.hist(Mu50TrgCorr, bins=corrBins1)
   plt.title("Mu50TrgCorr")
   fig.savefig("./mutau_plots/correction_plots/Mu50TrgCorr.png")
   plt.clf()

   plt.hist([PUCorrection50, PUCorrection27], bins=corrBins2)
   plt.title("PUCorrection")
   fig.savefig("./mutau_plots/correction_plots/PUCorrection.png")
   plt.clf()

   plt.hist(Mu27TrgCorr, bins=corrBins1)
   plt.title("Mu27TrgCorr")
   fig.savefig("./mutau_plots/correction_plots/Mu27TrgCorr.png")
   plt.clf()

   plt.hist([pTCorrection50, pTCorrection27], bins=corrBins2)
   plt.title("pTCorrection")
   fig.savefig("./mutau_plots/correction_plots/pTCorrection.png")
   plt.clf()

   #Send files out for Correction factor finding
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


   #Plot Visible mass again this time matching Abdollah's specifications and QCD Estimation
   #Data-driven QCD Estimation
   bins = np.append(np.linspace(0, 125, 25, endpoint=False), 125.)
   Data_SS_h, Data_SS_bins = np.histogram(Data_SS, bins=bins)
   print("Bins: ", Data_SS_bins)
   DY_SS_h, DY_SS_bins = np.histogram(DY_SS, bins=bins, weights=DY_SS_w)
   WJets_SS_h, WJets_SS_bins = np.histogram(WJets_SS, bins=bins, weights=WJets_SS_w)
   Top_SS_h, Top_SS_bins = np.histogram(Top_SS, bins=bins, weights=Top_SS_w)
   QCD_h = np.multiply(np.subtract(np.subtract(np.subtract(Data_SS_h, DY_SS_h, dtype=object, out=None), WJets_SS_h, dtype=object, out=None), Top_SS_h, dtype=object, out=None), QCDScaleFactor)
   print("QCD: ", QCD_h)
   for i in range(QCD_h.size):
      if QCD_h[i] < 0.0:
         QCD_h[i] = 0.0
   QCD_w = np.full(shape=QCD_h.shape, fill_value=QCDScaleFactor, dtype=np.double)
   QCD_hist = (QCD_h, Data_SS_bins)

   fig = plt.figure(figsize=(10, 8))
   ax = hist.axis.Regular(25, 0, 125, name=r"$m_{\mu \tau}$", flow=False)
   cax = hist.axis.StrCategory(["VV", "TT", "WJets", "QCD", "DY"], name="c")
   full_Hist = Hist(ax, cax)
   full_Hist.fill(DY, weight=DY_w, c="DY")
   full_Hist.fill(Top, weight=Top_w, c="TT")
   full_Hist.fill(WJets, weight=WJets_w, c="WJets")
   full_Hist.fill(np.append(Diboson, SingleTop), weight=np.append(Diboson_w, SingleTop_w), c="VV")
   full_Hist[:, hist.loc("QCD")] = QCD_h
   s = full_Hist.stack("c")
   s.plot(stack=True, histtype="fill")
   hist_2 = hist.Hist(hist.axis.Regular(25, 0, 125, name=r"$m_{\mu \tau}$", label=r"$m_{\mu \tau}$", flow=False))
   hist_2.fill(Data)
   hist_2.plot(histtype="errorbar", color='black')
   plt.legend(loc = 'upper right', ncols = 2, fontsize = 8)
   fig.savefig("./mutau_plots/MuTau_VISIBLE_MASS.png")
   plt.clf() 




   stack = (full_Hist[:, hist.loc("DY")].view() + 
            full_Hist[:, hist.loc("QCD")].view() +
            full_Hist[:, hist.loc("WJets")].view() + 
            full_Hist[:, hist.loc("TT")].view() + 
            full_Hist[:, hist.loc("VV")].view()
   )

   hist_1 = Hist(ax)
   hist_1[:] = stack

   main_ax_artists, sublot_ax_arists = hist_2.plot_ratio(
    hist_1,
    rp_ylabel=r"Data/Bkg",
    rp_num_label="Data",
    rp_denom_label="MC",
    rp_uncert_draw_type="line", 
   )
   fig.savefig("./mutau_plots/MuTau_Ratio_VISIBLE_MASS.png")
   plt.clf()