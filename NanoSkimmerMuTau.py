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

   def process(self, events, name, num_events, PUWeight, syst):
      dataset = events.metadata['dataset']
      #Call to find weight factor / XS
      XSection = self.weightCalc(name)
      #print(name, " ", XSection)
      #print(events.fields)


      electron = events.Electron
      print("ELECTRON:")
      print(electron.fields)
      #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! electron.scEta? does scEtOverPt work?
      ele_cut = (electron.pt >= 15) & (np.abs(electron.eta) <= 2.5)
      lowMVAele = electron[(np.abs(electron.scEtOverPt) <= 0.8) & (electron.mvaIso_WPL > -0.83) & ele_cut]
      midMVAele = electron[(np.abs(electron.scEtOverPt) > 0.8) & (np.abs(electron.scEtOverPt) <= 1.5) & (electron.mvaIso_WPL > -0.77) & ele_cut] 
      highMVAele = electron[(np.abs(electron.scEtOverPt) >= 1.5) & (electron.mvaIso_WPL > -0.69) & ele_cut] 
      events = events[(ak.num(lowMVAele) == 0) & (ak.num(midMVAele) == 0) & (ak.num(highMVAele) == 0)]
      #Extra muon veto
      muon = events.Muon
      print("MUON:")
      print(muon.fields)
      muon = muon[ak.num(muon) > 1]
      #!!!!!!!!!!!!!!!!!!!!!!!!!! JUST USE PFRELISO04? TRIGGERIDLOOSE?
      #RelIsoMu = (muon.muPFChIso + muon.muPFNeuIso + muon.muPFPhoIso - (0.5 * muon.muPFPUIso)) / muon.pt
      badMuon = muon[(np.abs(muon.pfRelIso04_all) > 0.3) & (muon[:,1].pt > 10) & (np.bitwise_and(muon.triggerIdLoose, self.bit_mask(2)) == self.bit_mask(2))]
      events = events[(ak.num(muon) > 0) & (ak.num(badMuon) == 0)]

      
      print("JET")
      print(events.Jet.fields)
      #!!!!!!!!!!!!!!!!!!!!!! SYSTEMATIC VARIATIONS? OR WHAT WOULD YOU LIKE? CURRENTLY DOING TAU ENERGY SCALE, JET ENERGY SCALE, MET, UNCLUSTERED ENERGIES FOR ABDOLLAH
      #if syst == "_JESUp": events['_jetPt'] = events.jetPtTotUncUp,
      #elif syst == "_JESDown": events['_jetPt']= events.jetPtTotUncDown,
      #else:  events['_jetPt'] =  events.jetPt,
      #print(events.jetPt)
      #print(events.jetPtTotUncUp)
      #print(events._jetPt)
      #print(events['_jetPt'][0])
      #HT Cut
      # jets = ak.zip( 
		# {
		# 		"pt": events['_jetPt'][0],
		# 		"eta": events.jetEta,
		# 		"phi": events.jetPhi,
      #       "energy": events.jetEn,
      #       "jetPFLooseId": events.jetPFLooseId,
      #       "jetCSV2BJetTags": events.jetCSV2BJetTags,
		# },
		# 	   with_name="PtEtaPhiELorentzVector",
		#       behavior=vector.behavior,
		# )
      # if syst == "_JESUp" or syst == "_JESDown":
      #    events["Met"]  = events.pfMetNoRecoil; events["Metphi"]=events.pfMetPhiNoRecoil,
      #    Oldjets = ak.zip( 
		#    {
		# 		"pt": events.jetPt,
		# 		"eta": events.jetEta,
		# 		"phi": events.jetPhi,
      #       "energy": events.jetEn,
      #       "jetPFLooseId": events.jetPFLooseId,
      #       "jetCSV2BJetTags": events.jetCSV2BJetTags,
		#    },
		# 	   with_name="PtEtaPhiELorentzVector",
		#       behavior=vector.behavior,
		#    )
      #    MET_x = np.subtract(np.multiply(events["Met"], np.cos(events["Metphi"][0])), (ak.sum(Oldjets.px, axis=1) - ak.sum(jets.px, axis=1)))
      #    MET_y = np.subtract(np.multiply(events["Met"],np.sin(events["Metphi"][0])), (ak.sum(Oldjets.py, axis=1) - ak.sum(jets.py, axis=1)))
      #    events["Met"] = np.sqrt(np.power(MET_x, 2) + np.power(MET_y, 2))
      #    events["Metphi"] = [np.arctan2(MET_y, MET_x)]
      jets = events.Jet
      goodJets= jets[(jets.jetId > 0.5) & (jets.pt > 30) & (np.abs(jets.eta) < 3.0)]
      HT = ak.sum(goodJets.pt, axis=-1)

      #BJet Veto 
      #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! BTAGSCSVV2 OR DIFFERENT FOR BJET CUT?
      bJets = jets[(jets.btagCSVV2 > .7527) & (jets.jetId > .5) & (jets.pt > 30) & (np.abs(jets.eta) < 2.4)]
      events = events[(ak.num(jets) > 0) & (HT > 200) & (ak.num(bJets) == 0)]


      # if syst != "_JESUp" or syst != "_JESDown":
      #    if syst == "_MissingEn_JESUp": events["Met"] = events.pfMET_T1JESUp; events["Metphi"]=events.pfMETPhi_T1JESUp,
      #    elif syst == "_MissingEn_JESDown": events["Met"]  = events.pfMET_T1JESDo;  events["Metphi"]=events.pfMETPhi_T1JESDo,
      #    elif syst == "_MissingEn_UESUp": events["Met"]  = events.pfMET_T1UESUp;  events["Metphi"]=events.pfMETPhi_T1UESUp,
      #    elif syst == "_MissingEn_UESDown": events["Met"]  = events.pfMET_T1UESDo;  events["Metphi"]=events.pfMETPhi_T1UESDo,
      #    else: events["Met"]  = events.pfMetNoRecoil; events["Metphi"]=events.pfMetPhiNoRecoil,

      # if syst == "_en_scale_up": en_var = "_en_scale_up"
      # elif syst == "_en_scale_down": en_var = "_en_scale_down"
      # else: en_var = "_nominal"

      mtEvents = events[ak.all((ak.num(events.Muon) > 0) & (ak.num(events.boostedTau) > 0), axis = -1)]
      muonC = events.Muon
      tauC = events.boostedTau

      print("TAU")
      print(tauC.fields)

      muon = ak.zip( 
			{
			"pt": muonC.pt,
         "eta": muonC.eta,
			"phi": muonC.phi,
			"mass": muonC.mass,
		   },
		   with_name="PtEtaPhiMLorentzVector",
         behavior=vector.behavior,
		)
      tau = ak.zip( 
			{
			"pt": tauC.pt,
         "eta": tauC.eta,
         "phi": tauC.phi,
			"mass": tauC.mass,
		   },
		   with_name="PtEtaPhiMLorentzVector",
		   behavior=vector.behavior,
		)

      # tauOld = tau
      # if XSection != 1:
      #    #Gen-tau matching for systematics - boosted Tau Energy
      #    genTauExist = ak.any(events.mcPID == 15, axis=-1)
      #    genTauIndex = events.mcPID == 15
      #    allEta= ak.cartesian({'tau': events.boostedTauEta, 'mc': events.mcEta[genTauIndex]}, axis=1, nested=True)
      #    allPhi= ak.cartesian({'tau': events.boostedTauPhi, 'mc': events.mcPhi[genTauIndex]}, axis=1, nested=True)
      #    deltaR = np.sqrt((np.subtract(allEta['tau'], allEta['mc']))**2 + (np.subtract(allPhi['tau'], allPhi['mc']))**2)
      #    genTauCut = ak.min(deltaR, axis=-1) < .1
      #    newtau = ak.where(genTauExist & genTauCut, tau.multiply(events[f"{en_var}"]), tau)
      #    tau = ak.where(genTauExist, newtau, tau)
      
      # if (syst == "_en_scale_up") or (syst == "_en_scale_down"): #Do for jets
      #    MET_x = np.subtract(np.multiply(events["Met"], np.cos(events["Metphi"][0])), (ak.sum(tauOld.px, axis=1) - ak.sum(tau.px, axis=1)))
      #    MET_y = np.subtract(np.multiply(events["Met"],np.sin(events["Metphi"][0])), (ak.sum(tauOld.py, axis=1) - ak.sum(tau.py, axis=1)))
      #    events["Met"] = np.sqrt(np.power(MET_x, 2) + np.power(MET_y, 2))
      #    events["Metphi"] = [np.arctan2(MET_y, MET_x)]

      
      #Split into pairs
      pairs = ak.cartesian({'tau': tau, 'muon': muon}, nested=False)
      pairsC = ak.cartesian({'tau': tauC, 'muon': muonC}, nested=False) 
      #Trigger Cut
      trigger_mask_Mu27 = self.bit_mask(19)
      trigger_mask_Mu50 = self.bit_mask(21)

      #Delta r cut
      dr = pairs['tau'].delta_r(pairs['muon'])
      #ID Cut
      MuID = ((np.bitwise_and(pairsC['muon'].triggerIdLoose, self.bit_mask(1)) == self.bit_mask(1)) & (np.abs(pairsC['muon'].dz) < 0.2)) # & (np.abs(pairsC['muon'].d0) < 0.045))
      #Apply everything at once
      muTau_mask27 = ((np.abs(pairs['muon'].eta) < 2.4)
                  & (MuID)
                  & (pairs['muon'].pt >= 27)
                  & (pairs['muon'].pt < 52)
                  & (pairsC['muon'].pfRelIso04_all < .3)
                  & (mtEvents.MET.pt > 30)
                  & (pairs['tau'].pt > 30)
                  & (np.bitwise_and(pairsC['muon'].triggerIdLoose, trigger_mask_Mu27) == trigger_mask_Mu27)
                  & (np.absolute(pairs['tau'].eta) < 2.3)
                  & (pairsC['tau'].idAntiMu >= 0.5)
                  & (pairsC['tau'].rawIsodR03 >= 0.5)
                  & (dr > .1) 
                  & (dr < .8))
      muTau_mask50 = ((np.abs(pairs['muon'].eta) < 2.4)
                  & (MuID)
                  & (pairs['muon'].pt >= 52)
                  & ((np.bitwise_and(pairsC['muon'].triggerIdLoose, trigger_mask_Mu50) == trigger_mask_Mu50))
                  & (pairs['tau'].pt > 30)
                  & (np.absolute(pairs['tau'].eta) < 2.3)
                  & (pairsC['tau'].idAntiMu >= 0.5)
                  & (pairsC['tau'].rawIsodR03 >= 0.5)
                  & (dr > .1) 
                  & (dr < .8))

      #If everything cut return 0 (to avoid pointer errors)
      if not ak.any(muTau_mask27) and not ak.any(muTau_mask50):
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
      mtEvents = mtEvents[(muTau_mask27) | (muTau_mask50)]
      print(ak.count(pairs['muon'].pt))
      #Separate based on charge
      OS_pairs = pairs[pairsC['tau'].charge + pairsC['muon'].charge == 0]
      SS_pairs = pairs[pairsC['tau'].charge + pairsC['muon'].charge != 0]
      OS_mtEvents = mtEvents[pairsC['tau'].charge + pairsC['muon'].charge == 0]
      SS_mtEvents = mtEvents[pairsC['tau'].charge + pairsC['muon'].charge != 0]
      #Separate based on trigger again
      OS_50 = OS_pairs[OS_pairs['muon'].pt >= 52]
      OS_27 = OS_pairs[OS_pairs['muon'].pt < 52]
      SS_50 = SS_pairs[SS_pairs['muon'].pt >= 52]
      SS_27 = SS_pairs[SS_pairs['muon'].pt < 52]

      OS_50C = OS_mtEvents[OS_pairs['muon'].pt >= 52]
      OS_27C = OS_mtEvents[OS_pairs['muon'].pt < 52]
      SS_50C = SS_mtEvents[SS_pairs['muon'].pt >= 52]
      SS_27C = SS_mtEvents[SS_pairs['muon'].pt < 52]

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
      tmass50 = np.sqrt(np.square(OS_50['muon'].pt + OS_50C.MET.pt) - np.square(OS_50['muon'].px + OS_50C.MET.pt * np.cos(OS_50.MET.phi)) - np.square(OS_50['muon'].py + OS_50.MET.pt * np.sin(OS_50.MET.phi)))
      tmass27 = np.sqrt(np.square(OS_27['muon'].pt + OS_27.MET.pt) - np.square(OS_27['muon'].px + OS_27.MET.pt * np.cos(OS_27.MET.phi)) - np.square(OS_27['muon'].py + OS_27.MET.pt * np.sin(OS_27.MET.phi)))
      OS_50 = OS_50[tmass50 < 80]
      OS_27 = OS_27[tmass27 < 80]
      OS_50C = OS_50C[tmass50 < 80]
      OS_27C = OS_27C[tmass27 < 80]

      tmass50_SS = np.sqrt(np.square(SS_50['muon'].pt + SS_50C.MET.pt) - np.square(SS_50['muon'].px + SS_50C.MET.pt * np.cos(SS_50.MET.phi)) - np.square(SS_50['muon'].py + SS_50.MET.pt * np.sin(SS_50.MET.phi)))
      tmass27_SS = np.sqrt(np.square(SS_27['muon'].pt + SS_27.MET.pt) - np.square(SS_27['muon'].px + SS_27.MET.pt * np.cos(SS_27.MET.phi)) - np.square(SS_27['muon'].py + SS_27.MET.pt * np.sin(SS_27.MET.phi)))

      SS_50 = SS_50[tmass50_SS < 80]
      SS_27 = SS_27[tmass27_SS < 80]
      SS_50C = SS_50C[tmass50_SS < 80]
      SS_27C = SS_27C[tmass27_SS < 80]

      #MET Vector
      MetVec27 =  ak.zip(
      {
         "pt": OS_27C.MET.pt,
         "eta": 0,
         "phi": OS_27C.MET.phi,
         "mass": 0,
         },
      with_name="PtEtaPhiMLorentzVector",
      behavior=vector.behavior,
      ) 
      MetVec50 =  ak.zip(
      {
         "pt": OS_50C.MET.pt,
         "eta": 0,
         "phi": OS_50C.MET.phi,
         "mass": 0,
         },
      with_name="PtEtaPhiMLorentzVector",
      behavior=vector.behavior,
      )   
      MetVec27_SS =  ak.zip(
      {
         "pt": SS_27C.MET.pt,
         "eta": 0,
         "phi": SS_27C.MET.phi,
         "mass": 0,
         },
      with_name="PtEtaPhiMLorentzVector",
      behavior=vector.behavior,
      )  
      MetVec50_SS =  ak.zip(
      {
         "pt": SS_50C.MET.pt,
         "eta": 0,
         "phi": SS_50C.MET.phi,
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



         #!!!!!!!!!!!!!!!!!!!!!!! WHERE WOULD PUTRUE BE FOR PILEUP CORRECTION?
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
   def run_cuts(self, fileList, directory, dataset, PUWeight, corrArrays, en_var):
      for sample in fileList:

         if ("SingleMuon" in sample) and (en_var != "_nominal"): continue

         fname = os.path.join(directory, sample)
         file = uproot.open(fname)
         events = NanoEventsFactory.from_root(
            file,
            treepath="/Events",
            schemaclass=NanoAODSchema,
            metadata={"dataset": dataset},
         ).events()
         string = str(sample)
         #num_events = file['hEvents'].member('fEntries') / 2 # DO WE STILL NEED TO DO THIS
         num_events = 28000
         events["_en_scale_down"] = 0.97
         events["_nominal"] = 1.00
         events["_en_scale_up"] = 1.03

         print(events.fields)

         out = p.process(events, fname, num_events, PUWeight, en_var)

         for i in corrArrays:
            d[i + "" + en_var] = np.append(d[i + "" + en_var], out[dataset][f"{i}"], axis=0)

         #Sort files in their respective arrays
         if "DY" in string:
               d["DY" + en_var] = np.append(d["DY" + en_var], out[dataset]["mass"], axis=0)
               d["DY_w" + en_var] = np.append(d["DY_w" + en_var], out[dataset]["mass_w"], axis=0)
               d["DY_SS" + en_var] = np.append(d["DY_SS" + en_var], out[dataset]["ss_mass"], axis=0)
               d["DY_SS_w" + en_var] = np.append(d["DY_SS_w" + en_var], out[dataset]["ss_mass_w"], axis=0)
         if "WJets" in string:
               d["WJets" + en_var] = np.append(d["WJets" + en_var], out[dataset]["mass"], axis=0)
               d["WJets_w" + en_var] = np.append(d["WJets_w" + en_var], out[dataset]["mass_w"], axis=0)
               d["WJets_SS" + en_var] = np.append(d["WJets_SS" + en_var], out[dataset]["ss_mass"], axis=0)
               d["WJets_SS_w" + en_var] = np.append(d["WJets_SS_w" + en_var], out[dataset]["ss_mass_w"], axis=0)
         matches = ["WZ", "VV2l2nu", "ZZ4l", "ZZ2l2q"]
         if any([x in fname for x in matches]):
               d["Diboson" + en_var] = np.append(d["Diboson" + en_var], out[dataset]["mass"], axis=0)
               d["Diboson_w" + en_var] = np.append(d["Diboson_w" + en_var], out[dataset]["mass_w"], axis=0)
         matches = ["gg", "qqH125", "toptop", "WMinus", "WPlus", "ZH125", "TauTau"] 
         if any([x in fname for x in matches]):
               d["SMHiggs" + en_var] = np.append(d["SMHiggs" + en_var], out[dataset]["mass"], axis=0)
               d["SMHiggs_w" + en_var]= np.append(d["SMHiggs_w" + en_var], out[dataset]["mass_w"], axis=0) 
         matches = ["Tbar", "T-tchan", "tW"]
         if any([x in fname for x in matches]):
               d["SingleTop" + en_var] = np.append(d["SingleTop" + en_var], out[dataset]["mass"], axis=0)
               d["SingleTop_w" + en_var] = np.append(d["SingleTop_w" + en_var], out[dataset]["mass_w"], axis=0) 
         if "TTTo" in string:
               d["Top" + en_var] = np.append(d["Top" + en_var], out[dataset]["mass"], axis=0)
               d["Top_w" + en_var] = np.append(d["Top_w" + en_var], out[dataset]["mass_w"], axis=0) 
               d["Top_SS" + en_var] = np.append(d["Top_SS" + en_var], out[dataset]["ss_mass"], axis=0)
               d["Top_SS_w" + en_var] = np.append(d["Top_SS_w" + en_var], out[dataset]["ss_mass_w"], axis=0)
         if ("SingleMuon" in string) and (en_var == "_nominal"):
               d["Data"] = np.append(d["Data"], out[dataset]["mass"], axis=0)
               d["Data_w"] = np.append(d["Data_w"], out[dataset]["mass_w"], axis=0)
               d["Data_SS"] = np.append(d["Data_SS"], out[dataset]["ss_mass"], axis=0)
               d["Data_SS_w"] = np.append(d["Data_SS_w"], out[dataset]["ss_mass_w"], axis=0)
      return




   def postprocess(self, accumulator):
      pass
   

if __name__ == "__main__":
   #directory = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedH3/2018/mt/v2_fast_Hadd"
   directory = "."
   dataset = "SingleMuon"
   fig, ax = plt.subplots()
   #hep.style.use(hep.style.ROOT)
   plt.style.use('seaborn-v0_8-colorblind')
   p = MyProcessor()
   fileList = ['NANO_NANO_76.root']
   # fileList = ["DYJetsToLL_Pt-100To250.root",    "Tbar-tW.root",
   #    'DYJetsToLL_Pt-250To400.root',            'toptopH125.root',
   #    'DYJetsToLL_Pt-400To650.root',            'T-tchan.root',
   #    'DYJetsToLL_Pt-50To100.root',             'TTTo2L2Nu.root',
   #    'DYJetsToLL_Pt-650ToInf.root',            'TTToHadronic.root',
   #    'TTToSemiLeptonic.root',                  'Tbar-tchan.root',
   #    'ggH125.root',                            'T-tW.root',
   #    'ggZHLL125.root',                         'VV2l2nu.root',
   #    'ggZHNuNu125.root',                       'WJetsToLNu_HT-100To200.root',
   #    'ggZHQQ125.root',                         'WJetsToLNu_HT-1200To2500.root',
   #    'JJH0PMToTauTauPlusOneJets.root',         'WJetsToLNu_HT-200To400.root',
   #    'JJH0PMToTauTauPlusTwoJets.root',         'WJetsToLNu_HT-2500ToInf.root',
   #    'JJH0PMToTauTauPlusZeroJets.root',        'WJetsToLNu_HT-400To600.root',
   #    'qqH125.root',                            'WJetsToLNu_HT-600To800.root',
   #    'SingleMuon_Run2018A-17Sep2018-v2.root',  'WJetsToLNu_HT-800To1200.root',
   #    'SingleMuon_Run2018B-17Sep2018-v1.root',  'WMinusH125.root',
   #    'SingleMuon_Run2018C-17Sep2018-v1.root',  'WPlusH125.root',
   #    'SingleMuon_Run2018D-22Jan2019-v2.root',  'WZ1l1nu2q.root',
   #    'WZ1l3nu.root',                           'WZ2l2q.root',
   #    'WZ3l1nu.root',                           'ZH125.root',
   #    'ZZ2l2q.root',                            'ZZ4l.root']

   #Create empty arrays for data and all variations
   en_variations = ["_nominal", "_en_scale_up", "_en_scale_down"]
   jet_variations = ["_JESUp", "_JESDown"]
   met_variations = ["_nominal", "_MissingEn_JESUp", "_MissingEn_JESDown", "_MissingEn_UESUp", "_MissingEn_UESDown"]




   allSysts = ["_nominal", "_JESUp", "_JESDown"] # "_en_scale_up", "_en_scale_down", "_MissingEn_JESUp", "_MissingEn_JESDown", "_MissingEn_UESUp", "_MissingEn_UESDown"]
   d = {}

   emptyArrays = ["DY", "WJets", "Diboson", "SMHiggs", "SingleTop", "Top"]
   corrArrays= ["Mu50IsoCorr", "Mu50IDCorr", "Mu50TrgCorr", "PUCorrection50", "pTCorrection50", "Mu27IsoCorr", "Mu27IDCorr", "Mu27TrgCorr", "PUCorrection27", "pTCorrection27"]
   for var in allSysts:
      for i in emptyArrays:
         d[i + "" + var] = []
         d[i + "_w" + var] = []
         if i in {"DY", "WJets", "Top", "Data"}:
            d[i + "_SS" + var] = []
            d[i + "_SS_w" + var] = []
      for i in corrArrays:
         d[i + "" + var] = []
   d["Data"], d["Data_SS"], d["Data_w"], d["Data_SS_w"] = [], [], [], []

   bins=np.linspace(0, 150, 30)

   #Get Pileup Weight
   with uproot.open("pu_distributions_mc_2018.root") as f1:
      with uproot.open("pu_distributions_data_2018.root") as f2:
         mc = f1["pileup"].values()
         data = f2["pileup"].values()
         HistoPUMC = np.divide(mc, ak.sum(mc))
         HistoPUData = np.divide(data, ak.sum(data))
         PUWeight = np.divide(HistoPUData, HistoPUMC)

   #for en_var in en_variations:
   #   print("Variation: ", en_var)
   #   p.run_cuts(fileList, directory, dataset, PUWeight, corrArrays, en_var)

   #for met_var in met_variations:
   #   print("Variation: ", met_var)
    #  p.run_cuts(fileList, directory, dataset, PUWeight, corrArrays, met_var)

   for jet_var in jet_variations:
      print("Variation: ", jet_var)
      p.run_cuts(fileList, directory, dataset, PUWeight, corrArrays, jet_var)

   QCDScaleFactor = 1.6996559936491136


   for en_var in allSysts:
      #Turn into histograms
      if en_var == "_nominal":
         d["Data_h"], d["Data_bins"] = np.histogram(d["Data"], bins=bins)
         d["Data_SS_h"], d["Data_SS_bins"] = np.histogram(d["Data_SS"], bins=bins)
      labels1 = ["DY", "WJets", "Top", "SingleTop", "Diboson", "SMHiggs", "DY_SS", "WJets_SS", "Top_SS"]
      for i in labels1:
         d[i + "_h" + en_var], d[i + "_bins" + en_var] = np.histogram(d[i + "" + en_var], bins=bins, weights= d[i + "_w" + en_var])  
      
   for en_var in allSysts:
      d["QCD_h" + en_var] = np.subtract(np.subtract(np.subtract(d["Data_SS_h"], d["DY_SS_h" + en_var], dtype=object, out=None), d["WJets_SS_h" + en_var], dtype=object, out=None), d["Top_SS_h" + en_var], dtype=object, out=None)
      for i in range(d["QCD_h" + en_var].size):
         if d["QCD_h" + en_var][i] < 0.0:
            d["QCD_h" + en_var][i] = 0.0
      d["QCD_w" + en_var] = np.full(shape=d["QCD_h" + en_var].shape, fill_value=QCDScaleFactor, dtype=np.double)
      d["QCD_hist" + en_var] = (d["QCD_h" + en_var], d["Data_SS_bins"])


   # #Plot and labels
   # mass =   [SMHiggs_h,   Diboson_h,   SingleTop_h,  Top_h, WJets_h, DY_h, QCD_h]
   # labels = ["SMHiggs", "Diboson", "SingleTop", "Top", "WJets", "DY", "QCD"]
   # ss_mass = [Top_SS_h, WJets_SS_h, DY_SS_h]
   # ss_mass_labels=["Top", "WJets", "DY"]

   # #Plot OS boostedTau Visible Mass
   # hep.histplot(mass, label=labels, histtype=("fill"), bins=bins, stack=True)
   # hep.histplot(Data_h, label="Data", histtype=("errorbar"), bins=bins, color='k')
   # plt.legend(loc = 'upper right', ncols = 2, fontsize = 8)
   # plt.title("Boosted Tau + Muon Visible Mass", fontsize= 'small')
   # ax.set_xlabel("Mass (GeV)")
   # fig.savefig("./mutau_plots/SingleMuon_VISIBLE_MASS.png")
   # plt.clf()

   # #Plot SS boostedTau Visible Mass
   # hep.histplot(ss_mass, label=ss_mass_labels, histtype=("fill"), bins=bins, stack=True)
   # hep.histplot(Data_SS_h, label="Data", histtype=("errorbar"), bins=bins, color='k')
   # plt.legend(loc = 'upper right', ncols = 2, fontsize = 8)
   # plt.title("Boosted Tau + Muon Visible Mass (SS Region)", fontsize= 'small')
   # ax.set_xlabel("Mass (GeV)")
   # fig.savefig("./mutau_plots/SS_SingleMuon_VISIBLE_MASS.png")
   # plt.clf()

   # corrBins1= np.linspace(.9, 1.1, 80)
   # corrBins2= np.linspace(0, 2, 80)
   # plt.hist([Mu50IsoCorr, Mu27IsoCorr], bins=corrBins1)
   # plt.title("MuIsoCorr")
   # fig.savefig("./mutau_plots/correction_plots/MuIsoCorr.png")
   # plt.clf()


   # plt.hist([Mu50IDCorr, Mu27IDCorr], bins=corrBins1)
   # plt.title("MuIDCorr")
   # fig.savefig("./mutau_plots/correction_plots/MuIDCorr.png")
   # plt.clf()


   # plt.hist(Mu50TrgCorr, bins=corrBins1)
   # plt.title("Mu50TrgCorr")
   # fig.savefig("./mutau_plots/correction_plots/Mu50TrgCorr.png")
   # plt.clf()

   # plt.hist([PUCorrection50, PUCorrection27], bins=corrBins2)
   # plt.title("PUCorrection")
   # fig.savefig("./mutau_plots/correction_plots/PUCorrection.png")
   # plt.clf()

   # plt.hist(Mu27TrgCorr, bins=corrBins1)
   # plt.title("Mu27TrgCorr")
   # fig.savefig("./mutau_plots/correction_plots/Mu27TrgCorr.png")
   # plt.clf()

   # plt.hist([pTCorrection50, pTCorrection27], bins=corrBins2)
   # plt.title("pTCorrection")
   # fig.savefig("./mutau_plots/correction_plots/pTCorrection.png")
   # plt.clf()

   outFile = uproot.recreate("boostedHTT_mt_2018_jetEn.input.root")
   for en_var in allSysts:
      #Send files out for Correction factor finding
      if en_var == "_nominal":
         d["Data_h"] = np.histogram(d["Data"], bins=bins) 
         outFile["DYJets_met_1_13TeV/data_obs"] = d["Data_h"]
         
      labels2 = ["DY", "TT", "VV", "WJets", "QCD"]
      j = ""
      k = ""
      for i in labels2:
         if i == "QCD":
            outFile["DYJets_met_1_13TeV/" + i + "" + en_var] = d["QCD_hist" + en_var]
            continue
         
         if i == "TT": 
            j = "Top"
         else: j = i
         

         if i == "VV":
            d[i + "_h" + en_var] = np.histogram(np.append(d["Diboson" + en_var], d["SingleTop" + en_var]), bins=bins, weights=np.append(d["Diboson_w" + en_var], d["SingleTop_w" + en_var]) ) 
         else: d[i + "_h" + en_var] = np.histogram(d[j + "" + en_var], bins=bins, weights= d[j + "_w" + en_var]) 
         
         if i == "DY": k = "DYJets125"
         else: k = i

         outFile["DYJets_met_1_13TeV/" + k + "" + en_var] = d[i + "_h" + en_var]

   print("Past outFile!")


      # outFile["DY_Jets_mt_1_13TeV/DYJets125" + en_var] = DY_h
      # outFile["DY_Jets_mt_1_13TeV/TT" + en_var] = TT_h
      # outFile["DY_Jets_mt_1_13TeV/VV" + en_var] = VV_h
      # outFile["DY_Jets_mt_1_13TeV/W" + en_var] = WJets_h
      # outFile["DY_Jets_mt_1_13TeV/QCD" + en_var] = QCD_hist
      # outFile["DY_Jets_mt_1_13TeV/data_obs" + en_var] = Data_h
   

   """
   #Plot Visible mass again this time matching Abdollah's specifications and QCD Estimation
   #Data-driven QCD Estimation
   bins = np.append(np.linspace(0, 125, 25, endpoint=False), 125.)
   for en_var in en_variations:
      #Turn into histograms
      labels1 = ["Data", "DY", "WJets", "Top", "SingleTop", "Diboson", "SMHiggs", "Data_SS", "DY_SS", "WJets_SS", "Top_SS"]
      for i in labels1:
         d[i + "_h" + en_var], d[i + "_bins" + en_var] = np.histogram(d[i + "" + en_var], bins=bins, weights= d[i + "_w" + en_var])  
      
   for en_var in en_variations:
      d["QCD_h" + en_var] = np.subtract(np.subtract(np.subtract(d["Data_SS_h" + en_var], d["DY_SS_h" + en_var], dtype=object, out=None), d["WJets_SS_h" + en_var], dtype=object, out=None), d["Top_SS_h" + en_var], dtype=object, out=None)
      for i in range(d["QCD_h" + en_var].size):
         if d["QCD_h" + en_var][i] < 0.0:
            d["QCD_h" + en_var][i] = 0.0
      d["QCD_w" + en_var] = np.full(shape=d["QCD_h" + en_var].shape, fill_value=QCDScaleFactor, dtype=np.double)
      d["QCD_hist" + en_var] = (d["QCD_h" + en_var], d["Data_SS_bins" + en_var])




   # Data_SS_h, Data_SS_bins = np.histogram(Data_SS, bins=bins)
   # print("Bins: ", Data_SS_bins)
   # DY_SS_h, DY_SS_bins = np.histogram(DY_SS, bins=bins, weights=DY_SS_w)
   # WJets_SS_h, WJets_SS_bins = np.histogram(WJets_SS, bins=bins, weights=WJets_SS_w)
   # Top_SS_h, Top_SS_bins = np.histogram(Top_SS, bins=bins, weights=Top_SS_w)
   # QCD_h = np.multiply(np.subtract(np.subtract(np.subtract(Data_SS_h, DY_SS_h, dtype=object, out=None), WJets_SS_h, dtype=object, out=None), Top_SS_h, dtype=object, out=None), QCDScaleFactor)
   # print("QCD: ", QCD_h)
   # for i in range(QCD_h.size):
   #    if QCD_h[i] < 0.0:
   #       QCD_h[i] = 0.0
   # QCD_w = np.full(shape=QCD_h.shape, fill_value=QCDScaleFactor, dtype=np.double)
   # QCD_hist = (QCD_h, Data_SS_bins)

   fig = plt.figure(figsize=(10, 8))
   ax = hist.axis.Regular(25, 0, 125, name=r"$m_{\mu \tau}$", flow=False)
   cax = hist.axis.StrCategory(["VV", "TT", "WJets", "QCD", "DY"], name="c")
   vax = hist.axis.StrCategory(en_variations, name="v")

   full_Hist = Hist(ax, cax, vax)

   for en_var in en_variations:
      full_Hist.fill(d["DY" + en_var], weight=d["DY_w" + en_var], c="DY", v=f"{en_var}")
      full_Hist.fill(d["Top" + en_var], weight=d["Top_w" + en_var], c="TT", v=f"{en_var}")
      full_Hist.fill(d["WJets" + en_var], weight=d["WJets_w" + en_var], c="WJets", v=f"{en_var}")
      full_Hist.fill(np.append(d["Diboson" + en_var], d["SingleTop" + en_var]), weight=np.append(d["Diboson_w" + en_var], d["SingleTop_w" + en_var]), c="VV", v=f"{en_var}")
      full_Hist[:, hist.loc("QCD")] = d["QCD_h" + en_var]
      s = full_Hist.stack("c")
   s["_en_nominal"].plot(stack=True, histtype="fill")
   hist_2 = hist.Hist(hist.axis.Regular(25, 0, 125, name=r"$m_{\mu \tau}$", label=r"$m_{\mu \tau}$", flow=False))
   hist_2.fill(d["Data" + "_en_nominal"])
   hist_2.plot(histtype="errorbar", color='black')
   plt.legend(loc = 'upper right', ncols = 2, fontsize = 8)
   fig.savefig("./mutau_plots/MuTau_VISIBLE_MASS_NOMINALEN.png")
   plt.clf() 

   s["_en_scale_up"].plot(stack=True, histtype="fill")
   hist_2 = hist.Hist(hist.axis.Regular(25, 0, 125, name=r"$m_{\mu \tau}$", label=r"$m_{\mu \tau}$", flow=False))
   hist_2.fill(d["Data" + "_en_nominal"])
   hist_2.plot(histtype="errorbar", color='black')
   plt.legend(loc = 'upper right', ncols = 2, fontsize = 8)
   fig.savefig("./mutau_plots/MuTau_VISIBLE_MASS_SCALEUPEN.png")
   plt.clf() 

   s["_en_scale_up"].plot(stack=True, histtype="fill")
   hist_2 = hist.Hist(hist.axis.Regular(25, 0, 125, name=r"$m_{\mu \tau}$", label=r"$m_{\mu \tau}$", flow=False))
   hist_2.fill(d["Data" + "_en_nominal"])
   hist_2.plot(histtype="errorbar", color='black')
   plt.legend(loc = 'upper right', ncols = 2, fontsize = 8)
   fig.savefig("./mutau_plots/MuTau_VISIBLE_MASS_SCALEDOWNEN.png")
   plt.clf()  """
   # stack = (full_Hist[:, hist.loc("DY")].view() + 
   #          full_Hist[:, hist.loc("QCD")].view() +
   #          full_Hist[:, hist.loc("WJets")].view() + 
   #          full_Hist[:, hist.loc("TT")].view() + 
   #          full_Hist[:, hist.loc("VV")].view()
   # )

   # hist_1 = Hist(ax)
   # hist_1[:] = stack

   # main_ax_artists, sublot_ax_arists = hist_2.plot_ratio(
   #  hist_1,
   #  rp_ylabel=r"Data/Bkg",
   #  rp_num_label="Data",
   #  rp_denom_label="MC",
   #  rp_uncert_draw_type="line", 
   # )
   # fig.savefig("./mutau_plots/MuTau_Ratio_VISIBLE_MASS.png")
   # plt.clf()