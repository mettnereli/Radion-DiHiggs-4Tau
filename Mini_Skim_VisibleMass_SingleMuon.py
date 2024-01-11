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

   def process(self, events, name, num_events, PUWeight):
      dataset = events.metadata['dataset']
      #Call to find weight factor / XS
      XSection = self.weightCalc(name)
      print(name, " ", XSection)
      #print(events.fields)


      #At least on muon
      events = events[(ak.num(events.muPt) > 0)]

      #Bjet veto - may need to fix
      events = events[ak.all((events.jetDeepCSVTags_b > .7527) & (events.jetPFLooseId > .5) & (events.jetPt > 20) & (np.abs(events.jetEta) < 2.4), axis=-1) == False]

      #Extra electron veto
      eleEvents = events[ak.num(events.elePt) > 0]
      RelIsoEle = (eleEvents.elePFChIso + eleEvents.elePFNeuIso + eleEvents.elePFPhoIso - (0.5 * eleEvents.elePFPUIso)) / eleEvents.elePt
      MVAId_low = (np.abs(eleEvents.eleSCEta) <= 0.8) & (eleEvents.eleIDMVANoIso > 0.837)
      MVAId_mid = (np.abs(eleEvents.eleSCEta) > 0.8) & (np.abs(eleEvents.eleSCEta) <= 1.5) & (eleEvents.eleIDMVANoIso > 0.715) 
      MVAId_high = (np.abs(eleEvents.eleSCEta) >= 1.5) & (eleEvents.eleIDMVANoIso > 0.357) 
      MVAId = (MVAId_low) | (MVAId_mid) | (MVAId_high)
      eleCut = ak.all((np.abs(RelIsoEle) < 0.3) & (eleEvents.elePt[:,0] > 10) & (MVAId == True), axis=-1) | ak.all((eleEvents.elePt[:,0] > 120) & (MVAId == True), axis=-1) 
      events = events[eleCut == False]


      #Extra muon veto
      extraMuEvents = events[ak.num(events.muPt) > 1]
      RelIsoMu = (extraMuEvents.muPFChIso + extraMuEvents.muPFNeuIso + extraMuEvents.muPFPhoIso - (0.5 * extraMuEvents.muPFPUIso)) / extraMuEvents.muPt
      muCut = ak.all((np.abs(RelIsoMu) < 0.3) & (extraMuEvents.muPt[:,1] > 10) & (np.bitwise_and(extraMuEvents.muIDbit, self.bit_mask(1)) == self.bit_mask(1)), axis=-1)
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
      print("HT: ", HT)
      events = events[HT > 200]


      #Trigger Cut
      trigger_mask_Mu27 = self.bit_mask(19)
      trigger_mask_Mu50 = self.bit_mask(21)

      Mu50 = events[(events.muPt[:,0] > 52)]
      Mu50 = Mu50[(np.bitwise_and(Mu50.HLTEleMuX, trigger_mask_Mu50) == trigger_mask_Mu50)]
      Mu27 = events[(events.muPt[:,0] < 52)]
      Mu27 = Mu27[Mu27.muPt[:,0] > 27]
      Mu27 = Mu27[(Mu27.pfMetNoRecoil > 30) & (np.bitwise_and(Mu27.HLTEleMuX, trigger_mask_Mu27) == trigger_mask_Mu27)]

      #Create muon and tau candidates (recombine trigger separations)
      #If statement exists because Data does not have puTrue but Bkg does, need to define candidate without it
      if XSection != 1:
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
            "puTrue": np.concatenate((Mu27.puTrue[:,0], Mu50.puTrue[:,0]),axis=0),
            "muPFNeuIso": np.concatenate((Mu27.muPFNeuIso, Mu50.muPFNeuIso), axis=0),
            "muPFPhoIso": np.concatenate((Mu27.muPFPhoIso, Mu50.muPFPhoIso), axis=0),
            "muPFPUIso": np.concatenate((Mu27.muPFPUIso, Mu50.muPFPUIso), axis=0),
            "muPFChIso": np.concatenate((Mu27.muPFChIso, Mu50.muPFChIso), axis=0),
            "muIDbit": np.concatenate((Mu27.muIDbit, Mu50.muIDbit), axis=0),
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
				"iso": np.concatenate((Mu27.boostedTaupfTausDiscriminationByDecayModeFinding, Mu50.boostedTaupfTausDiscriminationByDecayModeFinding), axis=0),
            "antiMu": np.concatenate((Mu27.boostedTauByLooseMuonRejection3, Mu50.boostedTauByLooseMuonRejection3), axis=0),
            "puTrue": np.concatenate((Mu27.puTrue[:,0], Mu50.puTrue[:,0]),axis=0),
            "Met": np.concatenate((Mu27.pfMetNoRecoil, Mu50.pfMetNoRecoil), axis=0),
            "Metphi": np.concatenate((Mu27.pfMetPhiNoRecoil, Mu50.pfMetPhiNoRecoil), axis=0),
			   },
			   with_name="TauArray",
			   behavior=candidate.behavior,
		   )
      else:
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
            "muPFNeuIso": np.concatenate((Mu27.muPFNeuIso, Mu50.muPFNeuIso), axis=0),
            "muPFPhoIso": np.concatenate((Mu27.muPFPhoIso, Mu50.muPFPhoIso), axis=0),
            "muPFPUIso": np.concatenate((Mu27.muPFPUIso, Mu50.muPFPUIso), axis=0),
            "muPFChIso": np.concatenate((Mu27.muPFChIso, Mu50.muPFChIso), axis=0),
            "muIDbit": np.concatenate((Mu27.muIDbit, Mu50.muIDbit), axis=0),
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
				"iso": np.concatenate((Mu27.boostedTaupfTausDiscriminationByDecayModeFinding, Mu50.boostedTaupfTausDiscriminationByDecayModeFinding), axis=0),
            "antiMu": np.concatenate((Mu27.boostedTauByLooseMuonRejection3, Mu50.boostedTauByLooseMuonRejection3), axis=0),
            "Met": np.concatenate((Mu27.pfMetNoRecoil, Mu50.pfMetNoRecoil), axis=0),
            "Metphi": np.concatenate((Mu27.pfMetPhiNoRecoil, Mu50.pfMetPhiNoRecoil), axis=0),
			   },
			   with_name="TauArray",
			   behavior=candidate.behavior,
		   )

      #Split into pairs
      pairs = ak.cartesian({'tau': tau, 'muon': muon[:,0]}, nested=False)
      #Check Isolation
      IsoCheck = ak.any(((pairs['muon'].muPFNeuIso + pairs['muon'].muPFPhoIso - 0.5 * pairs['muon'].muPFPUIso) > 0.0), axis=-1)
      IsoLep1Val = np.divide(pairs['muon'].muPFChIso, pairs['muon'].pt)
      IsoLep1Val = np.where(IsoCheck, (pairs['muon'].muPFChIso + pairs['muon'].muPFNeuIso + pairs['muon'].muPFPhoIso  - 0.5 * pairs['muon'].muPFPUIso) / pairs['muon'].pt, IsoLep1Val)

      #Delta r cut
      dr = self.delta_r(pairs['tau'], pairs['muon'])
      #ID Cut
      MuID = ((np.bitwise_and(pairs['muon'].muIDbit, self.bit_mask(1)) == self.bit_mask(1)) & (np.abs(pairs['muon'].Dz) < 0.2) & (np.abs(pairs['muon'].D0) < 0.045))
      #Apply everything at once
      muTau_mask = ((np.abs(pairs['muon'].eta) < 2.4)
                  & (MuID)
                  & (pairs['muon'].pt >= 28)
                  & (pairs['tau'].pt > 30)
                  & (np.absolute(pairs['tau'].eta) < 2.3)
                  & (pairs['tau'].antiMu >= 0.5)
                  & (pairs['tau'].iso >= 0.5)
                  & (IsoLep1Val < .3)
                  & (dr > .1) 
                  & (dr < .8))
      pairs = pairs[muTau_mask]
      print(ak.count(pairs['muon'].pt))


      #Separate based on charge
      OS_pairs = pairs[pairs['tau'].charge + pairs['muon'].charge == 0]
      SS_pairs = pairs[pairs['tau'].charge + pairs['muon'].charge != 0]

      #If everything cut return 0 (to avoid pointer errors)
      if ak.sum(OS_pairs['tau'].pt) == 0 and ak.sum(SS_pairs['tau'].pt) == 0:
         return {
            dataset: {
            "mass": np.zeros(0),
            "mass_w": np.zeros(0),
            "ss_mass": np.zeros(0),
            "ss_mass_w": np.zeros(0)
            }
         }


      #Create vectors

      #OS
      muVec = self.makeVector(OS_pairs['muon'], "muon", XSection)
      tauVec = self.makeVector(OS_pairs['tau'], "tau", XSection)

      #TMass Cut
      tmass = np.sqrt(np.square(muVec.pt + OS_pairs['tau'].Met) - np.square(muVec.px + OS_pairs['tau'].Met * np.cos(OS_pairs['tau'].Metphi)) - np.square(muVec.py + OS_pairs['tau'].Met * np.sin(OS_pairs['tau'].Metphi)))
      muVec = muVec[tmass < 80]
      tauVec = tauVec[tmass < 80]

      #Separate back for trigger corrections
      #OS
      muVec50 = muVec[muVec.pt > 52]
      muVec27 = muVec[(muVec.pt < 52) & (muVec.pt > 28)]
      tauVec50 = tauVec[muVec.pt > 52]
      tauVec27 = tauVec[(muVec.pt < 52) & (muVec.pt > 28)]
      #SS (create and separate, tmass cut too)
      muVec_SS = self.makeVector(SS_pairs['muon'], "muon", XSection)
      tauVec_SS = self.makeVector(SS_pairs['tau'], "tau", XSection)
      tmass_SS = np.sqrt(np.square(muVec_SS.pt + SS_pairs['tau'].Met) - np.square(muVec_SS.px + SS_pairs['tau'].Met * np.cos(SS_pairs['tau'].Metphi)) - np.square(muVec_SS.py + SS_pairs['tau'].Met * np.sin(SS_pairs['tau'].Metphi)))
      muVec_SS = muVec_SS[tmass_SS < 80]
      tauVec_SS = tauVec_SS[tmass_SS < 80]
      muVec50_SS = muVec_SS[muVec_SS.pt > 52]
      muVec27_SS = muVec_SS[(muVec_SS.pt < 52) & (muVec_SS.pt > 28)]
      tauVec50_SS = tauVec_SS[muVec_SS.pt > 52]
      tauVec27_SS = tauVec_SS[(muVec_SS.pt < 52) & (muVec_SS.pt > 28)]

      #MET Vector
      MetVec =  ak.zip(
      {
         "pt": OS_pairs['tau'].Met,
         "eta": 0,
         "phi": OS_pairs['tau'].Metphi,
         "mass": 0,
         },
      with_name="PtEtaPhiMLorentzVector",
      behavior=vector.behavior,
      )  
      MetVec_SS =  ak.zip(
      {
         "pt": SS_pairs['tau'].Met,
         "eta": 0,
         "phi": SS_pairs['tau'].Metphi,
         "mass": 0,
         },
      with_name="PtEtaPhiMLorentzVector",
      behavior=vector.behavior,
      )  

      #Make Z Vector
      #OS
      ZVec50 = tauVec50.add(muVec50)
      ZVec27 = tauVec27.add(muVec27)
      #Make Higgs Vector
      Higgs50 = ZVec50.add(MetVec[muVec.pt > 52])
      Higgs27 = ZVec27.add(MetVec[(muVec.pt < 52) & (muVec.pt > 28)])

      #SS
      ZVec50_SS = tauVec50_SS.add(muVec50_SS)
      ZVec27_SS = tauVec27_SS.add(muVec27_SS)
      Higgs50_SS = ZVec50_SS.add(MetVec_SS[muVec_SS.pt > 52])
      Higgs27_SS = ZVec27_SS.add(MetVec_SS[(muVec_SS.pt < 52) & (muVec_SS.pt > 28)])


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
         Mu27IsoCorr = evaluator["IsoCorr"](ZVec27pt, ZVec27eta) 
         Mu27TrgCorr = evaluator["Trg27Corr"](ZVec27pt, ZVec27eta) 
         Mu27IDCorr = evaluator["IDCorr"](ZVec27pt, ZVec27eta) 

         puTrue50 = np.array(np.rint(ak.flatten(muVec50[ZVec50.pt > 200].puTrue, axis=-1)), dtype=np.int8) #Pileup
         puTrue27 = np.array(np.rint(ak.flatten(muVec27[ZVec27.pt > 200].puTrue, axis=-1)), dtype=np.int8)
         PUCorrection50 = PUWeight[puTrue50]
         PUCorrection27 = PUWeight[puTrue27]

         Lep50Corr = Mu50IsoCorr * Mu50TrgCorr * Mu50IDCorr * PUCorrection50 #Combine
         Lep27Corr = Mu27IsoCorr * Mu27TrgCorr * Mu27IDCorr * PUCorrection27

         #SS
         Mu50IsoCorr_SS = evaluator["IsoCorr"](ZVec50_SSpt, ZVec50_SSeta)   
         Mu50TrgCorr_SS = evaluator["Trg50Corr"](ZVec50_SSpt, ZVec50_SSeta)   
         Mu50IDCorr_SS = evaluator["IDCorr"](ZVec50_SSpt, ZVec50_SSeta)  
         Mu27IsoCorr_SS = evaluator["IsoCorr"](ZVec27_SSpt, ZVec27_SSeta) 
         Mu27TrgCorr_SS = evaluator["Trg27Corr"](ZVec27_SSpt, ZVec27_SSeta)  
         Mu27IDCorr_SS = evaluator["IDCorr"](ZVec27_SSpt, ZVec27_SSeta) 

         puTrue50_SS = np.array(np.rint(ak.flatten(muVec50_SS[ZVec50_SS.pt > 200].puTrue, axis=-1)), dtype=np.int8)
         puTrue27_SS = np.array(np.rint(ak.flatten(muVec27_SS[ZVec27_SS.pt > 200].puTrue, axis=-1)), dtype=np.int8)
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
   bins=np.linspace(0, 150, 30)

   #Get Pileup Weight
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
         treepath="/mutau_tree",
         schemaclass=BaseSchema,
         metadata={"dataset": dataset},
      ).events()
      string = str(sample)
      num_events = file['hEvents'].member('fEntries') / 2
      out = p.process(events, fname, num_events, PUWeight)

      #Sort files in their respective arrays
      if "DY" in string:
            DY = np.append(DY, out[dataset]["mass"], axis=0)
            DY_w = np.append(DY_w, out[dataset]["mass_w"], axis=0)
            DY_SS = np.append(DY_SS, out[dataset]["ss_mass"], axis=0)
            DY_SS_w = np.append(DY_SS_w, out[dataset]["ss_mass_w"], axis=0)
      if "WJets" in string:
            WJets = np.append(WJets, out[dataset]["mass"], axis=0)
            WJets_w = np.append(WJets_w, out[dataset]["mass_w"], axis=0)
            WJets_SS = np.append(WJets_SS, out[dataset]["ss_mass"], axis=0)
            WJets_SS_w = np.append(WJets_SS_w, out[dataset]["ss_mass_w"], axis=0)
      matches = ["WZ", "VV2l2nu", "ZZ4l", "ZZ2l2q"]
      if any([x in fname for x in matches]):
            Diboson = np.append(Diboson, out[dataset]["mass"], axis=0)
            Diboson_w = np.append(Diboson_w, out[dataset]["mass_w"], axis=0)
      matches = ["gg", "qqH125", "toptop", "WMinus", "WPlus", "ZH125", "TauTau"] 
      if any([x in fname for x in matches]):
            SMHiggs = np.append(SMHiggs, out[dataset]["mass"], axis=0)
            SMHiggs_w = np.append(SMHiggs_w, out[dataset]["mass_w"], axis=0) 
      matches = ["Tbar", "T-tchan", "tW"]
      if any([x in fname for x in matches]):
            SingleTop = np.append(SingleTop, out[dataset]["mass"], axis=0)
            SingleTop_w = np.append(SingleTop_w, out[dataset]["mass_w"], axis=0) 
      if "TTTo" in string:
            Top = np.append(Top, out[dataset]["mass"], axis=0)
            Top_w = np.append(Top_w, out[dataset]["mass_w"], axis=0) 
            Top_SS = np.append(Top_SS, out[dataset]["ss_mass"], axis=0)
            Top_SS_w = np.append(Top_SS_w, out[dataset]["ss_mass_w"], axis=0)
      if "SingleMuon" in string:
            Data = np.append(Data, out[dataset]["mass"], axis=0)
            Data_w = np.append(Data_w, out[dataset]["mass_w"], axis=0)
            Data_SS = np.append(Data_SS, out[dataset]["ss_mass"], axis=0)

   QCDScaleFactor = 1.6996559936491136


   #Turn into histograms
   Data_h, Data_bins = np.histogram(Data, bins=bins)     
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
   bins = np.linspace(0,125,25)
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


   DY_h = np.histogram(DY, bins=bins, weights=DY_w)
   TT_h = np.histogram(Top, bins=bins, weights=Top_w)
   VV_h = np.histogram(np.append(Diboson, SingleTop), bins=bins, weights=np.append(Diboson_w, SingleTop_w))
   WJets_h = np.histogram(WJets, bins=bins, weights=WJets_w)
   Data_h = np.histogram(Data, bins=bins)
   newMass = [VV_h, TT_h, WJets_h, DY_h, QCD_hist]
   labels = ["VV", "TT", "WJets", "DY", "QCD"]
   hep.histplot(newMass, label=labels, histtype=("fill"), stack=True)
   hep.histplot(Data_h, label="Data", histtype=("errorbar"), color='k')
   plt.legend(loc = 'upper right', ncols = 2, fontsize = 8)
   plt.title("Muon + Boosted Tau Visible Mass", fontsize= 'small')
   ax.set_xlabel("Mass (GeV)")
   fig.savefig("./mutau_plots/MuTau_VISIBLE_MASS.png")
   plt.clf()