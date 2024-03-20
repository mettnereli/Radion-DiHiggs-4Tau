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

   def process(self,events,name, num_events, PUWeight, syst):
      dataset = events.metadata['dataset']
      #print(events.fields)
      #Find weighting factor and identify if Data
      XSection = self.weightCalc(name)
      print(name, " ", XSection)

      #At least two muons and a jet
      events = events[ak.num(events.muPt) > 1]
      events = events[ak.num(events.jetPt) > 0]

      #Extra electron veto
      electron = ak.zip( 
		{
				"pt": events.elePt,
				"eta": events.eleEta,
				"phi": events.elePhi,
            "E": events.eleEn,
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
            "E": events.muEn,
            "muPFChIso": events.muPFChIso,
            "muPFNeuIso": events.muPFNeuIso,
            "muPFPhoIso": events.muPFPhoIso,
            "muPFPUIso": events.muPFPUIso,
            "muIDbit": events.muIDbit,
		},
			   with_name="electronArray",
		      behavior=candidate.behavior,
		) 
      muon = muon[ak.num(muon.pt) > 2]
      RelIsoMu = (muon.muPFChIso + muon.muPFNeuIso + muon.muPFPhoIso - (0.5 * muon.muPFPUIso)) / muon.pt
      badMuon = muon[(np.abs(RelIsoMu) > 0.3) & (muon.pt[:,2] > 10) & (np.bitwise_and(muon.muIDbit, self.bit_mask(2)) == self.bit_mask(2))]
      events = events[ak.num(badMuon) == 0]

      #HT Cut
      jets = ak.zip( 
		{
				"pt": events.jetPt,
				"eta": events.jetEta,
				"phi": events.jetPhi,
            "E": events.jetEn,
            "jetPFLooseId": events.jetPFLooseId,
            "jetCSV2BJetTags": events.jetCSV2BJetTags,
		},
			   with_name="JetArray",
		      behavior=vector.behavior,
		)

      goodJets= jets[(jets.jetPFLooseId > 0.5) & (jets.pt > 30) & (np.abs(jets.eta) < 3.0)]
      HT = ak.sum(goodJets.pt, axis=-1) 
      events = events[HT > 200]

      #BJet Veto
      jets = jets[HT > 200]
      bJets = jets[(jets.jetCSV2BJetTags > .7527) & (jets.jetPFLooseId > .5) & (jets.pt > 30) & (np.abs(jets.eta) < 2.4)]
      events = events[(ak.num(bJets.pt) == 0)]

   
      if syst == "_MissingEn_JESUp": events["Met"] = events.pfMET_T1JESUp; events["Metphi"]=events.pfMETPhi_T1JESUp,
      elif syst == "_MissingEn_JESDown": events["Met"]  = events.pfMET_T1JESDo;  events["Metphi"]=events.pfMETPhi_T1JESDo,
      elif syst == "_MissingEn_UESUp": events["Met"]  = events.pfMET_T1UESUp;  events["Metphi"]=events.pfMETPhi_T1UESUp,
      elif syst == "_MissingEn_UESDown": events["Met"]  = events.pfMET_T1UESDo;  events["Metphi"]=events.pfMETPhi_T1UESDo,
      else: events["Met"]  = events.pfMetNoRecoil; events["Metphi"]=events.pfMetPhiNoRecoil,

      Mu50 = events
      #Define Muon vector:
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
      #Define muon candidate
      #Need to separate with IF statement since Data doesn't have puTrue
      if XSection != 1:
         muonC = ak.zip( 
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
            "Met": events["Met"],
            "Metphi": events["Metphi"][0],
            "Trigger": Mu50.HLTEleMuX,
			},
			with_name="MuonArray",
			behavior=candidate.behavior,
		   )
      else:
         muonC = ak.zip( 
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
            "Trigger": Mu50.HLTEleMuX,
			},
			with_name="MuonArray",
			behavior=candidate.behavior,
		   )

      #Get all mumu combinations
      dimuon = ak.combinations(muon, 2, fields=['i0', 'i1']) 
      dimuonC = ak.combinations(muonC, 2, fields=['i0', 'i1'])

      #Apply triggermask
      trigger_mask_Mu50 = self.bit_mask(21)
      #Check ID
      IDMask = self.bit_mask(2)
      MuID = ((np.bitwise_and(dimuonC['i0'].ID, IDMask) == IDMask) & (np.abs(dimuonC['i0'].muD0) < 0.045) & (np.abs(dimuonC['i0'].muDz) < .2))
      Sub_MuID = ((np.bitwise_and(dimuonC['i1'].ID, IDMask) == IDMask) & (np.abs(dimuonC['i1'].muD0) < 0.045) & (np.abs(dimuonC['i1'].muDz) < .2)) 
      #Check Delta_R
      dr = dimuon['i0'].delta_r(dimuon['i1'])

      #Create mask for cuts
      dimuon_mask =  ((dimuon['i0'].pt > 53)
                     & (dimuon['i1'].pt > 10)
                     & (np.bitwise_and(dimuonC['i0'].Trigger, trigger_mask_Mu50) == trigger_mask_Mu50)
                     & (np.bitwise_and(dimuonC['i1'].Trigger, trigger_mask_Mu50) == trigger_mask_Mu50)
                     & (np.absolute(dimuon['i0'].eta) < 2.4)
                     & (np.absolute(dimuon['i1'].eta) < 2.4)
                     & (dr > .1)
                     & (dr < .8)
                     & (MuID)
                     & (Sub_MuID))

      if not ak.any(dimuon_mask):
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
      dimuon = dimuon[dimuon_mask]
      dimuonC = dimuonC[dimuon_mask]

      #Corrections
      ext = extractor()
      ext.add_weight_sets(["IDCorr NUM_LooseID_DEN_genTracks_pt_abseta ./RunBCDEF_SF_ID.root", "TrgCorr Mu50_OR_TkMu50_PtEtaBins/pt_abseta_ratio ./Trigger_EfficienciesAndSF_RunBtoF.root", "pTCorr Ratio2D ./zmm_2d_2018.root"])
      ext.finalize()
      evaluator = ext.make_evaluator()

      #Separate based on charge
      OS_pairs = dimuon[(dimuonC['i0'].charge + dimuonC['i1'].charge == 0)]
      SS_pairs = dimuon[(dimuonC['i0'].charge + dimuonC['i1'].charge != 0)]

      OS_pairsC = dimuonC[(dimuonC['i0'].charge + dimuonC['i1'].charge == 0)]
      SS_pairsC = dimuonC[(dimuonC['i0'].charge + dimuonC['i1'].charge != 0)]

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
            "LeadMuIDCorrection": np.zeros(0),
            "SubMuIDCorrection": np.zeros(0),
            "TrgCorrection": np.zeros(0),
            "PUCorrection": np.zeros(0),
            "pTCorrection": np.zeros(0),
            }
         }

      # make into vectors
      mu1Vec = OS_pairs['i0']
      mu2Vec = OS_pairs['i1']

      mu1Vec_SS = SS_pairs['i0']
      mu2Vec_SS = SS_pairs['i1']


      #TMass cut
      tmass = np.sqrt(np.square(mu1Vec.pt + OS_pairsC['i0'].Met) - np.square(mu1Vec.px + OS_pairsC['i0'].Met * np.cos(OS_pairsC['i0'].Metphi)) - np.square(mu1Vec.py + OS_pairsC['i0'].Met * np.sin(OS_pairsC['i0'].Metphi)))
      tmass_SS = np.sqrt(np.square(mu1Vec_SS.pt + SS_pairsC['i0'].Met) - np.square(mu1Vec_SS.px + SS_pairsC['i0'].Met * np.cos(SS_pairsC['i0'].Metphi)) - np.square(mu1Vec_SS.py + SS_pairsC['i0'].Met * np.sin(SS_pairsC['i0'].Metphi))) 

      mu1Vec = mu1Vec[tmass < 80]
      mu2Vec = mu2Vec[tmass < 80]

      mu1Vec_SS = mu1Vec_SS[tmass_SS < 80]
      mu2Vec_SS = mu2Vec_SS[tmass_SS < 80]

      #MET Vector
      MetVec =  ak.zip(
      {
         "pt": OS_pairsC['i0'].Met,
         "eta": 0,
         "phi": OS_pairsC['i0'].Metphi,
         "mass": 0,
         },
      with_name="PtEtaPhiMLorentzVector",
      behavior=vector.behavior,
      )  

      MetVec_SS =  ak.zip(
      {
         "pt": SS_pairsC['i0'].Met,
         "eta": 0,
         "phi": SS_pairsC['i0'].Metphi,
         "mass": 0,
         },
      with_name="PtEtaPhiMLorentzVector",
      behavior=vector.behavior,
      ) 

      MetVec = MetVec[tmass < 80]
      MetVec_SS = MetVec_SS[tmass_SS < 80]
      #Combine two vectors
      diMuVec_OS = mu1Vec.add(mu2Vec)
      diMuVec_SS = mu1Vec_SS.add(mu2Vec_SS)

      #Make Higgs Vector
      Higgs_OS = diMuVec_OS.add(MetVec)
      Higgs_SS = diMuVec_SS.add(MetVec_SS)
      


      LeadMuIDCorrection, SubMuIDCorrection, TrgCorrection, PUCorrection, pTCorrection = [], [], [], [], []

      #Get weighting array for plotting
      shape = np.shape(ak.flatten(diMuVec_OS.pt, axis=-1))
      SS_shape = np.shape(ak.flatten(diMuVec_SS.pt, axis=-1))

      #If not data, calculate all weight corrections
      if XSection != 1:

         #Luminosity (2018)
         luminosity = 59830.
         lumiWeight = (XSection * luminosity) / num_events
         
         #Pileup
         puTrue = np.array(np.rint(ak.flatten(OS_pairsC['i0'][tmass < 80].puTrue, axis=-1)), dtype=np.int8)
         SS_puTrue = np.array(np.rint(ak.flatten(SS_pairsC['i0'][tmass_SS < 80].puTrue, axis=-1)), dtype=np.int8)

         #OS
         LeadMuIDCorrection = evaluator["IDCorr"](ak.flatten(mu1Vec.pt, axis=-1), ak.flatten(mu1Vec.eta, axis=-1)) #ID
         SubMuIDCorrection = evaluator["IDCorr"](ak.flatten(mu2Vec.pt, axis=-1), ak.flatten(mu2Vec.eta, axis=-1))
         TrgCorrection = evaluator["TrgCorr"](ak.flatten(mu1Vec.pt, axis=-1), ak.flatten(mu1Vec.eta, axis=-1)) #Trigger
         PUCorrection = PUWeight[puTrue] #Pileup
         #Plot and Send
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
      mass_h = np.column_stack((ak.flatten(diMuVec_OS.mass, axis=-1), mass_w))
      SS_mass_h = np.column_stack((ak.flatten(diMuVec_SS.mass, axis=-1), SS_mass_w))

      pt_h = np.column_stack((ak.flatten(diMuVec_OS.pt, axis=-1), mass_w))
      SS_pt_h = np.column_stack((ak.flatten(diMuVec_SS.pt, axis=-1), SS_mass_w))

      eta_h = np.column_stack((ak.flatten(diMuVec_OS.eta, axis=-1), mass_w))
      
      #200 GeV pT cut, 60-120 mass cut
      mass_h, pt_h, eta_h = mass_h[(pt_h[:,0] > 200) & (ak.flatten(Higgs_OS.pt, axis=-1) > 250)], pt_h[(pt_h[:,0] > 200) & (ak.flatten(Higgs_OS.pt, axis=-1) > 250)], eta_h[(pt_h[:,0] > 200) & (ak.flatten(Higgs_OS.pt, axis=-1) > 250)]
      mass_h, pt_h, eta_h = mass_h[(mass_h[:,0] > 60) & (mass_h[:,0] < 120)], pt_h[(mass_h[:,0] > 60) & (mass_h[:,0] < 120)], eta_h[(mass_h[:,0] > 60) & (mass_h[:,0] < 120)]

      SS_mass_h = SS_mass_h[(SS_pt_h[:,0] > 200) & (ak.flatten(Higgs_SS.pt, axis=-1) > 250)]
      SS_mass_h = SS_mass_h[(SS_mass_h[:,0] > 60) & (SS_mass_h[:,0] < 120)]

      return {
         dataset: {
            "mass": mass_h[:,0],
            "mass_w": mass_h[:,1],
            "pT": pt_h[:,0],
            "eta": eta_h[:,0],
            "ss_mass": SS_mass_h[:,0],
            "ss_mass_w": SS_mass_h[:,1],
            "LeadMuIDCorrection": LeadMuIDCorrection,
            "SubMuIDCorrection": SubMuIDCorrection,
            "TrgCorrection": TrgCorrection,
            "PUCorrection": PUCorrection,
            "pTCorrection": pTCorrection,
         }
      }

   def run_cuts(self, fileList, directory, dataset, PUWeight, corrArrays, en_var):
      for sample in fileList:

         if ("SingleMuon" in sample) and (en_var != "_nominal"): continue

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
         out = p.process(events, fname, num_events, PUWeight, en_var)

         #for i in corrArrays:
         #   d[i + "" + en_var] = np.append(d[i + "" + en_var], out[dataset][f"{i}"], axis=0)

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
   directory = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedH2/2018/mm/v2_Hadd/"
   dataset = "DiMuon"
   fig, ax = plt.subplots()
   #hep.style.use(hep.style.ROOT)
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

   jet_variations = ["_JESUp", "_JESDown"]
   met_variations = ["_nominal", "_MissingEn_JESUp", "_MissingEn_JESDown"] #"_MissingEn_UESUp", "_MissingEn_UESDown"]
   u_variations = ["_MissingEn_UESUp", "_MissingEn_UESDown", "_nominal"]

   allSysts = ["_nominal", "_JESUp", "_JESDown", "_MissingEn_JESUp", "_MissingEn_JESDown", "_MissingEn_UESUp", "_MissingEn_UESDown"]
   d = {}

   emptyArrays = ["DY", "WJets", "Diboson", "SMHiggs", "SingleTop", "Top"]
   corrArrays= ["LeadMuIDCorrection", "SubMuIDCorrection", "TrgCorrection", "PUCorrection", "pTCorrection"]
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

   bins=np.linspace(60, 120, 2)


   #Get pileup reweighting factors
   with uproot.open("pu_distributions_mc_2018.root") as f1:
      with uproot.open("pu_distributions_data_2018.root") as f2:
         mc = f1["pileup"].values()
         data = f2["pileup"].values()
         HistoPUMC = np.divide(mc, ak.sum(mc))
         HistoPUData = np.divide(data, ak.sum(data))
         PUWeight = np.divide(HistoPUData, HistoPUMC)


   #for jet_var in jet_variations:
   #   print("Variation: ", jet_var)
   #   p.run_cuts(fileList, directory, dataset, PUWeight, corrArrays, jet_var)

      
   #for met_var in met_variations:
   #   print("Variation: ", met_var)
   #   p.run_cuts(fileList, directory, dataset, PUWeight, corrArrays, met_var)

   for u_var in u_variations:
      print("Variation: ", u_var)
      p.run_cuts(fileList, directory, dataset, PUWeight, corrArrays, u_var)    

 

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

   # #Mass and labels for plotting
   # mass =   [SMHiggs_h,   Diboson_h,  SingleTop_h,   DY_h,   Top_h, WJets_h, QCD_h]
   # labels = ["SMHiggs", "Diboson", "SingleTop", "DY", "Top", "WJets", "QCD"]

   # #Plot MuMu Visible Mass
   # hep.histplot(mass, label=labels, histtype=("fill"), bins=bins, stack=True)
   # hep.histplot(Data_h, label="Data", histtype=("errorbar"), bins=bins, color="k")
   # plt.legend(loc = 'upper right', ncols = 2, fontsize = 8)
   # #plt.yscale("log")
   # #OS boostedTau visible mass
   # plt.title("DiMuon Visible Mass", fontsize= 'small')
   # ax.set_xlabel("Mass (GeV)")
   # ax.set_ylim(bottom=0)
   # fig.savefig("./mumu_plots/DIMUON_VISIBLE_MASS.png")
   # plt.clf()


   # #Plot MuMu SS Visible Mass
   # ss_mass = [DY_SS_h, Top_SS_h, WJets_SS_h]
   # ss_mass_labels = ["DY", "Top", "WJets"]

   # hep.histplot(ss_mass, label=ss_mass_labels, histtype=("fill"), bins=bins, stack=True)
   # hep.histplot(Data_SS_h, label="Data", histtype=("errorbar"), bins=bins, color="k")
   # plt.legend(loc = 'upper right', ncols = 2, fontsize = 8)

   # plt.title("DiMuon Visible Mass (SS Region)", fontsize= 'small')
   # ax.set_xlabel("Mass (GeV)")
   # ax.set_ylim(bottom=0)
   # fig.savefig("./mumu_plots/SS_DIMUON_VISIBLE_MASS.png")
   # plt.clf()

   # corrBins1= np.linspace(.85, 1.15, 90)
   # corrBins2= np.linspace(0, 2, 80)
   # plt.hist(LeadMuIDCorrection, bins=corrBins1)
   # plt.title("LeadMuIDCorr")
   # fig.savefig("./mumu_plots/correction_plots/LeadMuIDCorr.png")
   # plt.clf()


   # plt.hist(SubMuIDCorrection, bins=corrBins1)
   # plt.title("SubMuIDCorr")
   # fig.savefig("./mumu_plots/correction_plots/SubMuIDCorr.png")
   # plt.clf()


   # plt.hist(TrgCorrection, bins=corrBins1)
   # plt.title("TrgCorrection")
   # fig.savefig("./mumu_plots/correction_plots/TrgCorr.png")
   # plt.clf()

   # plt.hist(PUCorrection, bins=corrBins2)
   # plt.title("PUCorrection")
   # fig.savefig("./mumu_plots/correction_plots/PUCorrection.png")
   # plt.clf()

   # plt.hist(pTCorrection, bins=corrBins1)
   # plt.title("pTCorrection")
   # fig.savefig("./mumu_plots/correction_plots/pTCorrection.png")
   # plt.clf() 

   #Write to file
   oneBin = np.linspace(60, 120, num=2)


   outFile = uproot.recreate("boostedHTT_mm_2018_UEn.input.root")
   for en_var in allSysts:
      #Send files out for Correction factor finding
      if en_var == "_nominal":
         d["Data_h"] = np.histogram(d["Data"], bins=oneBin) 
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
            d[i + "_h" + en_var] = np.histogram(np.append(d["Diboson" + en_var], d["SingleTop" + en_var]), bins=oneBin, weights=np.append(d["Diboson_w" + en_var], d["SingleTop_w" + en_var]) ) 
         else: d[i + "_h" + en_var] = np.histogram(d[j + "" + en_var], bins=oneBin, weights= d[j + "_w" + en_var]) 
         
         if i == "DY": k = "DYJets125"
         else: k = i

         outFile["DYJets_met_1_13TeV/" + k + "" + en_var] = d[i + "_h" + en_var]

   print("Past outFile!")

   #Plot visible mass with Abdollah's specifications
   # newMass = [VV_h, TT_h, WJets_h, (DY_hist, DY_bins), QCD_hist]
   # labels = ["VV", "TT", "WJets", "DY", "QCD"]
   # hep.histplot(newMass, label=labels, histtype=("fill"), stack=True)
   # hep.histplot(Data_h, label="Data", histtype=("errorbar"), color="k")
   # plt.legend(loc = 'upper right', ncols = 2, fontsize = 8)
   # #plt.yscale("log")
   # #OS boostedTau visible mass
   # plt.title("DiMuon Visible Mass", fontsize= 'small')
   # ax.set_xlabel("Mass (GeV)")
   # ax.set_ylim(bottom=0)
   # fig.savefig("./mumu_plots/mumu_VISIBLE_MASS.png")
   # plt.clf()

   # #Plot Visible mass again this time matching Abdollah's specifications and QCD Estimation
   # #Data-driven QCD Estimation
   # bins = np.append(np.linspace(60, 120, 1, endpoint=False), 120.)
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

   # fig = plt.figure(figsize=(10, 8))
   # ax = hist.axis.Regular(1, 60, 120, name=r"$m_{\mu \mu}$", flow=False)
   # cax = hist.axis.StrCategory(["VV", "TT", "WJets", "QCD", "DY"], name="c")
   # full_Hist = Hist(ax, cax)
   # full_Hist.fill(DY, weight=DY_w, c="DY")
   # full_Hist.fill(Top, weight=Top_w, c="TT")
   # full_Hist.fill(WJets, weight=WJets_w, c="WJets")
   # full_Hist.fill(np.append(Diboson, SingleTop), weight=np.append(Diboson_w, SingleTop_w), c="VV")
   # full_Hist[:, hist.loc("QCD")] = QCD_h
   # s = full_Hist.stack("c")
   # s.plot(stack=True, histtype="fill")
   # hist_2 = hist.Hist(hist.axis.Regular(1, 60, 120, name=r"$m_{\mu \mu}$", label=r"$m_{\mu \mu}$", flow=False))
   # hist_2.fill(Data)
   # hist_2.plot(histtype="errorbar", color='black')
   # plt.legend(loc = 'upper right', ncols = 2, fontsize = 8)
   # fig.savefig("./mumu_plots/mumu_VISIBLE_MASS.png")
   # plt.clf() 




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
   # fig.savefig("./mumu_plots/mumu_Ratio_VISIBLE_MASS.png")
   # plt.clf()


   # #More histos
   # bins = np.linspace(200, 1000, 80)
   # Data_h, Data_bins = np.histogram(Data_pt, bins=bins)
   # DY_h, DY_bins = np.histogram(DY_pt, bins=bins, weights=DY_w)
   # WJets_h, WJets_bins = np.histogram(WJets_pt, bins=bins, weights=WJets_w) 
   # Top_h, Top_bins = np.histogram(Top_pt, bins=bins, weights= Top_w) 
   # SingleTop_h, SingleTop_bins = np.histogram(SingleTop_pt, bins=bins, weights=SingleTop_w) 
   # Diboson_h, Diboson_bins = np.histogram(Diboson_pt, bins=bins, weights=Diboson_w)
   # SMHiggs_h, SMHiggs_bins = np.histogram(SMHiggs_pt, bins=bins, weights=SMHiggs_w)   

   # #PT
   # pT =   [SMHiggs_h,   Diboson_h,   SingleTop_h,   DY_h,   Top_h,   WJets_h]
   # labels = ["SMHiggs", "Diboson", "SingleTop", "DY", "Top", "WJets"]
   # hep.histplot(pT, label=labels, histtype=("fill"), bins=bins, stack=True)
   # hep.histplot(Data_h, label="Data", histtype=("errorbar"), bins=bins, color='k')
   # #plt.yscale("log")
   # plt.legend(loc = 'upper right', ncols = 2, fontsize = 8)
   # #OS boostedTau pT fakerate
   # plt.title("DiMuon pT", fontsize= 'small')
   # ax.set_xlabel("pT (GeV)")
   # fig.savefig("./mumu_plots/DIMUON_pT.png")
   # plt.clf()

   # #Eta
   # bins = np.linspace(-3, 3, 20)
   # Data_h, Data_bins = np.histogram(Data_eta, bins=bins)
   # DY_h, DY_bins = np.histogram(DY_eta, bins=bins, weights=DY_w)
   # WJets_h, WJets_bins = np.histogram(WJets_eta, bins=bins, weights=WJets_w) 
   # Top_h, Top_bins = np.histogram(Top_eta, bins=bins, weights= Top_w) 
   # SingleTop_h, SingleTop_bins = np.histogram(SingleTop_eta, bins=bins, weights=SingleTop_w) 
   # Diboson_h, Diboson_bins = np.histogram(Diboson_eta, bins=bins, weights=Diboson_w)
   # SMHiggs_h, SMHiggs_bins = np.histogram(SMHiggs_eta, bins=bins, weights=SMHiggs_w)   

   # eta =   [SMHiggs_h,   Diboson_h,   SingleTop_h,   DY_h,   Top_h,   WJets_h]
   # hep.histplot(eta, label=labels, histtype=("fill"), bins=bins, stack=True)
   # hep.histplot(Data_h, label="Data", histtype=("errorbar"), bins=bins, color='k')
   # plt.legend(loc = 'upper right', ncols = 2, fontsize = 8)
   # ax.set_ylim(bottom=0)
   # #eta
   # #plt.yscale("log")
   # plt.title("DiMuon eta", fontsize= 'small')
   # ax.set_xlabel("Radians")
   # fig.savefig("./mumu_plots/DIMUON_eta.png")

 