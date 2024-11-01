import sys
import os
from pathlib import Path
import math
import json
import awkward as ak
import numpy as np
import uproot
import boost_histogram as bh
import hist
from hist import Hist, intervals, axis
import matplotlib.pyplot as plt
from matplotlib import cycler
import mplhep as hep
from coffea import nanoevents, lookup_tools, util
import coffea.processor as processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea.nanoevents.methods import vector, candidate, nanoaod
from coffea.nanoevents.methods.vector import PtEtaPhiELorentzVector
from coffea.lookup_tools import extractor, evaluator
from coffea.analysis_tools import PackedSelection, Weights
import dask
import correctionlib
from dask import delayed

#Bit mask method (not utilized anymore)
def bit_mask(bit):
      mask = 0
      mask += (1 << bit)
      return mask

#Function containing all xs values for each process (for luminosity weighting)
def weightCalc(name):
      WScaleFactor = 1.21
      TT_FulLep_BR= 0.1061
      TT_SemiLep_BR= 0.4392
      TT_Had_BR= 0.4544

      if "Data" in name: return 1
      elif "WZTo1L1Nu2Q" in name: return 9.119
      elif "WZTo1L3Nu" in name: return 3.414
      elif "WZTo2Q2L" in name: return 6.565
      elif "WWTo1L1Nu2Q" in name: return 51.65
      elif "ZZTo2Q2L" in name: return 3.676
      elif "ZZTo4L" in name: return 1.325
      elif "ZZTo2Nu2Q" in name: return 4.545
      elif "ST_s-channel_4f_leptonDecays" in name: return 3.549
      elif "ST_t-channel_antitop_4f_InclusiveDecays" in name: return 67.93
      elif "ST_t-channel_top_4f_InclusiveDecays" in name: return 113.4
      elif "ST_tW_antitop_5f_inclusiveDecays" in name: return 32.51
      elif "ST_tW_top_5f_inclusiveDecays" in name: return 32.45
      elif "DY" in name:
         if "70to100" in name: return 140
         elif "100to200" in name: return 139.2
         elif "200to400" in name: return 38.4
         elif "400to600" in name: return 5.174
         elif "600to800" in name: return 1.258
         elif "800to1200" in name: return 0.5598
         elif "1200to2500" in name: return 0.1305
         elif "2500toInf" in name: return 0.002997
         elif "0To50" in name: return 1485
         elif "50To100" in name: return 397.4
         elif "100To250" in name: return 97.2
         elif "250To400" in name: return 3.701
         elif "400To650" in name: return 0.5086
         elif "650ToInf" in name: return 0.04728
      elif "WJets" in name:
         if "70To100" in name: return 1283
         elif "100To200" in name: return 1244
         elif "200To400" in name: return 337.8
         elif "400To600" in name: return 44.93
         elif "600To800" in name: return 11.19
         elif "800To1200" in name: return 4.926
         elif "1200To2500" in name: return 1.152
         elif "2500ToInf" in name: return 0.02646
      elif "TTTo" in name:
         if "2L2Nu" in name: return 87.31
         elif "Hadronic" in name: return 378.93
         elif "SemiLeptonic" in name: return 364.35
      else: 
         print("Something's Wrong!")
         return 1.0


#Contains sum of all gen weights (for luminosity reweighting)
def sumGenCalc(name):
    if "Data" in name: return 1
    elif "WZTo1L1Nu2Q" in name: return sumGen['WZTo1L1Nu2Q_4f']
    elif "WZTo1L3Nu" in name: return sumGen['WZTo1L3Nu_4f']
    elif "WZTo2Q2L" in name: return sumGen['WZTo2Q2L_mllmin4p0']
    elif "WWTo1L1Nu2Q" in name: return sumGen['WWTo1L1Nu2Q_4f']
    elif "ZZTo2Q2L" in name: return sumGen['ZZTo2Q2L_mllmin4p0']
    elif "ZZTo4L" in name: return sumGen['ZZTo4L']
    elif "ZZTo2Nu2Q" in name: return sumGen['ZZTo2Nu2Q_5f']
    elif "ST_s-channel_4f_leptonDecays" in name: return sumGen['ST_s-channel_4f_leptonDecays']
    elif "ST_t-channel_antitop_4f_InclusiveDecays" in name: return sumGen['ST_t-channel_antitop_4f_InclusiveDecays']
    elif "ST_t-channel_top_4f_InclusiveDecays" in name: return sumGen['ST_t-channel_top_4f_InclusiveDecays']
    elif "ST_tW_antitop_5f_inclusiveDecays" in name: return sumGen['ST_tW_antitop_5f_inclusiveDecays']
    elif "ST_tW_top_5f_inclusiveDecays" in name: return sumGen['ST_tW_top_5f_inclusiveDecays']
    elif "DY" in name:
        if "70to100" in name: return sumGen['DYJetsToLL_M-50_HT-70to100']
        elif "100to200" in name: return sumGen['DYJetsToLL_M-50_HT-100to200']
        elif "200to400" in name: return sumGen['DYJetsToLL_M-50_HT-200to400']
        elif "400to600" in name: return sumGen['DYJetsToLL_M-50_HT-400to600']
        elif "600to800" in name: return sumGen['DYJetsToLL_M-50_HT-600to800']
        elif "800to1200" in name: return sumGen['DYJetsToLL_M-50_HT-800to1200']
        elif "1200to2500" in name: return sumGen['DYJetsToLL_M-50_HT-1200to2500']
        elif "2500toInf" in name: return sumGen['DYJetsToLL_M-50_HT-2500toInf']
        elif "0To50" in name: return 2455892442182.9556
        elif "50To100" in name: return 578496149642.9556
        elif "100To250" in name: return 47895381724.01761
        elif "250To400" in name: return 363928660.5752002
        elif "400To650" in name: return 6003499.3821627
        elif "650ToInf" in name: return 647942.7325666503
    elif "WJets" in name:
        if "70To100" in name: return sumGen['WJetsToLNu_HT-70To100']
        elif "100To200" in name: return sumGen['WJetsToLNu_HT-100To200']
        elif "200To400" in name: return sumGen['WJetsToLNu_HT-200To400']
        elif "400To600" in name: return sumGen['WJetsToLNu_HT-400To600']
        elif "600To800" in name: return sumGen['WJetsToLNu_HT-600To800']
        elif "800To1200" in name: return sumGen['WJetsToLNu_HT-800To1200']
        elif "1200To2500" in name: return sumGen['WJetsToLNu_HT-1200To2500']
        elif "2500ToInf" in name: return sumGen['WJetsToLNu_HT-2500ToInf']
    elif "TTTo" in name:
        if "2L2Nu" in name: return sumGen['TTTo2L2Nu']
        elif "Hadronic" in name: return sumGen['TTToHadronic']
        elif "SemiLeptonic" in name: return sumGen['TTToSemiLeptonic']
    else: 
        print("Something's Wrong!")
        return 1.0

#Function containing total events for each process
def totalEventCalc(name):
    if "Data" in name: return 1
    elif "WZTo1L1Nu2Q" in name: return 7395487.0
    elif "WZTo1L3Nu" in name: return 2497292.0
    elif "WZTo2Q2L" in name: return 28576996.0
    elif "WWTo1L1Nu2Q" in name: return 40272013.0
    elif "ZZTo2Q2L" in name: return 29357938.0
    elif "ZZTo4L" in name: return 75310000.0 + 23158000.0
    elif "ZZTo2Nu2Q" in name: return 4950564.0
    elif "ST_s-channel_4f_leptonDecays" in name: return 19365999.0
    elif "ST_t-channel_antitop_4f_InclusiveDecays" in name: return 95833000.0
    elif "ST_t-channel_top_4f_InclusiveDecays" in name: return 178756000.0
    elif "ST_tW_antitop_5f_inclusiveDecays" in name: return 7749000.0
    elif "ST_tW_top_5f_inclusiveDecays" in name: return 7956000.0
    elif "DY" in name:
        if "0To50" in name: return 1
        elif "50To100" in name: return 123100779.0
        elif "100To250" in name: return 94224909.0
        elif "250To400" in name: return 23792059.0
        elif "400To650" in name: return 3518299.0
        elif "650ToInf" in name: return 3934651.0
    elif "WJets" in name:
        if "70To100" in name: 66569448.0
        elif "100To200" in name: return 51541593.0
        elif "200To400" in name: return 58331446.0
        elif "400To600" in name: return 9415474.0
        elif "600To800" in name: return 13472940.0
        elif "800To1200" in name: return 12725029.0
        elif "1200To2500" in name: return 12967787.0
        elif "2500ToInf" in name: return 12222295.0
    elif "TTTo" in name:
        if "2L2Nu" in name: return 146010000.0
        elif "Hadronic" in name: return 343200000.0
        elif "SemiLeptonic" in name: return 476966000.0
    else: 
        print("Something's Wrong!")
        return 1.0

#Quick function to make a PtEtaPhiMLorentzVector candidate
def makeVector(particle):
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

#Processor class
class MyProcessor(processor.ProcessorABC):
    def __init__(self):
        ak.behavior.update(nanoaod.behavior)
        self._accumulator = {}
    
    @property
    def accumulator(self):
        return self._accumulator


    #MAIN PROCESS CODE STARTS HERE
    def process(self,events):
        dataset = events.metadata["dataset"]
        
        print("Running new event")
        ax = hist.axis.Regular(30, 0, 150, flow=False, name="mass", label=dataset)
        output = {}
        eventCount = len(events)

        #Create dictionary containing all histograms to be written
        output[dataset] =  {
            "mass": Hist(hist.axis.Regular(30, 0, 150, flow=False, name="mass", label=dataset), storage=hist.storage.Weight()),
            "ss_mass": Hist(hist.axis.Regular(30, 0, 150, flow=False, name="mass", label=dataset), storage=hist.storage.Weight()),
            "pt": Hist(hist.axis.Regular(30, 150, 400, flow=False, name="pt", label=dataset), storage=hist.storage.Weight()),
            "ss_pt": Hist(hist.axis.Regular(30, 150, 400, flow=False, name="pt", label=dataset), storage=hist.storage.Weight()),
            "eta": Hist(hist.axis.Regular(40, -4, 4, flow=False, name="eta", label=dataset), storage=hist.storage.Weight()),
            "ss_eta": Hist(hist.axis.Regular(40, -4, 4, flow=False, name="eta", label=dataset), storage=hist.storage.Weight()),
            "lumiWeight": Hist(hist.axis.Regular(40, -5, 5, flow=False, name="lumiWeight", label=dataset), storage=hist.storage.Weight()),
            "IsoCorr": Hist(hist.axis.Regular(40, 0, 2, flow=False, name="isoCorr", label=dataset), storage=hist.storage.Weight()),
            "IDCorr": Hist(hist.axis.Regular(40, 0, 2, flow=False, name="IDCorr", label=dataset), storage=hist.storage.Weight()),
            "Trg27Corr": Hist(hist.axis.Regular(40, 0, 2, flow=False, name="Trg27Corr", label=dataset), storage=hist.storage.Weight()),
            "Trg50Corr": Hist(hist.axis.Regular(40, 0, 2, flow=False, name="Trg50Corr", label=dataset), storage=hist.storage.Weight()),
            "puCorr": Hist(hist.axis.Regular(40, 0, 2, flow=False, name="puCorr", label=dataset), storage=hist.storage.Weight()),
            "lepCorr": Hist(hist.axis.Regular(100, -2, 2, flow=False, name="lepCorr", label=dataset), storage=hist.storage.Weight()),
            "dr": Hist(hist.axis.Regular(40, 0, 1, flow=False, name="dr", label=dataset), storage=hist.storage.Weight()),
            "dr_test": Hist(hist.axis.Regular(40, 0, 1, flow=False, name="dr", label=dataset), storage=hist.storage.Weight()),
            "muPt": Hist(hist.axis.Regular(50, 29, 200, flow=False, name="pt", label=dataset), storage=hist.storage.Weight()),
            "tauPt": Hist(hist.axis.Regular(50, 29, 200, flow=False, name="pt", label=dataset), storage=hist.storage.Weight()),
            "total": Hist(hist.axis.Regular(2, 0, 2, flow=False, name="count", label=dataset), storage=hist.storage.Weight()),
            "hcount": Hist(hist.axis.Regular(12, 0, 12, flow=False, name="count", label=dataset), storage=hist.storage.Weight()),
            "tmass": Hist(hist.axis.Regular(150, 0, 150, flow=False, name="mass", label=dataset), storage=hist.storage.Weight()),
            "zPt": Hist(hist.axis.Regular(100, 0, 400, flow=False, name="pt", label=dataset), storage=hist.storage.Weight()),
            "HiggsPt": Hist(hist.axis.Regular(100, 0, 400, flow=False, name="pt", label=dataset), storage=hist.storage.Weight())
        }      
        #Fill histograms (starting with total number of events)
        output[dataset]["total"].fill(count= np.ones(len(events)))

        #Get name
        name = str(events.metadata["filename"])

        #Get function calls from above
        XSection = weightCalc(name)
        sumOfGenWeights = sumGenCalc(name)
        totalWeight = totalEventCalc(name)


        #Select events with at least one muon, at least one tau, met pt > 30, passing either Mu27 or Mu50 HLT
        events = events[(ak.num(events.Muon) > 0)]
        events = events[(ak.num(events.boostedTau) > 0)]
        events = events[events.MET.pt > 30]
        events = events[events.HLT.Mu27 | events.HLT.Mu50]
        output[dataset]["hcount"].fill(count= np.ones(len(events)) * 2)    


        #Split into pairs - ak.cartesian creates pairs of all boostedTaus and Muons in event, a columnar for-loop equivalent
        pairs = ak.cartesian({'tau': events.boostedTau, 'muon': events.Muon}, axis=1, nested=False) 
        MuID = (pairs['muon'].tightId) & (np.abs(pairs['muon'].dz) < 0.2) & (np.abs(pairs['muon'].dxy) < 0.045)
        dr = pairs['muon'].delta_r(pairs['tau'])

        #Apply cuts to pairs
        pairs = pairs[((np.abs(pairs['muon'].eta) < 2.4)
              & (MuID)
              & (pairs['muon'].pt > 28)
              & (pairs['tau'].pt > 30)
              & (np.absolute(pairs['tau'].eta) < 2.3)
              & (pairs['tau'].idAntiMu >= 2)
              & (pairs['tau'].idDecayModeNewDMs)
              & (pairs['tau'].idMVAnewDM2017v2 >= 4)
              & (pairs['tau'].idAntiEle2018 >= 2)
              & (dr > .1)
              & (dr < .8))]

        #Remove events that contain no valid pairs
        events = events[ak.num(pairs, axis=-1) > 0]
        #Remove empty events from pair array [[pair], [pair], []] -> [[pair], [pair]]
        pairs = pairs[ak.num(pairs, axis=-1) > 0] 
        
        output[dataset]["hcount"].fill(count= np.ones(len(events))* 3)


        #Extra Electron veto
        electron = events.Electron
        ele_cut = (electron.pt >= 15) & (np.abs(electron.eta) <= 2.5) & electron.mvaIso_WPL
        pairs = pairs[(ak.any(ele_cut, axis=-1) == False)]
        events = events[ak.num(pairs, axis=-1) > 0]       
        pairs = pairs[ak.num(pairs, axis=-1) > 0] 

        output[dataset]["hcount"].fill(count= np.ones(len(events))* 4)
        
        #Extra muon veto
        i0_pass = (pairs['muon'].pfRelIso04_all < .3) & (pairs['muon'].pt > 10) & (pairs['muon'].tightId) #if leading mu passes cut
        #check if all muons pass cut
        extraMuon = events.Muon[((events.Muon.pfRelIso04_all < .3) & (events.Muon.pt > 10) & (events.Muon.tightId))] 
        overallVeto = (ak.num(extraMuon, axis=-1) < 2)
        ExtraVeto = i0_pass & (ak.num(extraMuon, axis=-1) == 1)
        pairs = pairs[overallVeto | ExtraVeto]
        events = events[(ak.num(pairs, axis=-1) > 0)]
        pairs = pairs[(ak.num(pairs, axis=-1) > 0)]
        output[dataset]["hcount"].fill(count= np.ones(len(events))* 5)
        
        #Jets - HT
        goodJets= events.Jet[(events.Jet.jetId >= 1) & (events.Jet.pt > 30) & (np.abs(events.Jet.eta) < 3.0)]
        HT = ak.sum(goodJets.pt, axis=-1)
        pairs = pairs[(HT > 200)]
        events = events[ak.num(pairs, axis=-1) > 0]       
        pairs = pairs[ak.num(pairs, axis=-1) > 0] 
        
        output[dataset]["hcount"].fill(count= np.ones(len(events))* 6)

        #BJet Veto
        #!!!!!!!!!!!!!!! change btagdeepflavb for each year https://btv-wiki.docs.cern.ch/ScaleFactors/UL2018/
        bJets = (events.Jet.btagDeepFlavB > .7100) & (events.Jet.jetId >= 1) & (events.Jet.pt > 30) & (np.abs(events.Jet.eta) < 2.4)
        pairs = pairs[ak.any(bJets, axis=-1) == False]
        events = events[ak.num(pairs, axis=-1) > 0]       
        pairs = pairs[ak.num(pairs, axis=-1) > 0] 
        
        output[dataset]["hcount"].fill(count= np.ones(len(events))* 7)   


        
        #Define 4Vectors
        muon = makeVector(pairs['muon'])
        tau = makeVector(pairs['tau'])

        #If all events are gone, just return now
        if len(events.MET) == 0: return output
        if len(events.MET.pt) == 0: return output

        #TMass Cut
        tmass = np.sqrt(np.square(muon.pt + events.MET.pt) - np.square(muon.px + events.MET.pt * np.cos(events.MET.phi)) - np.square(muon.py + events.MET.pt * np.sin(events.MET.phi)))
        pairs = pairs[tmass < 40]
        events = events[ak.num(pairs, axis=-1) > 0]       
        pairs = pairs[ak.num(pairs, axis=-1) > 0] 
        output[dataset]["hcount"].fill(count= np.ones(len(events))* 8)
        output[dataset]["tmass"].fill(mass=ak.ravel(tmass))

        #ZVec pt cut
        muon = makeVector(pairs['muon'])
        tau = makeVector(pairs['tau'])
        ZVec = tau.add(muon)
        ptCut = (ZVec.pt > 200)
        pairs = pairs[ptCut]
        events = events[ak.num(pairs, axis=-1) > 0]       
        pairs = pairs[ak.num(pairs, axis=-1) > 0] 
        output[dataset]["hcount"].fill(count= np.ones(len(events))* 9) 

        #Same check again if no events
        if len(events.MET) == 0: return output
        if len(events.MET.pt) == 0: return output


        
        #Create MET Vector
        MetVec =  ak.zip(
        {
        "pt": events.MET.pt,
        "eta": 0,
        "phi": events.MET.phi,
        "mass": 0,
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
        ) 

        #Higgs PT cut
        muon = makeVector(pairs['muon'])
        tau = makeVector(pairs['tau'])
        ZVec = tau.add(muon)        
        Higgs = ZVec.add(MetVec)
        HiggsCut = (Higgs.pt > 250) 
        
        pairs = pairs[HiggsCut]
        events = events[ak.num(pairs, axis=-1) > 0]       
        pairs = pairs[ak.num(pairs, axis=-1) > 0] 
        output[dataset]["hcount"].fill(count= np.ones(len(events))* 10)
        output[dataset]["HiggsPt"].fill(pt= ak.ravel(Higgs.pt)) 
        
        #Create Z Candidate and select one Z Candidate per event
        muon = makeVector(pairs['muon'])
        tau = makeVector(pairs['tau'])
        ZVec = tau.add(muon)
        #If there are multiple Z Candidates per event:
        if ak.any(ak.num(ZVec, axis=-1) > 1, axis=-1):
            #Select whichever Z Candidate has closest mass to 91.187, and make sure dimuon keeps that too
            ZVec = ZVec[ak.argmin(np.absolute(ZVec.mass - 91.187), axis=-1, keepdims=True)]
            pairs = pairs[ak.argmin(np.absolute(ZVec.mass - 91.187), axis=-1, keepdims=True)]

        #Split events by trigger - if between 27 and 52 pt and passing isolation, then it is Mu27. Any pt >= 52 if Mu50
        trg27 = ((pairs['muon'].pt >= 27)
              & (pairs['muon'].pt < 52)
              & (pairs['muon'].pfRelIso04_all > .3))
        trg50 = ((pairs['muon'].pt >= 52))

        #Get weights
        if XSection != 1: 
            #Apply trigger
            pairs27 = pairs[trg27]
            pairs50 = pairs[trg50]

            #Split by trigger 
            events27 = events[ak.num(pairs27, axis=1) > 0]       
            pairs27 = pairs[ak.num(pairs27, axis=1) > 0] 

            events50 = events[ak.num(pairs50, axis=1) > 0]       
            pairs50 = pairs[ak.num(pairs50, axis=1) > 0] 

            output[dataset]["hcount"].fill(count= np.ones(len(events27))* 11 )
            output[dataset]["hcount"].fill(count= np.ones(len(events50))* 11 )


            #Luminosity reweighting
            luminosity2018 = 59830.
            lumiWeight27 = np.multiply(((XSection * luminosity2018) / sumOfGenWeights), events27.genWeight)
            lumiWeight50 = np.multiply(((XSection * luminosity2018) / sumOfGenWeights), events50.genWeight)

            #If statement to make sure that that there are still events that pass Mu27
            if len(events27.genWeight) > 0:
                #Fill weights for Mu27
                output[dataset]["lumiWeight"].fill(lumiWeight=ak.ravel(lumiWeight27))
                mu27 = makeVector(pairs27['muon'])
                tau27 = makeVector(pairs27['tau'])

                output[dataset]["muPt"].fill(pt=ak.ravel(mu27.pt), weight=ak.ravel(lumiWeight27))
                output[dataset]["tauPt"].fill(pt=ak.ravel(tau27.pt), weight=ak.ravel(lumiWeight27))

                Z27 = tau27.add(mu27)

                #Only take events between 0 and 150 GeV mass
                Z27 = Z27[(Z27.mass < 150)]
                lumiWeight27 = lumiWeight27[(Z27.mass < 150)]
                
                output[dataset]["zPt"].fill(pt= ak.ravel(Z27.pt), weight=ak.ravel(lumiWeight27)) 
                MuIsoCorr27 = evaluator["IsoCorr"](Z27.pt, Z27.eta)
                output[dataset]["IsoCorr"].fill(isoCorr=ak.ravel(MuIsoCorr27))

                MuIDCorr27 = evaluator["IDCorr"](Z27.pt, Z27.eta)
                output[dataset]["IDCorr"].fill(IDCorr=ak.ravel(MuIDCorr27))

                Mu27TrgCorr = evaluator["Trg27Corr"](Z27.pt, Z27.eta)
                output[dataset]["Trg27Corr"].fill(Trg27Corr = ak.ravel(Mu27TrgCorr))

                puTrue27 = np.array(np.rint(events27.Pileup.nTrueInt), dtype=np.int8)
                puWeight27 = evaluator_pu["Collisions18_UltraLegacy_goldenJSON"].evaluate(puTrue27, 'nominal')
                output[dataset]["puCorr"].fill(puCorr=ak.ravel(puWeight27))

                #Combine weights for Mu27
                lepCorr_27 = MuIsoCorr27 * MuIDCorr27 * Mu27TrgCorr * puWeight27 * lumiWeight27 
                if ("DYJets" in name): 
                    pTCorrection27 = evaluator["pTCorr"](Z27.mass, Z27.pt)
                    lepCorr_27 = lepCorr_27 * pTCorrection27
                    
                output[dataset]["lepCorr"].fill(lepCorr=ak.ravel(lepCorr_27))

                
                #Split by sign
                Z27_OS = ak.flatten(Z27[pairs27['muon'].charge * pairs27['tau'].charge == -1], axis=1)
                Z27_SS = ak.flatten(Z27[pairs27['muon'].charge * pairs27['tau'].charge != -1], axis=1)
                lepCorr27_OS = ak.flatten(lepCorr_27[pairs27['muon'].charge + pairs27['tau'].charge == 0], axis=1)
                lepCorr27_SS = ak.flatten(lepCorr_27[pairs27['muon'].charge + pairs27['tau'].charge != 0], axis=1)

                
                #Fill
                output[dataset]["mass"].fill(mass=Z27_OS.mass, weight=lepCorr27_OS)
                output[dataset]["ss_mass"].fill(mass=Z27_SS.mass, weight=lepCorr27_SS)
                output[dataset]["pt"].fill(pt=Z27_OS.pt, weight=lepCorr27_OS)
                output[dataset]["ss_pt"].fill(pt=Z27_SS.pt, weight=lepCorr27_SS)
                output[dataset]["eta"].fill(eta=Z27_OS.eta, weight=lepCorr27_OS)
                output[dataset]["ss_eta"].fill(eta=Z27_SS.eta, weight=lepCorr27_SS)

            #If statement to make sure that that there are still events that pass Mu50
            if len(events50.genWeight) > 0:
                #Fill weights
                output[dataset]["lumiWeight"].fill(lumiWeight=ak.ravel(lumiWeight50))
    
                mu50 = makeVector(pairs50['muon'])      
                tau50 = makeVector(pairs50['tau'])  
    
                output[dataset]["muPt"].fill(pt=ak.ravel(mu50.pt), weight=ak.ravel(lumiWeight50))
                output[dataset]["tauPt"].fill(pt=ak.ravel(tau50.pt), weight=ak.ravel(lumiWeight50))
                
                Z50 = tau50.add(mu50)
                #Only take events between 0 and 150 GeV mass
                Z50 = Z50[(Z50.mass < 150)]
                lumiWeight50 = lumiWeight50[(Z50.mass < 150)]

                
                output[dataset]["zPt"].fill(pt= ak.ravel(Z50.pt), weight=ak.ravel(lumiWeight50))    
                MuIsoCorr50 = evaluator["IsoCorr"](Z50.pt, Z50.eta)
                output[dataset]["IsoCorr"].fill(isoCorr=ak.ravel(MuIsoCorr50))
                
                MuIDCorr50 = evaluator["IDCorr"](Z50.pt, Z50.eta)
                output[dataset]["IDCorr"].fill(IDCorr=ak.ravel(MuIDCorr50))
                 
                Mu50TrgCorr = evaluator["Trg50Corr"](Z50.pt, Z50.eta)
                output[dataset]["Trg50Corr"].fill(Trg50Corr= ak.ravel(Mu50TrgCorr))
                
                puTrue50 = np.array(np.rint(events50.Pileup.nTrueInt), dtype=np.int8)
                puWeight50 = evaluator_pu["Collisions18_UltraLegacy_goldenJSON"].evaluate(puTrue50, 'nominal')
                output[dataset]["puCorr"].fill(puCorr=ak.ravel(puWeight50))

                #Combine all weights
                lepCorr_50 = MuIsoCorr50 * MuIDCorr50 * Mu50TrgCorr * puWeight50 * lumiWeight50
                #Add pTCorrection if DYJets
                if ("DYJets" in name): 
                    pTCorrection50 = evaluator["pTCorr"](Z50.mass, Z50.pt)
                    lepCorr_50 = lepCorr_50 * pTCorrection50
    
                output[dataset]["lepCorr"].fill(lepCorr=ak.ravel(lepCorr_50))

                #Split by sign
                Z50_OS = ak.flatten(Z50[pairs50['muon'].charge * pairs50['tau'].charge == -1], axis=1)
                Z50_SS = ak.flatten(Z50[pairs50['muon'].charge * pairs50['tau'].charge != -1], axis=1)
                lepCorr50_OS = ak.flatten(lepCorr_50[pairs50['muon'].charge + pairs50['tau'].charge == 0], axis=1)
                lepCorr50_SS = ak.flatten(lepCorr_50[pairs50['muon'].charge + pairs50['tau'].charge != 0], axis=1)


                #Output histograms
                output[dataset]["mass"].fill(mass=Z50_OS.mass, weight=lepCorr50_OS)
                output[dataset]["ss_mass"].fill(mass=Z50_SS.mass, weight=lepCorr50_SS)
                output[dataset]["pt"].fill(pt=Z50_OS.pt, weight=lepCorr50_OS)
                output[dataset]["ss_pt"].fill(pt=Z50_SS.pt, weight=lepCorr50_SS)
                output[dataset]["eta"].fill(eta=Z50_OS.eta, weight=lepCorr50_OS)
                output[dataset]["ss_eta"].fill(eta=Z50_SS.eta, weight=lepCorr50_SS)
            return output 
        else: #If Data (XSection = 1)
            #Apply trigger
            pairs27 = pairs[trg27]
            pairs50 = pairs[trg50]

            #Separate by trigger
            events27 = events[ak.num(pairs27, axis=1) > 0]       
            pairs27 = pairs[ak.num(pairs27, axis=1) > 0] 

            events50 = events[ak.num(pairs50, axis=1) > 0]       
            pairs50 = pairs[ak.num(pairs50, axis=1) > 0] 

            output[dataset]["hcount"].fill(count= np.ones(len(events27))* 11 )
            output[dataset]["hcount"].fill(count= np.ones(len(events50))* 11 )
            

            
            mu27 = makeVector(pairs27['muon'])
            mu50 = makeVector(pairs50['muon'])      
            tau27 = makeVector(pairs27['tau'])
            tau50 = makeVector(pairs50['tau'])  

            output[dataset]["muPt"].fill(pt=ak.ravel(mu27.pt))
            output[dataset]["muPt"].fill(pt=ak.ravel(mu50.pt))
            output[dataset]["tauPt"].fill(pt=ak.ravel(tau27.pt))
            output[dataset]["tauPt"].fill(pt=ak.ravel(tau50.pt))

            output[dataset]["dr"].fill(dr=ak.ravel(tau27.delta_r(mu27)))
            output[dataset]["dr"].fill(dr=ak.ravel(tau50.delta_r(mu50)))

            #Create Z Vecs
            Z27 = tau27.add(mu27)
            Z50 = tau50.add(mu50)

            #Keep mass between 0 and 150
            Z27 = Z27[Z27.mass < 150]
            Z50 = Z50[Z50.mass < 150]

            #Split by sign
            Z27_OS = ak.flatten(Z27[pairs27['muon'].charge * pairs27['tau'].charge == -1], axis=1)
            Z27_SS = ak.flatten(Z27[pairs27['muon'].charge * pairs27['tau'].charge != -1], axis=1)

            Z50_OS = ak.flatten(Z50[pairs50['muon'].charge * pairs50['tau'].charge == -1], axis=1)
            Z50_SS = ak.flatten(Z50[pairs50['muon'].charge * pairs50['tau'].charge != -1], axis=1)

            #Output histograms
            output[dataset]["mass"].fill(mass=Z27_OS.mass)
            output[dataset]["mass"].fill(mass=Z50_OS.mass)
            
            output[dataset]["ss_mass"].fill(mass=Z27_SS.mass)
            output[dataset]["ss_mass"].fill(mass=Z50_SS.mass)
            
            output[dataset]["pt"].fill(pt=Z27_OS.pt)
            output[dataset]["pt"].fill(pt=Z50_OS.pt)
            
            output[dataset]["ss_pt"].fill(pt=Z27_SS.pt)
            output[dataset]["ss_pt"].fill(pt=Z50_SS.pt)

            output[dataset]["eta"].fill(eta=Z27_OS.eta)
            output[dataset]["eta"].fill(eta=Z50_OS.eta)
            
            output[dataset]["ss_eta"].fill(eta=Z27_SS.eta)
            output[dataset]["ss_eta"].fill(eta=Z50_SS.eta)
            return output 
        return output
    def postprocess(self, accumulator):
        return accumulator

#Dataset - changes from condor job script
dataset = 'Local'

#PATHS
mc_path = "root://cmsxrootd.hep.wisc.edu//store/user/emettner/Radion/Skimmed/2018/MC"
data_path = "root://cmsxrootd.hep.wisc.edu//store/user/emettner/Radion/Skimmed/2018/Data"
redirector = "root://cmsxrootd.hep.wisc.edu//store/user/gparida/HHbbtt/Full_Production_CMSSW_13_0_13_Nov24_23"
redirector2 = "root://cmsxrootd.hep.wisc.edu//store/user/cgalloni/HHbbtt/Full_Production_CMSSW_13_0_13_Nov24_23"


#MIXTURE DATASET FOR LOCAL TESTING:
localArr = np.concatenate((
[redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018A-UL2018_MiniAODv2_GT36-v2/231222_133142/0000/NANO_NANO_{i}.root" for i in range( 1 , 3 )],
[redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018B-UL2018_MiniAODv2_GT36-v2/231222_133202/0000/NANO_NANO_{i}.root" for i in range( 1 , 3 )],
[redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018C-UL2018_MiniAODv2_GT36-v3/231222_133222/0000/NANO_NANO_{i}.root" for i in range( 1 , 3 )],
[redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018D-UL2018_MiniAODv2_GT36-v2/231222_133242/0000/NANO_NANO_{i}.root" for i in range( 1 , 3 )],
[redirector+f"/2018/MC/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/231225_152342/0000/NANO_NANO_{i}.root" for i in range( 1 , 3 )],
[redirector+f"/2018/MC/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/240203_190026/0000/NANO_NANO_{i}.root" for i in range( 1 , 3 )],
[redirector+f"/2018/MC/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/231225_151948/0000/NANO_NANO_{i}.root" for i in range( 1 , 3 )],
[redirector+f"/2018/MC/DYJetsToLL_LHEFilterPtZ-250To400_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/DYJetsToLL_LHEFilterPtZ-250To400_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/240704_133710/0000/NANO_NANO_{i}.root" for i in range( 1 , 3 )], 
))

local_fileset = {
    "Local": localArr.tolist()
}


#UNSKIMMED FILE PATHS - CURRENTLY IN USE
DYJets_unskimmed = np.concatenate((
[redirector+f"/2018/MC/DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/231225_151736/0000/NANO_NANO_{i}.root" for i in range( 1 , 355 )],
[redirector+f"/2018/MC/DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/231225_151751/0000/NANO_NANO_{i}.root" for i in range( 1 , 550 )],
[redirector+f"/2018/MC/DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/231225_151806/0000/NANO_NANO_{i}.root" for i in range( 1 , 393 )],
[redirector+f"/2018/MC/DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/231225_151821/0000/NANO_NANO_{i}.root" for i in range( 1 , 196 )],
[redirector+f"/2018/MC/DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/231225_151836/0000/NANO_NANO_{i}.root" for i in range( 1 , 162 )],
[redirector+f"/2018/MC/DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/231225_151851/0000/NANO_NANO_{i}.root" for i in range( 1 , 168 )],
[redirector+f"/2018/MC/DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/231225_151905/0000/NANO_NANO_{i}.root" for i in range( 1 , 150 )],
[redirector+f"/2018/MC/DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/231225_151920/0000/NANO_NANO_{i}.root" for i in range( 1 , 68 )],
))

DYJetsPt_unskimmed = np.concatenate((
    [redirector+f"/2018/MC/DYJetsToLL_LHEFilterPtZ-0To50_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/DYJetsToLL_LHEFilterPtZ-0To50_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/240704_133625/0000/NANO_NANO_{i}.root" for i in range( 1 , 1000 )],
    [redirector+f"/2018/MC/DYJetsToLL_LHEFilterPtZ-0To50_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/DYJetsToLL_LHEFilterPtZ-0To50_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/240704_133625/0001/NANO_NANO_{i}.root" for i in range( 1000 , 2000 )],
    [redirector+f"/2018/MC/DYJetsToLL_LHEFilterPtZ-0To50_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/DYJetsToLL_LHEFilterPtZ-0To50_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/240704_133625/0002/NANO_NANO_{i}.root" for i in range( 2000 , 2060 )],
    [redirector+f"/2018/MC/DYJetsToLL_LHEFilterPtZ-100To250_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/DYJetsToLL_LHEFilterPtZ-100To250_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/240704_133649/0000/NANO_NANO_{i}.root" for i in range( 1 , 1000 )],
    [redirector+f"/2018/MC/DYJetsToLL_LHEFilterPtZ-100To250_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/DYJetsToLL_LHEFilterPtZ-100To250_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/240704_133649/0001/NANO_NANO_{i}.root" for i in range( 1000 , 1004 )],
    [redirector+f"/2018/MC/DYJetsToLL_LHEFilterPtZ-250To400_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/DYJetsToLL_LHEFilterPtZ-250To400_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/240704_133710/0000/NANO_NANO_{i}.root" for i in range( 1 , 400 )],
    [redirector+f"/2018/MC/DYJetsToLL_LHEFilterPtZ-400To650_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/DYJetsToLL_LHEFilterPtZ-400To650_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/240704_133735/0000/NANO_NANO_{i}.root" for i in range( 1 , 80 )],
    [redirector+f"/2018/MC/DYJetsToLL_LHEFilterPtZ-50To100_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/DYJetsToLL_LHEFilterPtZ-50To100_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/240704_133755/0000/NANO_NANO_{i}.root" for i in range( 1 , 1000 )],
    [redirector+f"/2018/MC/DYJetsToLL_LHEFilterPtZ-50To100_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/DYJetsToLL_LHEFilterPtZ-50To100_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/240704_133755/0001/NANO_NANO_{i}.root" for i in range( 1000 , 1376 )],
    [redirector+f"/2018/MC/DYJetsToLL_LHEFilterPtZ-650ToInf_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/DYJetsToLL_LHEFilterPtZ-650ToInf_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/240704_133818/0000/NANO_NANO_{i}.root" for i in range( 1 , 107 )]
))
WJets_unskimmed = np.concatenate((
    [redirector+f"/2018/MC/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8/231225_152652/0000/NANO_NANO_{i}.root" for i in range( 1 , 1000 )],
    [redirector+f"/2018/MC/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8/231225_152652/0001/NANO_NANO_{i}.root" for i in range( 1000 , 1116 )],
    [redirector+f"/2018/MC/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8/231225_152707/0000/NANO_NANO_{i}.root" for i in range( 1 , 149 )],
    [redirector+f"/2018/MC/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_ext1-v2_otherPart/231225_153158/0000/NANO_NANO_{i}.root" for i in range( 1 , 153 )],
    [redirector+f"/2018/MC/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/231225_151948/0000/NANO_NANO_{i}.root" for i in range( 1 , 1000 )],
    [redirector+f"/2018/MC/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/231225_151948/0001/NANO_NANO_{i}.root" for i in range( 1000 , 1253 )],
    [redirector+f"/2018/MC/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8/231225_152721/0000/NANO_NANO_{i}.root" for i in range( 1 , 95 )],
    [redirector+f"/2018/MC/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8_ext1-v2_otherPart/231225_153212/0000/NANO_NANO_{i}.root" for i in range( 1 , 354 )],
    [redirector+f"/2018/MC/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8/231225_152003/0000/NANO_NANO_{i}.root" for i in range( 1 , 187 )],
    [redirector+f"/2018/MC/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_ext1-v2_otherPart/231225_153115/0000/NANO_NANO_{i}.root" for i in range( 1 , 54 )],
    [redirector+f"/2018/MC/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8/231225_152017/0000/NANO_NANO_{i}.root" for i in range( 1 , 177 )],
    [redirector+f"/2018/MC/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8_ext1-v2_otherPart/231225_153129/0000/NANO_NANO_{i}.root" for i in range( 1 , 128 )],
    [redirector+f"/2018/MC/WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8/231225_151934/0000/NANO_NANO_{i}.root" for i in range( 1 , 1000 )],
    [redirector+f"/2018/MC/WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8/231225_151934/0001/NANO_NANO_{i}.root" for i in range( 1000 , 1418 )],
    [redirector+f"/2018/MC/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8/231225_152032/0000/NANO_NANO_{i}.root" for i in range( 1 , 161 )],
    [redirector+f"/2018/MC/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8_ext1-v2_otherPart/231225_153143/0000/NANO_NANO_{i}.root" for i in range( 1 , 120 )]
))

TT_unskimmed = np.concatenate((
    [redirector+f"/2018/MC/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/231225_152259/0000/NANO_NANO_{i}.root" for i in range( 1 , 1000 )],
    [redirector+f"/2018/MC/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/231225_152259/0001/NANO_NANO_{i}.root" for i in range( 1000 , 2000 )],
    [redirector+f"/2018/MC/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/231225_152259/0002/NANO_NANO_{i}.root" for i in range( 2000 , 3000 )],
    [redirector+f"/2018/MC/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/231225_152259/0003/NANO_NANO_{i}.root" for i in range( 3000 , 3070 )],
    [redirector+f"/2018/MC/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/231225_152244/0000/NANO_NANO_{i}.root" for i in range( 1 , 1000 )],
    [redirector+f"/2018/MC/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/231225_152244/0001/NANO_NANO_{i}.root" for i in range( 1000 , 2000 )],
    [redirector+f"/2018/MC/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/231225_152244/0002/NANO_NANO_{i}.root" for i in range( 2000 , 3000 )],
    [redirector+f"/2018/MC/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/231225_152244/0003/NANO_NANO_{i}.root" for i in range( 3000 , 4000 )],
    [redirector+f"/2018/MC/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/231225_152244/0004/NANO_NANO_{i}.root" for i in range( 4000 , 5000 )],
    [redirector+f"/2018/MC/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/231225_152244/0005/NANO_NANO_{i}.root" for i in range( 5000 , 6000 )],
    [redirector+f"/2018/MC/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/231225_152244/0006/NANO_NANO_{i}.root" for i in range( 6000 , 7000 )],
    [redirector+f"/2018/MC/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/231225_152244/0007/NANO_NANO_{i}.root" for i in range( 7000 , 7196 )],
    [redirector+f"/2018/MC/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/240203_190026/0000/NANO_NANO_{i}.root" for i in range( 1 , 1000 )],
    [redirector+f"/2018/MC/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/240203_190026/0001/NANO_NANO_{i}.root" for i in range( 1000 , 2000 )],
    [redirector+f"/2018/MC/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/240203_190026/0002/NANO_NANO_{i}.root" for i in range( 2000 , 3000 )],
    [redirector+f"/2018/MC/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/240203_190026/0003/NANO_NANO_{i}.root" for i in range( 3000 , 4000 )],
    [redirector+f"/2018/MC/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/240203_190026/0004/NANO_NANO_{i}.root" for i in range( 4000 , 5000 )],
    [redirector+f"/2018/MC/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/240203_190026/0005/NANO_NANO_{i}.root" for i in range( 5000 , 5010 )]
))

VV_unskimmed = np.concatenate((
    [redirector+f"/2018/MC/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8/231225_152313/0000/NANO_NANO_{i}.root" for i in range( 1 , 428 )],
    [redirector+f"/2018/MC/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/231225_152328/0000/NANO_NANO_{i}.root" for i in range( 1 , 1000 )],
    [redirector+f"/2018/MC/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/231225_152328/0001/NANO_NANO_{i}.root" for i in range( 1000 , 1915 )],
    [redirector+f"/2018/MC/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/231225_152342/0000/NANO_NANO_{i}.root" for i in range( 1 , 1000 )],
    [redirector+f"/2018/MC/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/231225_152342/0001/NANO_NANO_{i}.root" for i in range( 1000 , 2000 )],
    [redirector+f"/2018/MC/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/231225_152342/0002/NANO_NANO_{i}.root" for i in range( 2000 , 3000 )],
    [redirector+f"/2018/MC/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/231225_152342/0003/NANO_NANO_{i}.root" for i in range( 3000 , 3658 )],
    [redirector+f"/2018/MC/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/231225_152806/0000/NANO_NANO_{i}.root" for i in range( 1 , 141 )],
    [redirector+f"/2018/MC/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/231225_152821/0000/NANO_NANO_{i}.root" for i in range( 1 , 166 )],
    [redirector+f"/2018/MC/WWTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/WWTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/231225_152850/0000/NANO_NANO_{i}.root" for i in range( 1 , 741 )],
    [redirector+f"/2018/MC/WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/231225_152904/0000/NANO_NANO_{i}.root" for i in range( 1 , 142 )],
    [redirector+f"/2018/MC/WZTo1L3Nu_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/WZTo1L3Nu_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/231225_153046/0000/NANO_NANO_{i}.root" for i in range( 1 , 68 )],
    [redirector+f"/2018/MC/WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/231225_152919/0000/NANO_NANO_{i}.root" for i in range( 1 , 514 )],
    [redirector+f"/2018/MC/ZZTo2Nu2Q_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8/ZZTo2Nu2Q_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8/231225_153100/0000/NANO_NANO_{i}.root" for i in range( 1 , 49 )],
    [redirector+f"/2018/MC/ZZTo2Nu2Q_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8/ZZTo2Nu2Q_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8/231225_153100/0000/NANO_NANO_{i}.root" for i in range( 50 , 121 )],
    [redirector+f"/2018/MC/ZZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/ZZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/231225_152934/0000/NANO_NANO_{i}.root" for i in range( 1 , 598 )],
    [redirector+f"/2018/MC/ZZTo4L_TuneCP5_13TeV_powheg_pythia8/ZZTo4L_TuneCP5_13TeV_powheg_pythia8/231225_152835/0000/NANO_NANO_{i}.root" for i in range( 1 , 1000 )],
    [redirector+f"/2018/MC/ZZTo4L_TuneCP5_13TeV_powheg_pythia8/ZZTo4L_TuneCP5_13TeV_powheg_pythia8/231225_152835/0001/NANO_NANO_{i}.root" for i in range( 1000 , 1318 )]
))

Data_unskimmed = np.concatenate((
    [redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018A-UL2018_MiniAODv2_GT36-v2/231222_133142/0000/NANO_NANO_{i}.root" for i in range( 1 , 1000 )],
    [redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018A-UL2018_MiniAODv2_GT36-v2/231222_133142/0001/NANO_NANO_{i}.root" for i in range( 1000 , 2000 )],
    [redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018A-UL2018_MiniAODv2_GT36-v2/231222_133142/0002/NANO_NANO_{i}.root" for i in range( 2000 , 2963 )],
    [redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018B-UL2018_MiniAODv2_GT36-v2/231222_133202/0000/NANO_NANO_{i}.root" for i in range( 1 , 1000 )],
    [redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018B-UL2018_MiniAODv2_GT36-v2/231222_133202/0001/NANO_NANO_{i}.root" for i in range( 1000 , 1370 )],
    [redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018C-UL2018_MiniAODv2_GT36-v3/231222_133222/0000/NANO_NANO_{i}.root" for i in range( 1 , 1000 )],
    [redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018C-UL2018_MiniAODv2_GT36-v3/231222_133222/0001/NANO_NANO_{i}.root" for i in range( 1000 , 1297 )],
    [redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018D-UL2018_MiniAODv2_GT36-v2/231222_133242/0000/NANO_NANO_{i}.root" for i in range( 1 , 1000 )],
    [redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018D-UL2018_MiniAODv2_GT36-v2/231222_133242/0001/NANO_NANO_{i}.root" for i in range( 1000 , 2000 )],
    [redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018D-UL2018_MiniAODv2_GT36-v2/231222_133242/0002/NANO_NANO_{i}.root" for i in range( 2000 , 3000 )],
    [redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018D-UL2018_MiniAODv2_GT36-v2/231222_133242/0003/NANO_NANO_{i}.root" for i in range( 3000 , 4000 )],
    [redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018D-UL2018_MiniAODv2_GT36-v2/231222_133242/0004/NANO_NANO_{i}.root" for i in range( 4000 , 5000 )],
    [redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018D-UL2018_MiniAODv2_GT36-v2/231222_133242/0005/NANO_NANO_{i}.root" for i in range( 5000 , 5591 )],
))


#DICTIONARY FOR ALL FILEPATHS
DYJets_fileset = {
    "DYJets": DYJets_unskimmed.tolist(),
}

Data_fileset = {
    "Data": Data_unskimmed.tolist(),
}

TT_fileset = {
    "TT": TT_unskimmed.tolist(),
}

VV_fileset = {
    "VV": VV_unskimmed.tolist(),
}

WJets_fileset = {
    "WJets": WJets_unskimmed.tolist(),
}

#DASK JOB SPECIFICATIONS
MAX_WORKERS = 150
CHUNKSIZE = 60_000
MAX_CHUNKS = None

#EXTRACT WEIGHT SETS USING COFFEA
ext = extractor()
ext.add_weight_sets(["IDCorr NUM_LooseID_DEN_genTracks_pt_abseta ./RunBCDEF_SF_ID.root", "Trg50Corr Mu50_OR_TkMu50_PtEtaBins/pt_abseta_ratio ./Trigger_EfficienciesAndSF_RunBtoF.root", "Trg27Corr IsoMu24_OR_IsoTkMu24_PtEtaBins/pt_abseta_ratio ./Trigger_EfficienciesAndSF_RunBtoF.root", "IsoCorr NUM_LooseRelIso_DEN_LooseID_pt_abseta ./RunBCDEF_SF_ISO.root", "pTCorr Ratio2D ./zmm_2d_2018.root"])
ext.finalize()
evaluator = ext.make_evaluator()
evaluator_pu = correctionlib.CorrectionSet.from_file("./puWeights.json")
f = open('2018_weight.json')
sumGen = json.load(f)
print("Extracted weight sets")

#CREATE EXECUTOR
local_executor = processor.IterativeExecutor(status=True)

#CREATE RUNNER
print("Creating runner")
print("Using chunksize: {} and maxchunks {}".format(CHUNKSIZE,MAX_CHUNKS))
runner = processor.Runner(
    executor=local_executor,
    schema=NanoAODSchema,
    chunksize=CHUNKSIZE,
    maxchunks=MAX_CHUNKS,
    skipbadfiles=True,
    xrootdtimeout=300,
)
print("Running processor")
#RUN PROCESS
mt_results_local = runner(local_fileset, treename="Events", processor_instance=MyProcessor(),)

#OUTPUT TO FILE
outFile = uproot.recreate("boostedHTT_mt_2018_local_Local.input.root")
outFile["DYJets_met_1_13TeV/" + dataset + "_mass"] = mt_results_local[dataset]['mass'].to_numpy()
outFile["DYJets_met_1_13TeV/" + dataset + "_ss_mass"] = mt_results_local[dataset]['ss_mass'].to_numpy()
outFile["DYJets_met_1_13TeV/pt"] = mt_results_local[dataset]['pt'].to_numpy()
outFile["DYJets_met_1_13TeV/ss_pt"] = mt_results_local[dataset]['ss_pt'].to_numpy()
outFile["DYJets_met_1_13TeV/eta"] = mt_results_local[dataset]['eta'].to_numpy()
outFile["DYJets_met_1_13TeV/ss_eta"] = mt_results_local[dataset]['ss_eta'].to_numpy()
outFile["DYJets_met_1_13TeV/MuIDCorr"] = mt_results_local[dataset]['IDCorr'].to_numpy()
outFile["DYJets_met_1_13TeV/MuIsoCorr"] = mt_results_local[dataset]['IsoCorr'].to_numpy()
outFile["DYJets_met_1_13TeV/MuTrg27Corr"] = mt_results_local[dataset]['Trg27Corr'].to_numpy()
outFile["DYJets_met_1_13TeV/MuTrg50Corr"] = mt_results_local[dataset]['Trg50Corr'].to_numpy()
outFile["DYJets_met_1_13TeV/puCorr"] = mt_results_local[dataset]['puCorr'].to_numpy()
outFile["DYJets_met_1_13TeV/lepCorr"] = mt_results_local[dataset]['lepCorr'].to_numpy()
outFile["DYJets_met_1_13TeV/lumiWeight"] = mt_results_local[dataset]['lumiWeight'].to_numpy()
outFile["DYJets_met_1_13TeV/dr"] = mt_results_local[dataset]['dr'].to_numpy()
outFile["DYJets_met_1_13TeV/muPt"] = mt_results_local[dataset]['muPt'].to_numpy()
outFile["DYJets_met_1_13TeV/tauPt"] = mt_results_local[dataset]['tauPt'].to_numpy()
outFile["DYJets_met_1_13TeV/hcount"] = mt_results_local[dataset]['hcount'].to_numpy()
outFile["DYJets_met_1_13TeV/total"] = mt_results_local[dataset]['total'].to_numpy()
outFile["DYJets_met_1_13TeV/zPt"] = mt_results_local[dataset]['zPt'].to_numpy()
outFile["DYJets_met_1_13TeV/HiggsPt"] = mt_results_local[dataset]['HiggsPt'].to_numpy()
outFile["DYJets_met_1_13TeV/dr_test"] = mt_results_local[dataset]['dr_test'].to_numpy()
outFile["DYJets_met_1_13TeV/tmass"] = mt_results_local[dataset]['tmass'].to_numpy()
outFile.close()