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
         if "70To100" in name: return 140
         elif "100To200" in name: return 139.2
         elif "200To400" in name: return 38.4
         elif "400To600" in name: return 5.174
         elif "600To800" in name: return 1.258
         elif "800To1200" in name: return 0.5598
         elif "1200To2500" in name: return 0.1305
         elif "2500ToInf" in name: return 0.002997
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
        elif "100To200" in name: return sumGen['DYJetsToLL_M-50_HT-100to200']
        elif "200To400" in name: return sumGen['DYJetsToLL_M-50_HT-200to400']
        elif "400To600" in name: return sumGen['DYJetsToLL_M-50_HT-400to600']
        elif "600To800" in name: return sumGen['DYJetsToLL_M-50_HT-600to800']
        elif "800To1200" in name: return sumGen['DYJetsToLL_M-50_HT-800to1200']
        elif "1200To2500" in name: return sumGen['DYJetsToLL_M-50_HT-1200to2500']
        elif "2500ToInf" in name: return sumGen['DYJetsToLL_M-50_HT-2500toInf']
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
            "mass": Hist(hist.axis.Regular(12, 60, 120, flow=False, name="mass", label=dataset), storage=hist.storage.Weight()),
            "ss_mass": Hist(hist.axis.Regular(12, 60, 120, flow=False, name="mass", label=dataset), storage=hist.storage.Weight()),
            "pt": Hist(hist.axis.Regular(60, 150, 600, flow=False, name="pt", label=dataset), storage=hist.storage.Weight()),
            "ss_pt": Hist(hist.axis.Regular(60, 150, 600, flow=False, name="pt", label=dataset), storage=hist.storage.Weight()),
            "eta": Hist(hist.axis.Regular(40, -4, 4, flow=False, name="eta", label=dataset), storage=hist.storage.Weight()),
            "ss_eta": Hist(hist.axis.Regular(40, -4, 4, flow=False, name="eta", label=dataset), storage=hist.storage.Weight()),
            "lumiWeight": Hist(hist.axis.Regular(40, -5, 5, flow=False, name="lumiWeight", label=dataset), storage=hist.storage.Weight()),
            "IsoCorr": Hist(hist.axis.Regular(40, 0, 2, flow=False, name="isoCorr", label=dataset), storage=hist.storage.Weight()),
            "IDCorr": Hist(hist.axis.Regular(40, 0, 2, flow=False, name="IDCorr", label=dataset), storage=hist.storage.Weight()),
            "Trg27Corr": Hist(hist.axis.Regular(40, 0, 2, flow=False, name="Trg27Corr", label=dataset), storage=hist.storage.Weight()),
            "Trg50Corr": Hist(hist.axis.Regular(40, 0, 2, flow=False, name="Trg50Corr", label=dataset), storage=hist.storage.Weight()),
            "puCorr": Hist(hist.axis.Regular(40, 0, 2, flow=False, name="puCorr", label=dataset), storage=hist.storage.Weight()),
            "lepCorr": Hist(hist.axis.Regular(100, -2, 2, flow=False, name="lepCorr", label=dataset), storage=hist.storage.Weight()),
            "muPt": Hist(hist.axis.Regular(50, 0, 400, flow=False, name="pt", label=dataset), storage=hist.storage.Weight()),
            "subMuPt": Hist(hist.axis.Regular(50, 0, 400, flow=False, name="pt", label=dataset), storage=hist.storage.Weight()),
            "hcount": Hist(hist.axis.Regular(13, 0, 13, flow=False, name="count", label=dataset), storage=hist.storage.Weight()),
            "total": Hist(hist.axis.Regular(12, 0, 12, flow=False, name="count", label=dataset), storage=hist.storage.Weight()),
            "tmass": Hist(hist.axis.Regular(50, 0, 50, flow=False, name="mass", label=dataset), storage=hist.storage.Weight()),
            "dr": Hist(hist.axis.Regular(40, 0, 1, flow=False, name="dr", label=dataset), storage=hist.storage.Weight()),
            "zPt": Hist(hist.axis.Regular(100, 0, 400, flow=False, name="pt", label=dataset), storage=hist.storage.Weight()),
            "HiggsPt": Hist(hist.axis.Regular(100, 200, 450, flow=False, name="pt", label=dataset), storage=hist.storage.Weight()),
            "MET": Hist(hist.axis.Regular(60, 0, 400, flow=False, name="pt", label=dataset), storage=hist.storage.Weight()),
            "numEle_pre": Hist(hist.axis.Regular(11, 0, 10, flow=False, name="num", label=dataset)),
            "numEle_post": Hist(hist.axis.Regular(11, 0, 10, flow=False, name="num", label=dataset)),
            "numMuon_pre": Hist(hist.axis.Regular(11, 0, 10, flow=False, name="num", label=dataset)),
            "numMuon_post": Hist(hist.axis.Regular(11, 0, 10, flow=False, name="num", label=dataset)),
            "numbJet": Hist(hist.axis.Regular(11, 0, 10, flow=False, name="num", label=dataset)),
        }      

        #Fill histograms (starting with total number of events)
        output[dataset]["total"].fill(count= np.ones(len(events)))
        output[dataset]["hcount"].fill(count= np.ones(len(events)))
        
        #Get name
        name = str(events.metadata["filename"])
        print(name)
        #Get XS Value
        XSection = weightCalc(name)

        #Get sum of Gen Weights
        sumOfGenWeights = sumGenCalc(name)

        #Get total weights
        totalWeight = totalEventCalc(name)

        #Trigger, muon selection, remaining cuts
        events = events[ak.num(events.Muon) > 1]
        events = events[events.HLT.Mu50]

        #ak.combinations - creates pairs from all muons in events, names them with fields 'i0' and 'i1'
        dimuon = ak.combinations(events.Muon, 2, fields=['i0', 'i1']) 
        MuID = (dimuon['i0'].tightId) & (np.abs(dimuon['i0'].dz) < 0.2) & (np.abs(dimuon['i0'].dxy) < 0.045)
        SubMuID = (dimuon['i1'].tightId) & (np.abs(dimuon['i1'].dz) < 0.2) & (np.abs(dimuon['i1'].dxy) < 0.045)
        dr = dimuon['i0'].delta_r(dimuon['i1'])
        cut = ((np.abs(dimuon['i0'].eta) < 2.4)
              & (MuID)
              & (dimuon['i0'].pt > 52)
              & (dimuon['i1'].pt > 10)
              & (np.abs(dimuon['i1'].eta) < 2.4)
              & (SubMuID)
              & (dr > .1)
              & (dr < .8))

        #"dimuon" array contains all dimuon pairs. Cut is applied to it
        dimuon = dimuon[cut]
        #If the cut removed all possible dimuon pairs in an event, also remove that event from the event array
        # [ event, event, eventwithnopairs] -> [event, event]
        events = events[(ak.num(dimuon, axis=-1) > 0)]
        #If dimuon has an empty event "[[stuff], [stuff], []]" just get rid of it ->  "[[stuff], [stuff]]"
        dimuon = dimuon[(ak.num(dimuon, axis=-1) > 0)]
        #This process is repeated throughout
        
        output[dataset]["hcount"].fill(count= np.ones(len(events))* 2)


        #Extra Electron veto
        electron = events.Electron
        output[dataset]["numEle_pre"].fill(num=ak.num(events.Electron))
        ele_cut = (electron.pt >= 15) & (np.abs(electron.eta) <= 2.5) & electron.mvaIso_WPL
        dimuon = dimuon[(ak.any(ele_cut, axis=-1) == False)]
        events = events[(ak.num(dimuon, axis=-1) > 0)]
        dimuon = dimuon[(ak.num(dimuon, axis=-1) > 0)]
        output[dataset]["numEle_post"].fill(num= ak.num(events.Electron))
        output[dataset]["hcount"].fill(count= np.ones(len(events))* 3)

        
        #Extra muon veto
        output[dataset]["numMuon_pre"].fill(num=ak.num(events.Muon))
        i0_pass = (dimuon['i0'].pfRelIso04_all < .3) & (dimuon['i0'].pt > 10) & (dimuon['i0'].tightId) #if leading mu passes cut
        i1_pass = (dimuon['i1'].pfRelIso04_all < .3) & (dimuon['i1'].pt > 10) & (dimuon['i1'].tightId) #if subleading mu passes cut
        #check if all muons pass cut
        extraMuon = events.Muon[((events.Muon.pfRelIso04_all < .3) & (events.Muon.pt > 10) & (events.Muon.tightId))] 
        overallVeto = (ak.num(extraMuon, axis=-1) < 3)
        Extra2Veto = i0_pass & i1_pass & (ak.num(extraMuon, axis=-1) == 2)
        Extrai0Veto = i0_pass & (ak.num(extraMuon, axis=-1) == 1)
        Extrai1Veto = i1_pass & (ak.num(extraMuon, axis=-1) == 1)
        dimuon = dimuon[overallVeto | Extra2Veto | Extrai0Veto | Extrai1Veto]
        events = events[(ak.num(dimuon, axis=-1) > 0)]
        dimuon = dimuon[(ak.num(dimuon, axis=-1) > 0)]
        output[dataset]["numMuon_post"].fill(num=ak.num(events.Muon))
        output[dataset]["hcount"].fill(count= np.ones(len(events))* 4)

        
    
        #Jet Vetos - HT
       # goodJets= events.Jet[(events.Jet.jetId >= 1) & (events.Jet.pt > 30) & (np.abs(events.Jet.eta) < 3.0)]
       # HT = ak.sum(goodJets.pt, axis=-1)
       # dimuon = dimuon[(HT > 200)]
       # events = events[(ak.num(dimuon, axis=-1) > 0)]
       # dimuon = dimuon[(ak.num(dimuon, axis=-1) > 0)]
       # output[dataset]["hcount"].fill(count= np.ones(len(events))* 5)

        
        ##!!!!!!!!!!!!!!! change btagdeepflavb for each year https://btv-wiki.docs.cern.ch/ScaleFactors/UL2018/
        #bJet veto
        bJets = (events.Jet.btagDeepFlavB > .7100) & (events.Jet.jetId >= 2) & (events.Jet.pt > 30) & (np.abs(events.Jet.eta) < 2.4)
        output[dataset]["numbJet"].fill(num=ak.num(events.Jet[bJets]))
        dimuon = dimuon[ak.any(bJets, axis=-1) == False]
        events = events[(ak.num(dimuon, axis=-1) > 0)]
        dimuon = dimuon[(ak.num(dimuon, axis=-1) > 0)]
        output[dataset]["hcount"].fill(count= np.ones(len(events))* 6)


        ##Define two 4Vectors
        i0 = makeVector(dimuon['i0'])
        i1 = makeVector(dimuon['i1'])

        #If for whatever reason all events have been cut already, stop wasting process time and just return
        if len(events.MET) == 0: return output
        if len(events.MET.pt) == 0: return output

        #TMass Cut
        tmass = np.sqrt(np.square(i0.pt + events.MET.pt) - np.square(i0.px + events.MET.pt * np.cos(events.MET.phi)) - np.square(i0.py + events.MET.pt * np.sin(events.MET.phi)))
        tmass_cut = tmass < 40
        dimuon = dimuon[tmass_cut]
        events = events[(ak.num(dimuon, axis=-1) > 0)]
        dimuon = dimuon[(ak.num(dimuon, axis=-1) > 0)]
        output[dataset]["hcount"].fill(count= np.ones(len(events))* 7)

        #Z pt cut
        i0 = makeVector(dimuon['i0'])
        i1 = makeVector(dimuon['i1'])
        ZVec = i0.add(i1)  
        ptCut = (ZVec.pt > 200)
        dimuon = dimuon[ptCut]
        events = events[(ak.num(dimuon, axis=-1) > 0)]
        dimuon = dimuon[(ak.num(dimuon, axis=-1) > 0)]
        output[dataset]["hcount"].fill(count= np.ones(len(events))* 8) 

        #Similar check again
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

        #Higgs pt cut
        i0 = makeVector(dimuon['i0'])
        i1 = makeVector(dimuon['i1'])
        ZVec = i0.add(i1)        
        Higgs = ZVec.add(MetVec)
        HiggsCut = (Higgs.pt > 250) 
        dimuon = dimuon[HiggsCut]
        events = events[ak.num(dimuon, axis=1) > 0]       
        dimuon = dimuon[ak.num(dimuon, axis=1) > 0] 
        output[dataset]["hcount"].fill(count= np.ones(len(events))* 9)
        

        
        #Create Z Candidate and select one Z Candidate per event
        i0 = makeVector(dimuon['i0'])
        i1 = makeVector(dimuon['i1'])
        ZVec = i0.add(i1)
        #If there are multiple Z Candidates per event:
        if ak.any(ak.num(ZVec, axis=-1) > 1, axis=-1):
            #Select whichever Z Candidate has closest mass to 91.187, and make sure dimuon keeps that too
            ZVec = ZVec[ak.argmin(np.absolute(ZVec.mass - 91.187), axis=-1, keepdims=True)]
            dimuon = dimuon[ak.argmin(np.absolute(ZVec.mass - 91.187), axis=-1, keepdims=True)]
            events = events[ak.num(dimuon, axis=1) > 0]
            #check multiplicty
            #zmass > 15 
        output[dataset]["hcount"].fill(count= np.ones(len(events))* 10)
        
        #Make new vectors
        i0 = makeVector(dimuon['i0'])
        i1 = makeVector(dimuon['i1'])   
        Z = i0.add(i1)
        
        ##Get weights if not Data (if data, XSection = 1)
        if XSection != 1:             
            print("Not data!")
            luminosity2018 = 59830.
            #Luminosity weight - XSection / sumOfGenWeights, multiplied by the genWeight for each event
            lumiWeight = np.multiply(((XSection * luminosity2018) / sumOfGenWeights), events.genWeight)

            #Fill weight into histogram (note: ak.ravel removes extra nesting so it's just a 1D array of all events for fill)         
            #MuIsoCorr = evaluator["IsoCorr"](Z.pt, Z.eta)
            #output[dataset]["IsoCorr"].fill(isoCorr=ak.ravel(MuIsoCorr))
            
            MuIDCorr = evaluator["IDCorr"](Z.pt, Z.eta)
            output[dataset]["IDCorr"].fill(IDCorr=ak.ravel(MuIDCorr))
            
            Mu50TrgCorr = evaluator["Trg50Corr"](Z.pt, Z.eta)
            output[dataset]["Trg50Corr"].fill(Trg50Corr= ak.ravel(Mu50TrgCorr))
            
            puTrue = np.array(np.rint(events.Pileup.nTrueInt), dtype=np.int8)
            puWeight = evaluator_pu["Collisions18_UltraLegacy_goldenJSON"].evaluate(puTrue, 'nominal')
            output[dataset]["puCorr"].fill(puCorr=ak.ravel(puWeight))

            #Combine all weighting arrays together
            lepCorr = MuIDCorr * Mu50TrgCorr * lumiWeight * puWeight

            #Also at pTCorrection if DYJets bkg
            if ("DYJets" in name): 
                pTCorrection = evaluator["pTCorr"](Z.mass, Z.pt)
                lepCorr = lepCorr * pTCorrection

            output[dataset]["lepCorr"].fill(lepCorr=(ak.ravel(lepCorr)))
        else: #This is if Data
            print("Data!")
            lepCorr = ak.ones_like(Z.mass)
             
        output[dataset]["muPt"].fill(pt=ak.ravel(i0.pt), weight=ak.ravel(lepCorr))
        output[dataset]["subMuPt"].fill(pt=ak.ravel(i1.pt), weight=ak.ravel(lepCorr))

        output[dataset]["dr"].fill(dr=ak.ravel(i0.delta_r(i1)), weight=ak.ravel(lepCorr))
        
        #Similar check again
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
        Higgs = Z.add(MetVec)
        output[dataset]["HiggsPt"].fill(pt= ak.ravel(Higgs.pt), weight=ak.ravel(lepCorr)) 
        output[dataset]["MET"].fill(pt= ak.ravel(events.MET.pt), weight=ak.ravel(lepCorr)) 
        tmass = np.sqrt(np.square(i0.pt + events.MET.pt) - np.square(i0.px + events.MET.pt * np.cos(events.MET.phi)) - np.square(i0.py + events.MET.pt * np.sin(events.MET.phi)))
        output[dataset]["tmass"].fill(mass=ak.ravel(tmass), weight=ak.ravel(lepCorr))

        
        #Split into OS and SS regions
        Z_OS = Z[dimuon['i0'].charge + dimuon['i1'].charge == 0]
        Z_SS = Z[dimuon['i0'].charge + dimuon['i1'].charge != 0]
        lepCorr_OS = lepCorr[dimuon['i0'].charge + dimuon['i1'].charge == 0]
        lepCorr_SS = lepCorr[dimuon['i0'].charge + dimuon['i1'].charge != 0]

        output[dataset]["hcount"].fill(count= np.ones(len(Z_OS.mass))* 11)

        
        #Select only between 60 and 120 GeV
        Z_OS = (Z_OS[(Z_OS.mass > 60) & (Z_OS.mass < 120)])
        Z_SS = (Z_SS[(Z_SS.mass > 60) & (Z_SS.mass < 120)])
        lepCorr_OS = (lepCorr_OS[(Z_OS.mass > 60) & (Z_OS.mass < 120)])
        lepCorr_SS = (lepCorr_SS[(Z_SS.mass > 60) & (Z_SS.mass < 120)])

        output[dataset]["hcount"].fill(count= np.ones(len(Z_OS.mass))* 12)
        #Fill all remaining histograms
        output[dataset]["mass"].fill(mass=ak.ravel(Z_OS.mass), weight=ak.ravel(lepCorr_OS))
        
        output[dataset]["ss_mass"].fill(mass=ak.ravel(Z_SS.mass), weight=ak.ravel(lepCorr_SS))

        output[dataset]["pt"].fill(pt=ak.ravel(Z_OS.pt), weight=ak.ravel(lepCorr_OS))
        
        output[dataset]["ss_pt"].fill(pt=ak.ravel(Z_SS.pt), weight=ak.ravel(lepCorr_SS))

        output[dataset]["eta"].fill(eta=ak.ravel(Z_OS.eta), weight=ak.ravel(lepCorr_OS))
        
        output[dataset]["ss_eta"].fill(eta=ak.ravel(Z_SS.eta), weight=ak.ravel(lepCorr_SS))

        return output 
    def postprocess(self, accumulator):
        return accumulator


#dataset is set to 'Local' by default but is modified in condor run file
dataset = 'Local'

#All paths
mc_path = "root://cmsxrootd.hep.wisc.edu//store/user/emettner/Radion/Skimmed/DiMu/2018/MC"
data_path = "root://cmsxrootd.hep.wisc.edu//store/user/emettner/Radion/Skimmed/DiMu/2018/Data"
redirector = "root://cmsxrootd.hep.wisc.edu//store/user/gparida/HHbbtt/Full_Production_CMSSW_13_0_13_Nov24_23"
redirector2 = "root://cmsxrootd.hep.wisc.edu//store/user/cgalloni/HHbbtt/Full_Production_CMSSW_13_0_13_Nov24_23"

hdfspath2 = "/hdfs/store/user/cgalloni/HHbbtt/Full_Production_CMSSW_13_0_13_Nov24_23"
hdfspath = "/hdfs/store/user/gparida/HHbbtt/Full_Production_CMSSW_13_0_13_Nov24_23"

#MIXTURE DATASET FOR LOCAL TESTING:
localArr = np.concatenate((
[redirector+f"/2018/MC/DYJetsToLL_LHEFilterPtZ-250To400_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/DYJetsToLL_LHEFilterPtZ-250To400_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/240704_133710/0000/NANO_NANO_{i}.root" for i in range( 1 , 3 )], 
[redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018A-UL2018_MiniAODv2_GT36-v2/231222_133142/0000/NANO_NANO_{i}.root" for i in range( 1 , 3 )], 
[redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018B-UL2018_MiniAODv2_GT36-v2/231222_133202/0000/NANO_NANO_{i}.root" for i in range( 1 , 3 )],
[redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018C-UL2018_MiniAODv2_GT36-v3/231222_133222/0000/NANO_NANO_{i}.root" for i in range( 1 , 3 )],
[redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018D-UL2018_MiniAODv2_GT36-v2/231222_133242/0000/NANO_NANO_{i}.root" for i in range( 1 , 3 )],
[redirector+f"/2018/MC/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/231225_152342/0000/NANO_NANO_{i}.root" for i in range( 1 , 3 )],
[redirector+f"/2018/MC/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/240203_190026/0000/NANO_NANO_{i}.root" for i in range( 1 , 3 )],
[redirector+f"/2018/MC/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/231225_151948/0000/NANO_NANO_{i}.root" for i in range( 1 , 3 )],
))

local_fileset = {
    "Local": localArr.tolist()
}

#SKIMMED ARRAYS
DYJetsArr_skimmed = np.concatenate((
[mc_path+f"/DY70To100/DY70To100.root"],
[mc_path+f"/DY100To200/DY100To200.root"],
[mc_path+f"/DY200To400/DY200To400.root"],
[mc_path+f"/DY400To600/DY400To600.root"],
[mc_path+f"/DY600To800/DY600To800.root"],
[mc_path+f"/DY800To1200/DY800To1200.root"],
[mc_path+f"/DY1200To2500/DY1200To2500.root"],
[mc_path+f"/DY2500ToInf/DY2500ToInf.root"],
))


DataArr_skimmed = np.concatenate((
[data_path+f"/Run2018A/0000/NANO_NANO_{i}.root" for i in range(0, 999)],
[data_path+f"/Run2018A/0001/NANO_NANO_{i}.root" for i in range(0, 1000)],
[data_path+f"/Run2018A/0002/NANO_NANO_{i}.root" for i in range(0, 963)],
[data_path+f"/Run2018B/0000/NANO_NANO_{i}.root" for i in range(0, 999)],
[data_path+f"/Run2018B/0001/NANO_NANO_{i}.root" for i in range(0, 370)],
[data_path+f"/Run2018C/0000/NANO_NANO_{i}.root" for i in range(0, 999)],
[data_path+f"/Run2018C/0001/NANO_NANO_{i}.root" for i in range(0, 297)],
[data_path+f"/Run2018D/0000/NANO_NANO_{i}.root" for i in range(0, 999)],
[data_path+f"/Run2018D/0001/NANO_NANO_{i}.root" for i in range(0, 1000)],
[data_path+f"/Run2018D/0002/NANO_NANO_{i}.root" for i in range(0, 1000)],
[data_path+f"/Run2018D/0003/NANO_NANO_{i}.root" for i in range(0, 1000)],
[data_path+f"/Run2018D/0004/NANO_NANO_{i}.root" for i in range(0, 1000)],
[data_path+f"/Run2018D/0005/NANO_NANO_{i}.root" for i in range(0, 591)],
))



DataArr_addedSkimmedSplit = np.concatenate((
[data_path+f"/Run2018A/0000/Run2018A_{i}.root" for i in range(0, 10)],
[data_path+f"/Run2018A/0001/Run2018A_{i}.root" for i in range(0, 10)],
[data_path+f"/Run2018A/0002/Run2018A_{i}.root" for i in range(0, 10)],
[data_path+f"/Run2018B/0000/Run2018B_{i}.root" for i in range(0, 10)],
[data_path+f"/Run2018B/0001/Run2018B_{i}.root" for i in range(0, 10)],
[data_path+f"/Run2018C/0000/Run2018C_{i}.root" for i in range(0, 10)],
[data_path+f"/Run2018C/0001/Run2018C_{i}.root" for i in range(0, 10)],
[data_path+f"/Run2018D/0000/Run2018D_{i}.root" for i in range(0, 10)],
[data_path+f"/Run2018D/0001/Run2018D_{i}.root" for i in range(0, 10)],
[data_path+f"/Run2018D/0002/Run2018D_{i}.root" for i in range(0, 10)],
[data_path+f"/Run2018D/0003/Run2018D_{i}.root" for i in range(0, 10)],
[data_path+f"/Run2018D/0004/Run2018D_{i}.root" for i in range(0, 10)],
[data_path+f"/Run2018D/0005/Run2018D_{i}.root" for i in range(0, 10)],
))

DataArr_addedSkimmed = np.concatenate((
[data_path+f"/Run2018A/0000/Run2018A.root"],
[data_path+f"/Run2018A/0001/Run2018A.root"],
[data_path+f"/Run2018A/0002/Run2018A.root"],
[data_path+f"/Run2018B/0000/Run2018B.root"],
[data_path+f"/Run2018B/0001/Run2018B.root"],
[data_path+f"/Run2018C/0000/Run2018C.root"],
[data_path+f"/Run2018C/0001/Run2018C.root"],
[data_path+f"/Run2018D/0000/Run2018D.root"],
[data_path+f"/Run2018D/0001/Run2018D.root"],
[data_path+f"/Run2018D/0002/Run2018D.root"],
[data_path+f"/Run2018D/0003/Run2018D.root"],
[data_path+f"/Run2018D/0004/Run2018D.root"],
[data_path+f"/Run2018D/0005/Run2018D.root"],
))

TTArr_skimmed = np.concatenate((
[mc_path+"/TTTo2L2Nu/0000/TTTo2L2Nu.root"],
[mc_path+"/TTTo2L2Nu/0001/TTTo2L2Nu.root"],
[mc_path+"/TTTo2L2Nu/0002/TTTo2L2Nu.root"],
[mc_path+"/TTTo2L2Nu/0003/TTTo2L2Nu.root"],
[mc_path+"/TTToHadronic/0000/TTToHadronic.root"],
[mc_path+"/TTToHadronic/0001/TTToHadronic.root"],
[mc_path+"/TTToHadronic/0002/TTToHadronic.root"],
[mc_path+"/TTToHadronic/0003/TTToHadronic.root"],
[mc_path+"/TTToHadronic/0004/TTToHadronic.root"],
[mc_path+"/TTToHadronic/0005/TTToHadronic.root"],
[mc_path+"/TTToHadronic/0006/TTToHadronic.root"],
[mc_path+"/TTToHadronic/0007/TTToHadronic.root"],
[mc_path+"/TTToSemiLeptonic/0000/TTToSemiLeptonic.root"],
[mc_path+"/TTToSemiLeptonic/0001/TTToSemiLeptonic.root"],
[mc_path+"/TTToSemiLeptonic/0002/TTToSemiLeptonic.root"],
[mc_path+"/TTToSemiLeptonic/0003/TTToSemiLeptonic.root"],
[mc_path+"/TTToSemiLeptonic/0004/TTToSemiLeptonic.root"],
[mc_path+"/TTToSemiLeptonic/0005/TTToSemiLeptonic.root"],
))

VVArr_skimmed = np.concatenate((
[mc_path+"/ST_tW_top_5f_inclusiveDecays/ST_tW_top_5f_inclusiveDecays.root"],
[mc_path+"/WZTo2Q2L/WZTo2Q2L.root"],
[mc_path+"/ST_s-channel_4f_leptonDecays/ST_s-channel_4f_leptonDecays.root"],
[mc_path+"/ZZTo2Nu2Q/ZZTo2Nu2Q.root"],
[mc_path+"/ST_t-channel_antitop_4f_InclusiveDecays/0000/ST_t-channel_antitop_4f_InclusiveDecays.root"],
[mc_path+"/ST_t-channel_antitop_4f_InclusiveDecays/0001/ST_t-channel_antitop_4f_InclusiveDecays.root"],
[mc_path+"/WWTo1L1Nu2Q/WWTo1L1Nu2Q.root"],
[mc_path+"/ZZTo2Q2L/ZZTo2Q2L.root"],
[mc_path+"/ST_t-channel_top_4f_InclusiveDecays/0000/ST_t-channel_top_4f_InclusiveDecays.root"],
[mc_path+"/ST_t-channel_top_4f_InclusiveDecays/0001/ST_t-channel_top_4f_InclusiveDecays.root"],
[mc_path+"/ST_t-channel_top_4f_InclusiveDecays/0002/ST_t-channel_top_4f_InclusiveDecays.root"],
[mc_path+"/ST_t-channel_top_4f_InclusiveDecays/0003/ST_t-channel_top_4f_InclusiveDecays.root"],
[mc_path+"/WZTo1L1Nu2Q/WZTo1L1Nu2Q.root"],
[mc_path+"/ZZTo4L/0000/ZZTo4L.root"],
[mc_path+"/ZZTo4L/0001/ZZTo4L.root"], 
[mc_path+"/ST_tW_antitop_5f_inclusiveDecays/ST_tW_antitop_5f_inclusiveDecays.root"],
[mc_path+"/WZTo1L3Nu/WZTo1L3Nu.root"],
))

WJetsArr_skimmed = np.concatenate((
[mc_path+f"/WJets1200To2500/0000/WJets1200To2500_{i}.root" for i in range(0, 10)],
[mc_path+f"/WJets1200To2500/0001/WJets1200To2500_{i}.root" for i in range(0, 10)],
[mc_path+"/WJets70To100/0000/WJets70To100.root"],
[mc_path+"/WJets70To100/0001/WJets70To100.root"],
[mc_path+"/WJets200To400/0000/WJets200To400.root"],
[mc_path+"/WJets200To400/0001/WJets200To400.root"],
[mc_path+"/WJets800To1200/0000/WJets800To1200.root"],
[mc_path+"/WJets800To1200/0001/WJets800To1200.root"],
[mc_path+f"/WJets2500ToInf/0000/WJets2500ToInf_{i}.root" for i in range(0, 10)],
[mc_path+f"/WJets2500ToInf/0001/WJets2500ToInf_{i}.root" for i in range(0, 10)],
[mc_path+"/WJets400To600/0000/WJets400To600.root"],
[mc_path+"/WJets400To600/0001/WJets400To600.root"],
[mc_path+"/WJets100To200/0000/WJets100To200.root"],
[mc_path+"/WJets100To200/0001/WJets100To200.root"],
[mc_path+"/WJets600To800/0000/WJets600To800.root"],
[mc_path+"/WJets600To800/0001/WJets600To800.root"],
))
 

#UNSKIMMED ARRAY PATHS (WHAT I AM CURRENTLY USING)
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

Arr2018A_unskimmed = np.concatenate((
    [redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018A-UL2018_MiniAODv2_GT36-v2/231222_133142/0000/NANO_NANO_{i}.root" for i in range( 1 , 1000 )],
    [redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018A-UL2018_MiniAODv2_GT36-v2/231222_133142/0001/NANO_NANO_{i}.root" for i in range( 1000 , 2000 )],
    [redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018A-UL2018_MiniAODv2_GT36-v2/231222_133142/0002/NANO_NANO_{i}.root" for i in range( 2000 , 2963 )],))

Arr2018B_unskimmed = np.concatenate((
    [redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018B-UL2018_MiniAODv2_GT36-v2/231222_133202/0000/NANO_NANO_{i}.root" for i in range( 1 , 1000 )],
    [redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018B-UL2018_MiniAODv2_GT36-v2/231222_133202/0001/NANO_NANO_{i}.root" for i in range( 1000 , 1370 )],))

Arr2018C_unskimmed = np.concatenate((
    [redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018C-UL2018_MiniAODv2_GT36-v3/231222_133222/0000/NANO_NANO_{i}.root" for i in range( 1 , 1000 )],
    [redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018C-UL2018_MiniAODv2_GT36-v3/231222_133222/0001/NANO_NANO_{i}.root" for i in range( 1000 , 1297 )],))

Arr2018D_unskimmed = np.concatenate((
    [redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018D-UL2018_MiniAODv2_GT36-v2/231222_133242/0000/NANO_NANO_{i}.root" for i in range( 1 , 1000 )],
    [redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018D-UL2018_MiniAODv2_GT36-v2/231222_133242/0001/NANO_NANO_{i}.root" for i in range( 1000 , 2000 )],
    [redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018D-UL2018_MiniAODv2_GT36-v2/231222_133242/0002/NANO_NANO_{i}.root" for i in range( 2000 , 3000 )],
    [redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018D-UL2018_MiniAODv2_GT36-v2/231222_133242/0003/NANO_NANO_{i}.root" for i in range( 3000 , 4000 )],
    [redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018D-UL2018_MiniAODv2_GT36-v2/231222_133242/0004/NANO_NANO_{i}.root" for i in range( 4000 , 5000 )],
    [redirector2+f"/2018/Data/SingleMu/SingleMuon/Run2018D-UL2018_MiniAODv2_GT36-v2/231222_133242/0005/NANO_NANO_{i}.root" for i in range( 5000 , 5591 )],
))


#DICTIONARIES FOR EACH FILESET
DYJets_fileset = {
    "DYJets": DYJetsArr_skimmed.tolist(),
}

Data_fileset = {
    "Data": DataArr_addedSkimmed.tolist(),
}

TT_fileset = {
    "TT": TTArr_skimmed.tolist(),
}

VV_fileset = {
    "VV": VVArr_skimmed.tolist(),
}

WJets_fileset = {
    "WJets": WJetsArr_skimmed.tolist(),
}

#DASK JOB SUBMISSION SPECIFICATIONS
MAX_WORKERS = 50
CHUNKSIZE = 250_000
MAX_CHUNKS = None

#CREATE EXTRACTOR AND ADD WEIGHT SETS
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

#RUN EVENTS
mt_results_local = runner(local_fileset, treename="Events", processor_instance=MyProcessor(),)

#OUTPUT EVERYTHING TO NUMPY HISTOGRAM FILES (dataset SPECIFIED BY JOB SCRIPT)
outFile = uproot.recreate("boostedHTT_mt_2018_local_Local.input.root")
outFile["DYJets_met_1_13TeV/" + dataset + "_mass"] = mt_results_local[dataset]['mass'].to_numpy()
outFile["DYJets_met_1_13TeV/" + dataset + "_ss_mass"] = mt_results_local[dataset]['ss_mass'].to_numpy()
outFile["DYJets_met_1_13TeV/pt"] = mt_results_local[dataset]['pt'].to_numpy()
outFile["DYJets_met_1_13TeV/ss_pt"] = mt_results_local[dataset]['ss_pt'].to_numpy()
outFile["DYJets_met_1_13TeV/eta"] = mt_results_local[dataset]['eta'].to_numpy()
outFile["DYJets_met_1_13TeV/ss_eta"] = mt_results_local[dataset]['ss_eta'].to_numpy()
outFile["DYJets_met_1_13TeV/MuIDCorr"] = mt_results_local[dataset]['IDCorr'].to_numpy()
outFile["DYJets_met_1_13TeV/MuIsoCorr"] = mt_results_local[dataset]['IsoCorr'].to_numpy()
outFile["DYJets_met_1_13TeV/puCorr"] = mt_results_local[dataset]['puCorr'].to_numpy()
outFile["DYJets_met_1_13TeV/lepCorr"] = mt_results_local[dataset]['lepCorr'].to_numpy()
outFile["DYJets_met_1_13TeV/lumiWeight"] = mt_results_local[dataset]['lumiWeight'].to_numpy()
outFile["DYJets_met_1_13TeV/dr"] = mt_results_local[dataset]['dr'].to_numpy()
outFile["DYJets_met_1_13TeV/subMuPt"] = mt_results_local[dataset]['subMuPt'].to_numpy()
outFile["DYJets_met_1_13TeV/muPt"] = mt_results_local[dataset]['muPt'].to_numpy()
outFile["DYJets_met_1_13TeV/hcount"] = mt_results_local[dataset]['hcount'].to_numpy()
outFile["DYJets_met_1_13TeV/total"] = mt_results_local[dataset]['total'].to_numpy()
outFile["DYJets_met_1_13TeV/HiggsPt"] = mt_results_local[dataset]['HiggsPt'].to_numpy()
outFile["DYJets_met_1_13TeV/tmass"] = mt_results_local[dataset]['tmass'].to_numpy()
outFile["DYJets_met_1_13TeV/MET"] = mt_results_local[dataset]['MET'].to_numpy()
outFile.close()
