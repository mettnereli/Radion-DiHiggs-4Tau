import os
import numpy as np

data_dir = ["Run2018A/0000", "Run2018A/0001", "Run2018A/0002", "Run2018B/0000", "Run2018B/0001", "Run2018C/0000", "Run2018C/0001", "Run2018D/0000", "Run2018D/0001", "Run2018D/0002", "Run2018D/0003", "Run2018D/0004", "Run2018D/0005",]

redirector = "/hdfs/store/user/gparida/HHbbtt/Full_Production_CMSSW_13_0_13_Nov24_23"
redirector2 = "/hdfs/store/user/cgalloni/HHbbtt/Full_Production_CMSSW_13_0_13_Nov24_23"

data_orig = np.concatenate((
    [redirector2+"/2018/Data/SingleMu/SingleMuon/Run2018A-UL2018_MiniAODv2_GT36-v2/231222_133142/0000/NANO_NANO_*.root"],
    [redirector2+"/2018/Data/SingleMu/SingleMuon/Run2018A-UL2018_MiniAODv2_GT36-v2/231222_133142/0001/NANO_NANO_*.root"],
    [redirector2+"/2018/Data/SingleMu/SingleMuon/Run2018A-UL2018_MiniAODv2_GT36-v2/231222_133142/0002/NANO_NANO_*.root"],
    [redirector2+"/2018/Data/SingleMu/SingleMuon/Run2018B-UL2018_MiniAODv2_GT36-v2/231222_133202/0000/NANO_NANO_*.root"],
    [redirector2+"/2018/Data/SingleMu/SingleMuon/Run2018B-UL2018_MiniAODv2_GT36-v2/231222_133202/0001/NANO_NANO_*.root"],
    [redirector2+"/2018/Data/SingleMu/SingleMuon/Run2018C-UL2018_MiniAODv2_GT36-v3/231222_133222/0000/NANO_NANO_*.root"],
    [redirector2+"/2018/Data/SingleMu/SingleMuon/Run2018C-UL2018_MiniAODv2_GT36-v3/231222_133222/0001/NANO_NANO_*.root"],
    [redirector2+"/2018/Data/SingleMu/SingleMuon/Run2018D-UL2018_MiniAODv2_GT36-v2/231222_133242/0000/NANO_NANO_*.root"],
    [redirector2+"/2018/Data/SingleMu/SingleMuon/Run2018D-UL2018_MiniAODv2_GT36-v2/231222_133242/0001/NANO_NANO_*.root"],
    [redirector2+"/2018/Data/SingleMu/SingleMuon/Run2018D-UL2018_MiniAODv2_GT36-v2/231222_133242/0002/NANO_NANO_*.root"],
    [redirector2+"/2018/Data/SingleMu/SingleMuon/Run2018D-UL2018_MiniAODv2_GT36-v2/231222_133242/0003/NANO_NANO_*.root"],
    [redirector2+"/2018/Data/SingleMu/SingleMuon/Run2018D-UL2018_MiniAODv2_GT36-v2/231222_133242/0004/NANO_NANO_*.root"],
    [redirector2+"/2018/Data/SingleMu/SingleMuon/Run2018D-UL2018_MiniAODv2_GT36-v2/231222_133242/0005/NANO_NANO_*.root"],
)).tolist()

data_path = "/hdfs/store/user/emettner/Radion/Skimmed/DiMu/2018/Data/"

for (path, rootd) in zip(data_dir, data_orig):
    if "/000" in path: back = path[:-5]
    else: back = path
    print("Background: ", back)

    os.chdir(data_path + path)
    os.system("mkdir condor")
    os.chdir("./condor")
    os.system("mkdir output")
    os.chdir("..")
    os.system("rm skimDownDiMu.py")
    os.system("cp /nfs_scratch/emettner/jobs/Skim/skimDownDiMu.py .")
    os.system("cp /nfs_scratch/emettner/jobs/Skim/run_skimmer.sh .")
    os.system("cp /nfs_scratch/emettner/jobs/Skim/submit_skimmer.jdl .")
    with open("submit_skimmer.jdl", "a") as file:
        file.write("queue filename matching files " + rootd)
    os.system("condor_submit submit_skimmer.jdl")
    
