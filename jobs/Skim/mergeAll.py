import ROOT
import subprocess
import os

data_path = "./" #Insert yours here
data_dir = ["Run2018A/0000", "Run2018A/0001", "Run2018A/0002", "Run2018B/0000", "Run2018B/0001", "Run2018C/0000", "Run2018C/0001", "Run2018D/0000", "Run2018D/0001", "Run2018D/0002", "Run2018D/0003", "Run2018D/0004", "Run2018D/0005",]


for path in data_dir:
	directory = data_path + data_dir
	filepath = data_path + data_dir + "/NANO_NANO_*.root"
	outpath = str(path[:-5]) + ".root"
	subprocess.run(["python3", "haddnano.py", outpath, filepath])
	os.system("mv " + outpat + " " + directory) 
