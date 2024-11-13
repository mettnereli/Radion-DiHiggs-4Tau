Instructions for runnings skimmer:


LOCAL (FOR TESTING):
1. Enter singulary shell (see readme in "jobs")
2.Replace line 39 with the filepath to the file.
3. python3 skimDownDiMu.py

Will output a singular skimmed file.






CONDOR (FOR FULL SKIMMING OF DATA):
1. ENTER directoryRunner.py and make the following changes:
Change line 25 for the data_path to be whatever directory you want to store the output in
Change lines 38, 39, and 40 to the path of each files (in this repo will work!)
2. Create directories in data_path for each entry at line 4, ie Run2018A/0000, Run2018A/0001, . . . Run2018D/0005
3. python directoryRunner.py
Note: there is a chance that when submitting this, the modifications to the submit file (line 42) will not have fully transferred by the time line 43 executes. If this is the case, comment out line 43 and run the script, then run again but comment out 41/42 and uncomment 43 to submit all the jobs.

This will submit one job per file, so 10k jobs total! Don't waste any submissions if not necessary!





MERGING THE ROOT FILES (AFTER SKIM):
1. go into your new directory containing the root files. Make sure your environment has ROOT (I just cmsenv)
2. python haddnano.py output.root /your/path/to/skimmed/files/NANO_NANO_*.root

Which will output a merged file for all of the files in the directory. Repeat for all of the directories (I just create a quick bash script that executes all of them and runs in the background). 
NOTE: Hdfs may not allow writing directly to the hdfs area. In this case, Create a temporary directory in nfs_scratch, and mv the file to hdfs when it completes hadding (or build that into your script).






