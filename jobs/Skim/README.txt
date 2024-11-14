Instructions for runnings skimmer:



CONDOR (FOR FULL SKIMMING OF DATA):
1. ENTER directoryRunner.py and make the following changes:
Change line 25:
	data_path is whatever directory you want to store the output in.
	There will be lots of files, so using the /hdfs/store/user/USERNAME/..... pathway is best
Change lines 38, 39, and 40 to the path of each respective file (if following guide, /nfs_scratch/USERNAME/skim/.... 
2. Wherever you chose your datapath to be, create directories in that path matching each directory in"data_dir" at line 4. So  Run2018A/0000, Run2018A/0001, . . . Run2018D/0005
3. Make sure your proxy exists and is active:
	voms-proxy-init -vomses /etc/vomses -voms cms -rfc -valid 144:00 -bits 2048 -debug 
3. python directoryRunner.py
This script should enter each directory specified in your data_path, create a condor/output directory, copy all the necessary files there, and then submit jobs for each unskimmed file in the corresponding original hdfs directory. 

Note: there is a chance that when submitting this, the modifications to the submit file (line 42) will not have fully transferred by the time line 43 executes. If this is the case, comment out line 43 and run the script, then run again but comment out 41/42 and uncomment 43 to submit all the jobs.

This will submit one job per file, so 10k jobs total! Don't waste any submissions if not necessary!

If there is an accidental job submission, use condor_rm (PROC_ID) or just condor_rm USERNAME to remove all of the jobs you are running.



MERGING THE ROOT FILES (AFTER SKIM):
1. go into your new directory containing the root files. Make sure your environment has ROOT (I just cmsenv)
2. python haddnano.py output.root /your/path/to/skimmed/files/NANO_NANO_*.root

Which will output a merged file for all of the files in the directory. Repeat for all of the directories (I just create a quick bash script that executes all of them and runs in the background). 
NOTE: Hdfs may not allow writing directly to the hdfs area. In this case, Create a temporary directory in nfs_scratch, and mv the file to hdfs when it completes hadding (or build that into your script).






