Instructions for running jobs (login.hep.wisc.edu and condor):
This instruction will be for the files in jobs/fullSet, but jobs/fullSetDiMu should be the same.

1. Go to your preferred working area (I have been using nfs_sratch)
2. Create your own "fullSet" and "fullSetDiMu" directories and move the files from this repo into them respectively
3. Create a dir  "condor/output" in both directories (line 13 in submit_local.jdl)
4. You must edit "submit_local.jdl":
  line 11: change input file paths to your respective directories (although, if you're on the right Madison cluster, you might just be able to use my paths without changing anything)
  (Note: all weighting files are also included in this repo, in Radion-DiHiggs-4Tau/NanoRun. You can pull this and point to there as well if my paths aren't available)
  line 14-16: modify this if your condor/output directory is different
5. "condor_submit submit_local.jdl"

Should submit 5 jobs, one for each process. Wait about a day. Then, you will have a file containing all histograms from each proccess. These files can be output for plotting or viewed on their from the root browser!

RUN INSTRUCTIONS FOR LOCAL TESTING: (CURRENTLY ONLY FOR DIUMUON CHANNEL, WORKING ON INTEGRATING MUTAU CHANNEL NOW!)

1. Go into fullSetDiMu or fullSet
2. cp LocalAnalysis.py ../../NanoRun #so it is in the same directory as all of the weight files
3. cd ..
4. bash sourceFile.sh
5. source .bashrc
5. cd ../NanoRun
6. bash ../jobs/container_start.sh
6. python3 LocalAnalysis.py

Will output a root file in NanoRun containing histograms.
  
