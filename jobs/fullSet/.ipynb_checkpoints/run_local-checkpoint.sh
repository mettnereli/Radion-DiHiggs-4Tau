#!/bin/bash
#arg 1: Process name
echo "Argument: $1"
echo "Running LocalAnalysis"
sed -i "s@dataset = 'Local'@dataset = '$1'@g" LocalAnalysis.py
sed -i "s@mt_results_local = runner(local_fileset,@mt_results_local = runner($1_fileset,@g" LocalAnalysis.py
sed -i "s@boostedHTT_mt_2018_local_Local.input.root@boostedHTT_mt_2018_local_$1.input.root@g" LocalAnalysis.py
python LocalAnalysis.py
echo "Ran LocalAnalysis"
rm LocalAnalysis.py
echo "DONE!"
exit $exitcode
