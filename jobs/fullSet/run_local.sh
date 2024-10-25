#!/bin/bash
#arg 1: Process name
echo "Argument: $1"
echo "Running LocalSkim"
sed -i "s@dataset = 'DYJets'@dataset = '$1'@g" LocalSkim.py
sed -i "s@mt_results_local = runner(DYJets_fileset,@mt_results_local = runner($1_fileset,@g" LocalSkim.py
sed -i "s@boostedHTT_mt_2018_local_DYJets.input.root@boostedHTT_mt_2018_local_$1.input.root@g" LocalSkim.py
python LocalSkim.py
echo "Ran LocalSkim"
rm LocalSkim.py
echo "DONE!"
exit $exitcode
