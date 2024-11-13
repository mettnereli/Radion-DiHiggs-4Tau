#!/bin/bash
echo "Argument: $1 $2"
#arg 1: Path to file directory. Arg 2: Process ID
sed -i "s@filename = 'filename'@filename = '$1'@g" skimDownDiMu.py
sed -i "s@uproot.recreate('NANO_NANO_l.root')@uproot.recreate('NANO_NANO_${2}.root')@g" skimDownDiMu.py
python skimDownDiMu.py
echo "Ran skimDown"
rm skimDownDiMu.py
echo "DONE!"
exit $exitcode
