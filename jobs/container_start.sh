#!/usr/bin/env bash


if [ "$1" == "" ]; then
  export COFFEA_IMAGE=coffeateam/coffea-dask:0.7.22-py3.10-g7f049
else
  export COFFEA_IMAGE=$1
fi

EXTERNAL_BIND=${PWD} singularity exec -B ${PWD}:/srv -B /etc/condor -B /scratch -B /afs --pwd /srv \
  /cvmfs/unpacked.cern.ch/registry.hub.docker.com/${COFFEA_IMAGE} \
  /bin/bash --rcfile /srv/.bashrc
