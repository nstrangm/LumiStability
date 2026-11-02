#!/bin/bash
echo "Loading O2Physics environment"

eval `$(which alienv) -w /software/nstrangmann/alice/sw --no-refresh printenv O2Physics/latest-aliceo2-o2`
export JALIEN_TOKEN_CERT=/software/airflow/tokens/tokencert_10001.pem
export JALIEN_TOKEN_KEY=/software/airflow/tokens/tokenkey_10001.pem

root -q -b -l "prepareTrainOutput.C($1)"