#!/bin/bash
echo "Loading O2Physics environment"

eval `$(which alienv) -w /software/nstrangmann/alice/sw --no-refresh printenv O2Physics/latest-aliceo2-o2`
export JALIEN_TOKEN_CERT=/software/airflow/tokens/tokencert_10001.pem
export JALIEN_TOKEN_KEY=/software/airflow/tokens/tokenkey_10001.pem

# root -q -b -l "checkFT0Shadowing_Debug.C(780000,5)"
# root -q -b -l "checkFT0Shadowing_Debug.C(780000,50)"
# root -q -b -l "checkFT0Shadowing_Debug.C(780000,100)"
root -q -b -l "checkFT0Shadowing.C(780000)"
# root -q -b -l "checkFT0Shadowing_Debug.C(780000,300)"
# root -q -b -l "checkFT0Shadowing.C(767668)"