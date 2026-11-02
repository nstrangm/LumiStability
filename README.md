First adjust the used O2Physics paths in the bath scripts to your system

1. bash downloadTrain.sh <trainnum>
2. Put output directories in the created HYPaths.txt file
3. bash prepareTrainOutput.sh <trainnum>
4. bash checkFT0Shadowing.sh (adjust trainnum in the bash script first)
5. python3 plotStability.py --t=B,SLD5,SLD50,SLD100,L100,SLD300 --dataset="LHC26 (pp)" --outdir=pp_26 --input=Train_768530,Train_786327