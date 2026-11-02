#!/bin/bash
set -euo pipefail

TrainNum=${1:-}

# Validate TrainNum: must be exactly six digits
if ! [[ $TrainNum =~ ^[0-9]{6}$ ]]; then
  echo "Error: TrainNum must be a six-digit integer (e.g. 535345). Got: '${TrainNum}'" >&2
  exit 1
fi

echo "Creating directory for train $TrainNum"

mkdir -p "Data/Train_$TrainNum"
touch "Data/Train_$TrainNum/HYPaths.txt"

# Clear contents (including hidden files) if any
find "Data/Train_$TrainNum" -mindepth 1 ! -path "Data/Train_$TrainNum/HYPaths.txt" -exec rm -rf -- {} +

# Read comma-separated paths from file into array (handles files without trailing newline)
paths_content=$(cat "Data/Train_$TrainNum/HYPaths.txt")
IFS=',' read -r -a paths <<< "$paths_content"

echo "Found ${#paths[@]} paths to process"

echo "Loading O2Physics environment"

eval `$(which alienv) -w /software/nstrangmann/alice/sw --no-refresh printenv O2Physics/latest-aliceo2-o2`
export JALIEN_TOKEN_CERT=/software/airflow/tokens/tokencert_10001.pem
export JALIEN_TOKEN_KEY=/software/airflow/tokens/tokenkey_10001.pem

# Loop over the array
for i in "${!paths[@]}"; do
    path="${paths[$i]}"
    echo "Processing path $((i+1))/${#paths[@]}: $path"

    # Use a unique filename for each download to avoid overwriting
    temp_file="Data/Train_${TrainNum}/temp_${i}.xml"
    
    echo "Downloading ${path}/analysis.xml to $temp_file"
    
    # Try the alien_cp command and check if it succeeds
    if alien_cp "${path}/analysis.xml" "file://$temp_file"; then
        echo "Download successful"
        
        # Check if the file actually exists and has content
        if [[ -f "$temp_file" && -s "$temp_file" ]]; then
            echo "File exists and has content. Extracting run number..."
            
            # Extract the run number - handle grep failure gracefully
            lfn_line=$(grep '" lfn="' "$temp_file" | head -1 || true)
            
            if [[ -n "$lfn_line" ]]; then
                runNum=$(echo "$lfn_line" | sed 's/.*" lfn="//; s/".*//' | cut -d'/' -f6)
                if [[ -n "$runNum" ]]; then
                    echo "Extracted run number: $runNum"
                    mkdir -p "Data/Train_${TrainNum}/Run_${runNum}"

                    # Download AnalysisResults.root only if it exists; otherwise skip
                    if alien_ls "${path}/AnalysisResults.root" >/dev/null 2>&1; then
                        if alien_cp "${path}/AnalysisResults.root" "file://Data/Train_${TrainNum}/Run_${runNum}/AnalysisResults.root"; then
                            echo "Downloaded AnalysisResults.root"
                        else
                            echo "Warning: Failed to download ${path}/AnalysisResults.root (skipping)"
                        fi
                    else
                        echo "Warning: ${path}/AnalysisResults.root does not exist (skipping)"
                    fi
                else
                    echo "Warning: Could not extract run number from lfn line: $lfn_line"
                fi
            else
                echo "Warning: No lfn attribute found in $temp_file"
                # Let's see what's actually in the file
                echo "File contents preview:"
                head -5 "$temp_file"
            fi
        else
            echo "Error: Downloaded file is empty or doesn't exist: $temp_file"
        fi
    else
        echo "Error: Failed to download ${path}/analysis.xml"
        # Continue with next iteration instead of exiting
        continue
    fi
    
    echo "---"
done

find "Data/Train_${TrainNum}" -maxdepth 1 -type f -name 'temp_*.xml' -delete

echo "Processing complete"