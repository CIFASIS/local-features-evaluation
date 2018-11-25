#!/bin/bash

if [ -d "time_results" ]; then
   echo "There are already some measurements done, please clean time_results/ ot run this script"
   exit
fi

mkdir "time_results"
mkdir "time_results/detectors"
mkdir "time_results/descriptors"
mkdir "time_results/matchers"

#declare -a dets=("FAST" "ORB" "AGAST" "BRISK" "GFTT" "BAFT" "AKAZE" "LOCKY")
declare -a dets=("FAST")
declare -a descs=("BRIEF" "BRISK" "ORB" "LDB" "FREAK" "LATCH" "BAFT" "AKAZE")
declare -a matchers=("BFM" "GMS" "FLANN")

for det in "${dets[@]}"
do
   for desc in "${descs[@]}"
   do
      for match in "${matchers[@]}"
      do
         ./Features "$det" "$desc" "$match"
         mv "tiempos_match_""$det""_""$desc""_""$match"".txt" "time_results/matchers/"
         if [ ! "$det" == "GFTT" -a ! "$det" == "LOCKY" ]; then
            ./Features "$det" "$desc" "$match" "anms"
            mv "tiempos_match_""$det""_ANMS_""$desc""_""$match"".txt" "time_results/matchers/"
         fi
      done
      mv "tiempos_desc_""$det""_""$desc"".txt" "time_results/descriptors/"
      mv "tiempos_desc_""$det""_ANMS_""$desc"".txt" "time_results/descriptors/"
   done
   mv "tiempos_det_""$det"".txt" "time_results/detectors/"
   mv "tiempos_det_""$det""_ANMS.txt" "time_results/detectors/"
done
