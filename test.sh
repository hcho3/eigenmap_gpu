#!/bin/bash
FILE=lanczos_performance.log
echo "-------------------------------------------" >> $FILE
echo "               GPU Lanczos                 " >> $FILE
echo "-------------------------------------------" >> $FILE
echo "Processing KPIS001.jpg..." >> $FILE
./eigenmap KPIS001.mat 3 12 10 50 >> $FILE
echo "\nProcessing nCPM4.jpg..." >> $FILE
./eigenmap nCPM4.mat 3 48 10 50 >> $FILE
echo "\nProcessing capillary_skeletal_muscle.jpg..." >> $FILE
./eigenmap capillary_skeletal_muscle.mat 3 50 10 50 >> $FILE
echo "\nProcessing capillary_retina.jpg..." >> $FILE
./eigenmap capillary_retina.mat 3 199 10 50 >> $FILE
echo "\nProcessing nCPM9.jpg..." >> $FILE
./eigenmap nCPM9.mat 3 250 10 50 >> $FILE
echo "\nProcessing nCPM9_large.jpg..." >> $FILE
./eigenmap nCPM9_large.mat 3 250 10 50 >> $FILE
echo "-------------------------------------------" >> $FILE
echo "               GPU magma_dsyevdx           " >> $FILE
echo "-------------------------------------------" >> $FILE
echo "Processing KPIS001.jpg..." >> $FILE
./eigenmap_legacy KPIS001.mat 3 10 50 >> $FILE
echo "\nProcessing nCPM4.jpg..." >> $FILE
./eigenmap_legacy nCPM4.mat 3 10 50 >> $FILE
echo "\nProcessing capillary_skeletal_muscle.jpg..." >> $FILE
./eigenmap_legacy capillary_skeletal_muscle.mat 3 10 50 >> $FILE
echo "\nProcessing capillary_retina.jpg..." >> $FILE
./eigenmap_legacy capillary_retina.mat 3 10 50 >> $FILE
echo "\nProcessing nCPM9.jpg..." >> $FILE
./eigenmap_legacy nCPM9.mat 3 10 50 >> $FILE
