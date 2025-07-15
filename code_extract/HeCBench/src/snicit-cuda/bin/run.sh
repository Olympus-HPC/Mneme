#!/bin/bash
log=tab4_fig11.txt

printf "Log File - " > $log
date >> $log

echo "Running SNICIT on benchmark A..."
info=`./beyond -k A | grep -oP '(?<=SNICIT info:).*'`
echo == SNICIT on A ==, $info >> $log
# ////////////////////////////////////////////////////////////////


echo "Running SNICIT on benchmark B..."
info=`./beyond -k B | grep -oP '(?<=SNICIT info:).*'`
echo == SNICIT on B ==, $info >> $log
# ////////////////////////////////////////////////////////////////////


echo "Running SNICIT on benchmark C..."
info=`./beyond -k C | grep -oP '(?<=SNICIT info:).*'`
echo == SNICIT on C ==, $info >> $log
# ///////////////////////////////////////////////////////////////////


echo "Running SNICIT on benchmark D..."
info=`./beyond -k D | grep -oP '(?<=SNICIT info:).*'`
echo == SNICIT on D ==, $info >> $log

cat $log
