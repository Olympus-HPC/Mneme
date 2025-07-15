#!/bin/bash

rm -rf sorted*

file_21="localassm_extend_7-21.dat"
file_33="localassm_extend_1-33.dat"
file_55="localassm_extend_7-55.dat"
file_77="localassm_extend_9-77.dat"
kmer_sizes=(21 33 55 77)
i=0
kmer_size=${kmer_sizes[$i]}
for curr_file in $file_21 $file_33 $file_55 $file_77
do
    echo "running test for Kmer size: $kmer_size"
    ./main ../locassm_data/$curr_file $kmer_size ../test-out.dat 1> ${kmer_size}_result.log
    sort ../test-out.dat >> sorted_new
    sort ../locassm_data/"res_$curr_file" >> sorted_res
    difference=$(diff sorted_new sorted_res)

    if [ -z "$difference" ];
    then 
        echo "Test for Kmer size: $kmer_size PASSED!"
    else
        echo "Test for Kmer size: $kmer_size FAILED!"
    fi

    ((i=i+1))

    kmer_size=${kmer_sizes[$i]}
done
