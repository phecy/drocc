#!/bin/bash

for num in $(seq 1 1 30)
do
    echo $num
    tar -cf data_batch_$num.tar -T data_batch_$num.txt
done
