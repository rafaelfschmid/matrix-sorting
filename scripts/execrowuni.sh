#!/bin/bash
input=$1 #arquivos de entrada
time=$2  #caminho dos tempos

for block in 2 4 8 16 32 64 128 256 512 1024; do
	echo "rowsortinguni " $block
	./../scripts/exec.sh rowsortinguni$block.exe $input > $time/rowsortinguni$block.out 
done

