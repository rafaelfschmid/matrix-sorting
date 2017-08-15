#!/bin/bash
input=$1 #arquivos de entrada
time=$2  #caminho dos tempos

for block in 2 4 8 16 32 ; do
 	echo "rowsortingbi " $block
	./../scripts/exec.sh rowsortingbi${block}.exe $input > $time/rowsortingbi${block}.out 
done
