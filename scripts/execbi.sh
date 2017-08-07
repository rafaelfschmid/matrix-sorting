#!/bin/bash
input=$1 #arquivos de entrada
time=$2  #caminho dos tempos

for block in 2 4 8 16 32 ; do
 	./../scripts/exec.sh blocksortingbi$block.exe $input > $time/blocksortingbi$block.out 
done

