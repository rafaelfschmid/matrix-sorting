
times=$1

for i in 2 4 8 16 32 64 128 256 512 1024; do
	./parser.exe $times/blocksortinguni$i.out $times/00blocksortinguni$i.out
done

for i in 2 4 8 16 32 ; do
	./parser.exe $times/blocksortingbi$i.out $times/00blocksortingbi$i.out
done


