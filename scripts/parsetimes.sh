dir=$1

#./parser.exe $dir/fixbitonic.time $dir/00fixbitonic.time
./utils/parser.exe $dir/fixpass.time $dir/00fixpass.time
#./parser.exe $dir/fixpassdiff.time $dir/00fixpassdiff.time
./utils/parser.exe $dir/fixcub.time $dir/00fixcub.time
#./parser.exe $dir/fixmerge.time $dir/00fixmerge.time
./utils/parser.exe $dir/fixmergemgpu.time $dir/00fixmergemgpu.time
#./parser.exe $dir/fixoddeven.time $dir/00fixoddeven.time
#./parser.exe $dir/fixquick.time $dir/00fixquick.time
./utils/parser.exe $dir/fixthrust.time $dir/00fixthrust.time
./utils/parser.exe $dir/mergeseg.time $dir/00mergeseg.time
./utils/parser.exe $dir/radixseg.time $dir/00radixseg.time
#./parser.exe $dir/bitonicseg.time $dir/00bitonicseg.time

