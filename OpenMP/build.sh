make
nb_threads=$1
echo "Nb of threads $nb_threads"
if [ $# -gt 0 ]; then
    export OMP_NUM_THREADS=$nb_threads
    echo "OMP_NUM_THREADS=$nb_threads"
else
	echo "usage: ./build [NUM_THREADS]"
fi


if [ $# -eq 1 ]; then
	echo "---------- N=5*10**8----------\n">> performance.txt
	echo "************ neutron-seq ************\n">> performance.txt
	../neutron-seq >> performance.txt
	echo "./neutron-seq"
	echo "\n*********** neutron-openmp ***********\n">> performance.txt
	echo "OMP_NUM_THREADS=$nb_threads \n" >>performance.txt
    ./neutron-openmp >> performance.txt
    echo "./neutron-openmp"
else
	N=$2
	echo "---------- N=$N---------\n">> performance.txt
	echo "************ neutron-seq ************\n">> performance.txt
	../neutron-seq 1 $N >> performance.txt
	echo "./neutron-seq 1 $N"
	echo "\n*********** neutron-openmp ***********">> performance.txt
	echo "OMP_NUM_THREADS=$nb_threads \n" >>performance.txt
	./neutron-openmp 1 $N >> performance.txt
	echo "./neutron-openmp 1 $N"
fi
echo
echo "\n\n">> performance.txt
