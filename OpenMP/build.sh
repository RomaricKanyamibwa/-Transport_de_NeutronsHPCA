N=5000000
echo "-----------------------1st test for N=$N-----------------------"
echo "\n************************SEQ************************\n"
./../neutron-seq 1 $N
echo "\n************************OpenMP************************\n"
nb_threads=2
export OMP_NUM_THREADS=$nb_threads
#echo "Nb of threads $nb_threads"
./neutron-openmp 1 $N

nb_threads=4
export OMP_NUM_THREADS=$nb_threads
#echo "Nb of threads $nb_threads"
./neutron-openmp 1 $N

nb_threads=8
export OMP_NUM_THREADS=$nb_threads
#echo "Nb of threads $nb_threads"
./neutron-openmp 1 $N

nb_threads=16
export OMP_NUM_THREADS=$nb_threads
#echo "Nb of threads $nb_threads"
./neutron-openmp 1 $N


N=50000000
echo "-----------------------2nd test for N=$N-----------------------"
echo "\n************************SEQ************************\n"
./../neutron-seq 1 $N
echo "\n************************OpenMP************************\n"
nb_threads=2
export OMP_NUM_THREADS=$nb_threads
#echo "Nb of threads $nb_threads"
./neutron-openmp 1 $N

nb_threads=4
export OMP_NUM_THREADS=$nb_threads
#echo "Nb of threads $nb_threads"
./neutron-openmp 1 $N

nb_threads=8
export OMP_NUM_THREADS=$nb_threads
#echo "Nb of threads $nb_threads"
./neutron-openmp 1 $N

nb_threads=16
export OMP_NUM_THREADS=$nb_threads
#echo "Nb of threads $nb_threads"
./neutron-openmp 1 $N

N=500000000
echo "-----------------------3rd test for N=$N-----------------------"
echo "\n************************SEQ************************\n"
./../neutron-seq 1 $N
echo "\n************************OpenMP************************\n"
nb_threads=2
export OMP_NUM_THREADS=$nb_threads
#echo "Nb of threads $nb_threads"
./neutron-openmp 1 $N

nb_threads=4
export OMP_NUM_THREADS=$nb_threads
#echo "Nb of threads $nb_threads"
./neutron-openmp 1 $N

nb_threads=8
export OMP_NUM_THREADS=$nb_threads
#echo "Nb of threads $nb_threads"
./neutron-openmp 1 $N

nb_threads=16
export OMP_NUM_THREADS=$nb_threads
#echo "Nb of threads $nb_threads"
./neutron-openmp 1 $N
