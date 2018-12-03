N=5000000

make

#seq
echo "\n************************SEQ************************\n"
./neutron-seq 1 $N

#openmp benchmarking
echo "\n************************OpenMP************************\n"
cd OpenMP

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

nb_threads=10
export OMP_NUM_THREADS=$nb_threads
#echo "Nb of threads $nb_threads"
./neutron-openmp 1 $N
cd ../

#Cuda benchmarking
echo "\n************************CUDA************************\n"
cd CUDA

./neutron-gpu 1 $N
cd ..

filename="perform_gnuplot_$N.txt"
echo "Changing perform_gnuplot.txt name to $filename"
mv perform_gnuplot.txt $filename
N=50000000

#seq
echo "\n************************SEQ************************\n"
./neutron-seq 1 $N

#openmp benchmarking
echo "\n************************OpenMP************************\n"
cd OpenMP

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

nb_threads=10
export OMP_NUM_THREADS=$nb_threads
#echo "Nb of threads $nb_threads"
./neutron-openmp 1 $N
cd ../

#Cuda benchmarking
echo "\n************************CUDA************************\n"
cd CUDA

./neutron-gpu 1 $N
cd ..


filename="perform_gnuplot_$N.txt"
echo "Changing perform_gnuplot.txt name to $filename"
mv perform_gnuplot.txt $filename
N=500000000

#seq
echo "\n************************SEQ************************\n"
./neutron-seq 1 $N

#openmp benchmarking
echo "\n************************OpenMP************************\n"
cd OpenMP

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

nb_threads=10
export OMP_NUM_THREADS=$nb_threads
#echo "Nb of threads $nb_threads"
./neutron-openmp 1 $N
cd ../

#Cuda benchmarking
echo "\n************************CUDA************************\n"
cd CUDA

./neutron-gpu 1 $N
cd ..
