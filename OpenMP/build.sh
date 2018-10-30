make
nb_threads=$1
echo "Nb of threads $nb_threads"
if [ "$#" -eq 1 ]; then
    export OMP_NUM_THREADS=$nb_threads
    echo "OMP_NUM_THREADS=$nb_threads"
fi