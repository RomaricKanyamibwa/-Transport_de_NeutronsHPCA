#!/bin/bash

echo "starting ..."

#Narray=( 5000000 50000000 )
#OpenMP=( 2 4 6 8 16 32 42 64 88 )
k=1

for N in 5000000 50000000 500000000 #"${Narray[@]}"
do
   : 
   # do gpu,seq,openmp and hybride computations
   echo "----------------------Test $k for N=$N-----------------------"
   echo "\n\n----------------------Test $k for N=$N-----------------------\n\n" >> perform.txt
   #seq
   echo "\n************************SEQ************************\n"
   echo "\n************************SEQ************************\n" >> perform.txt
   ./neutron-seq 1 $N

   #openmp benchmarking
   echo "\n************************OpenMP************************\n"
   echo "\n************************OpenMP************************\n" >> perform.txt
   cd OpenMP
   for nb_threads in 2 4 6 8 16 32 42 64 88 #"${OpenMP[@]}"
   do
       : 
       export OMP_NUM_THREADS=$nb_threads
       echo "Nb of threads $nb_threads"
       ./neutron-openmp 1 $N
   done
   cd ../

   #Cuda benchmarking
   echo "\n************************CUDA************************\n"
   echo "\n************************CUDA************************\n" >> perform.txt
   cd CUDA

   ./neutron-gpu 1 $N
   cd ..

   #Hybride benchmarking
   echo "\n************************Hybride************************\n"
   echo "\n************************Hybride************************\n" >> perform.txt
   cd Hybride
   for nb_threads in 2 4 6 8 16 32 42 64 88 #"${OpenMP[@]}"
   do
       : 
       export OMP_NUM_THREADS=$nb_threads
       echo "Nb of threads $nb_threads"
       ./neutron-hybride 1 $N
   done
   cd ..

   k=$((k+1))
done


