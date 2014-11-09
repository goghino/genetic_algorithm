#!/bin/bash

N_POINTS=500

rm -f coeffs.log

for i in {1..100}
do
    ../generator $N_POINTS >> coeffs.log 
    mv input_file.txt input$i.txt

done
