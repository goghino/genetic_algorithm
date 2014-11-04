i=0 

if [ ! -f $1 ]; then
    echo "File not found!"
    exit
fi

while read line; do
    eval coeffs=($line)
    #echo ${coeffs[1]}
    gnuplot -e "c3g='${coeffs[3]}'" -e "c2g='${coeffs[2]}'" -e "c1g='${coeffs[1]}'" -e "c0g='${coeffs[0]}'" -e "out='plot$i.png'" -e "generation='Generation #$i'" plot_conv.gnu
    let "i+=1"
done < $1 

#concat pngs into movie
ls -v *.png > list.txt
zip plots.zip *.png list.txt
rm -f *.png

#Input file format: c3 c2 c1 c0 
#-5.00719 2.97788 4.04544 -2.00324
#-5.0066 2.97978 4.04286 -2.00263

#Create video from generated plots
#   unzip plots.zip &&
#   mencoder mf://@list.txt -mf w=1920:h=1080:fps=25:type=png -ovc lavc -#lavcopts vcodec=mpeg4:mbd=2:trell -oac copy -o output.avi


