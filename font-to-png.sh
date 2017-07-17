#!/bin/bash

fontdir='/usr/share/fonts'
imagedir='./fontImages'

fonts=$(find "$fontdir"/* -name '*ttf')

for font in $fonts; do
	fontname=$(basename $font)
	echo $fontname
	for letter in {A..Z}; do
		filename=$imagedir'/'$fontname'_'$letter.png
        	convert -background white -fill black -font $font -pointsize 300 label:"$letter" $filename
        	convert $filename -resize 64x128! $filename
	done
done

