#!/bin/bash

fontdir='/usr/share/fonts'
imagedir='./fontImages'
tabledir='./fontTables'

fonts=$(find "$fontdir"/* -name '*ttf' | sort)

for font in $fonts; do
	fontname=$(basename $font)
	echo $fontname
	fontTable=$tabledir'/'$fontname'.ttx'
	ttx -q -t cmap -o $fontTable $font
	for letter in {A..Z}; do
		exists=$(grep 'name="'$letter'"' $fontTable) # check if letter exists in that font
		if [ -z "$exists" ]; then continue; fi # if not skip font
		filename=$imagedir'/'$fontname'_'$letter.png
        	convert -background white -fill black -font $font -pointsize 300 label:"$letter" $filename
        	convert $filename -resize 64x128! $filename
	done
done

