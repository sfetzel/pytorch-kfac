#!/bin/sh

for file in comparison-*; do
	echo "====$file====";
	tail $file -n 2
	kfac=`grep "^{.*'kfac_damping': 0.1" $file | wc -l`
	echo "   KFAC: $kfac";
done
