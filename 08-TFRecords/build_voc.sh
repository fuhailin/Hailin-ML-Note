#! /bin/sh
#
# build_voc.sh
# Copyright (C) 2018 CloudBrain <byzhang@>
#
# Distributed under terms of the CloudBrain license.
#


for i in `seq 15 40`; do 
  c=$((i - 1))
  echo "processing col $c..."
  cut -f $i ./train.txt |sort |uniq -c | awk '{print $2}' > f$c.voc
done
