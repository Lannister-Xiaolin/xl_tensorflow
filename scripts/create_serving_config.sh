#!/usr/bin/env bash
d=$(ls -l $1|awk '/^d/ {print $NF}')
echo $d