#!/usr/bin/env bash

python -m trip.tools.torsion_benchmark --gpu 0 --start 0 --stop 60 &\
python -m trip.tools.torsion_benchmark --gpu 1 --start 60 --stop 120 &\
python -m trip.tools.torsion_benchmark --gpu 2 --start 120 --stop 180 &\
python -m trip.tools.torsion_benchmark --gpu 3 --start 180 --stop 240 &\
python -m trip.tools.torsion_benchmark --gpu 4 --start 240 --stop 300 &\
python -m trip.tools.torsion_benchmark --gpu 5 --start 300 --stop 360 &\
python -m trip.tools.torsion_benchmark --gpu 6 --start 360 --stop 420 &\
python -m trip.tools.torsion_benchmark --gpu 7 --start 420 --stop 482

