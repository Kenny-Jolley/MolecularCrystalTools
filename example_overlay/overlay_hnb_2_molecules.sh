#!/bin/bash

# Ensure the given structure is unwrapped first
echo Unwrapping hnb.xyz first
python ../scripts/unwrap_cell.py hnb.xyz hnb_unwrapped.xyz

# Overlay the 2 molecule structure on top
python ../scripts/overlay_molecules.py hnb_unwrapped.xyz hnb_2_molecules.xyz --rot_steps=3