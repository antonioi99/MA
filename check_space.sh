#!/bin/bash
#SBATCH --nodelist=dgx1
#SBATCH --partition=p_low
#SBATCH --time=00:05:00
#SBATCH --job-name=check_space

echo "=== Checking /run/pyxis ==="
df -h /run/pyxis

echo "=== Checking /tmp ==="
df -h /tmp

echo "=== Checking home ==="
df -h $HOME