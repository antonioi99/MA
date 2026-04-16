#!/bin/bash

#SBATCH --job-name="delete_slurm"
#SBATCH --container-image=/srv/home/users/a12225670cs/MA/loris3+antonio+latest.sqsh
#SBATCH --container-mount-home 
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=0-20:00:00
#SBATCH --container-workdir=/srv/home/users/a12225670cs/MA

for f in slurm-*.out; do
    [[ -e "$f" ]] || continue

    if grep -q "total instances: 9966" "$f"; then
        echo "Would delete: $f"
        # rm "$f"
    fi
done

echo "done"