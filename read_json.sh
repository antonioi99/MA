#!/bin/bash

#SBATCH --job-name="check_slurm"
#SBATCH --container-image=/srv/home/users/a12225670cs/MA/loris3+antonio+latest.sqsh
#SBATCH --container-mount-home 
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=0-20:00:00
#SBATCH --container-workdir=/srv/home/users/a12225670cs/MA

#!/bin/bash

for f in slurm-*.out; do
    [[ -e "$f" ]] || continue

    if ! grep -q "slurmstepd: error:" "$f"; then
        continue
    fi

    json_path=$(grep -m1 "Output file:" "$f" | sed 's/.*Output file:[[:space:]]*//')

    [[ -z "$json_path" ]] && continue

    if ! [[ -r "$json_path" ]]; then
        echo "Missing: $json_path"
    elif ! python3 -m json.tool "$json_path" > /dev/null 2>&1; then
        echo "Corrupted: $json_path"
    fi
done

echo "done"