#!/bin/bash
#SBATCH --mem=80000
#SBATCH --time=0
#SBATCH --output=slurm_out/slurm-%j.out
set -e

python hiro_code.py --wikitext-dir data/title_id_map --extracted-dir data/hiro_wikidata/ --output-dir data/subgraph/
