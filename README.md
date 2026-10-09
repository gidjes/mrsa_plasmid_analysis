# MRSA Plasmid Analysis
Runs all code/analysis for "Extensive plasmid sharing across genetically divergent methicillin-resistant _Staphylococcus aureus_ complex".
This is seperated in 5 steps:
1. Load the data
2. Mge-cluster bootstrap to create scheme
3. General population statistics/description
4. Cluster statistics/description
5. Near-identical neighbour analysis

# Installation
Get the repository files and install required packages
```bash
git clone https://github.com/gidjes/mrsa_plasmid_analysis.git
cd mrsa_plasmid_analysis
./init.sh
```

If succesfully installed you can now run:
```bash
python scripts/mrsa_plasmid_analysis.py
```