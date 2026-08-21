# MRSA Plasmid Analysis
Runs all components from the Dutch surveillance MRSA plasmidome paper.
This is seperated in 5 steps:
1. Verify all required files exist
2. Mge-cluster bootstrap to create scheme
3. General population statistics/description
4. Cluster statistics/description
5. Nearly-identical neighbour analysis

# Installation
Get the repository files and install required packages
```bash
git clone https://github.com/gidjes/mrsa_plasmid_analysis.git
cd mrsa_plasmid_analysis
./init.sh
```

If succesfully installed you can now run:
```bash
mrsa_plasmid_analysis
```
or
```bash
python scripts/mrsa_plasmid_analysis.py
```