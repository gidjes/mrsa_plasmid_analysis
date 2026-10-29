# MRSA Plasmid Analysis
Runs all code/analysis for "Extensive plasmid sharing across genetically divergent methicillin-resistant _Staphylococcus aureus_ complex".
This is seperated in 5 steps:
1. Load the data
2. Mge-cluster bootstrap to create scheme
3. General population statistics/description
4. Cluster statistics/description
5. Near-identical neighbour analysis

## Installation
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

### Software dependencies and operating system
The analysis was performed on a Linux server running Red Hat Enterprise Linux (RHEL) 8.10 (Ootpa), with Linux kernel 4.18.0-553.155.1.el8_10.x86_64 (x86_64 architecture).

Software dependencies were managed using Conda and pip. The required packages and their version numbers are specified in the environment configuration files in the `envs/` directory.

The Conda environments can be recreated using the corresponding YAML files. See the installation instructions for details.

### Typical installation time
Installation of the required software dependencies using Conda typically takes approximately **50 minutes** on a standard desktop computer (16 GB RAM, modern multi-core CPU, SSD, and a broadband internet connection).

Installation time may vary depending on internet speed, package availability, and whether dependencies are already cached.

This estimate covers installation of the software dependencies and does not include downloading the study dataset or executing the analysis pipeline.

### Testing
A small set of demo data is provided in the directory `test-data.zip`. Unzipping these files will create a small test data set in a directory called `fastas` with the matching isolate fastas in `fastas_chr` and the required metadata files in `data`. Running the code should then create all figures, tables, and supplementary files used in the paper. In addition, all other results mentioned in the paper itself will be written to txt and csv files.

