## Imports
import mge_bootstrap as mge


def main():
    mge.create_input_file()
    mge.bootstrap_mge_cluster()
    perplexity = mge.bootstrap_rand("perplexity")
    clustersize = mge.bootstrap_rand("clustersize")
    mge.optimised_mge(perplexity, clustersize)


if __name__ == "__main__":
    main()
