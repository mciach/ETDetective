# ETDetective
A software for estimating the intensities of ETD and accompanying charge-reduction reactions

## SYNOPSIS

>python ETDetective.py [OPTIONS] [INPUT FILE] [OUTPUT DIRECTORY]

## DESCRIPTION

Estimates reaction intensities from data in FILE.
Saves the estimates and predicted spectrum in DIRECTORY.
FILE has to be a tab-separated file with columns for product subsequence, active protons, neutral protons, ion type (c/z), intensities.
The first line has to contain the precursor molecule.

## OPTIONS

    -h
        Print help message
    -d: int
        Numbers of points on initial grid per dimention (default 6)
    -t: int
        Maximum number of threads (default 4)
    -s
        Turn on penalizing for decharged molecules
    -v
        Verbose mode
