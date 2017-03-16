# -*- coding: utf-8 -*-
"""A module to simulate ETD data. Only basic functions for simulations."""
import numpy.random as rd
import numpy as np
import scipy.stats as ss
from shared import CHARGE_RADIUS, CHARGE_FACTOR, BASIC_AA
from reaction_graph import create_molecule_graph
from math import ceil 
import os, time
from meanfield import get_params


def reaction(substrate, precursor, substrate_charges, intsy_dict, multiple_ETD=False):
    """
    Simulates one step of fragmentation (i.e. a single PTR, ETnoD or ETD reaction).

    Parameters
    -----
    substrate, precursor: molecules
    substrate_charges: str
        A string representing substrate charges. "+" for proton, "o" for quenched proton, "-" for no charge.
        E.g. "--+-0+-" for 2 positive and 1 quenched proton.
    
    Return: tuple
    -----
        Reaction time and a list of reaction products. 
    
    Details 
    -----
        Assumes that only one ETD reaction can occur. The total intensity
        of reactions is lower after ETD (i.e. the ETD intensity is not 
        included in the total reaction intensity).
    """
    assert sum(x=="+" for x in substrate_charges) == substrate[1], "Improper proton labelling"
    assert sum(x=="0" for x in substrate_charges) == substrate[2], "Improper electron labelling"
    assert len(substrate_charges) == len(substrate[0]), "Improper charge labelling"
    outflow_intensity = intsy_dict['PTR'] + intsy_dict['ETnoD'] + intsy_dict['ETD']  # theoretical outflow, used to get reaction time
    if (multiple_ETD or substrate[0] == precursor[0]) and len(substrate[0]) > 1:
        # ETD can occur
        r_types = intsy_dict.keys()  # reaction types
    else:
        # ETD can't occur
        r_types = [r for r in intsy_dict.keys() if r != 'ETD']
    c = sum(intsy_dict[x] for x in r_types)  # actual outflow intensity
    if substrate[1] == 0:
        return (-1, [substrate], [substrate_charges])
    elif abs(float(c)) <= 1e-08:
        return(-1, [substrate], [substrate_charges])
    p_vect = [intsy_dict[x]/c for x in r_types]
    outflow_intensity *= CHARGE_FACTOR(substrate[1])
    # sampling    
    reaction_type = rd.choice(a=r_types, p=p_vect)
    reaction_time = rd.exponential(scale=1./outflow_intensity)
    charge_sites = [i for i, x in enumerate(substrate_charges) if x == "+"]
    if reaction_type == 'PTR':
        reaction_site = rd.choice(charge_sites, 1)[0]
        products = [(substrate[0], substrate[1]-1, substrate[2], substrate[3])]
        product_charges = [substrate_charges[:reaction_site]+'-'+substrate_charges[(reaction_site+1):]]
    if reaction_type == 'ETnoD':
        reaction_site = rd.choice(charge_sites, 1)[0]
        products = [(substrate[0], substrate[1]-1, substrate[2]+1, substrate[3])]
        product_charges = [substrate_charges[:reaction_site]+'0'+substrate_charges[(reaction_site+1):]]
    elif reaction_type == 'ETD':
        if substrate[3] != 'precursor':
            # ETD already occured
            products = []
            product_charges = []
        else:
            # left_seq_len = rd.randint(low=1, high=len(substrate[0]))
            reacting_proton = rd.choice(charge_sites, 1)[0]
            breakable_bonds = [i for i in range(len(precursor[0])-1) if precursor[0][i+1] != 'P']
            cleavage_site = rd.choice(breakable_bonds) + 1  # length of the N-terminal product
            # ETD_list[cleavage_site-1] += 1
            result_charges = substrate_charges[:reacting_proton] + '-' + substrate_charges[(reacting_proton+1):]
            left_charges = result_charges[:cleavage_site]  # terminal electron is implicit in the c-fragment
            right_charges = result_charges[cleavage_site:]
            l_charged = sum(x=="+" for x in left_charges)
            r_charged = sum(x=="+" for x in right_charges)
            l_quenched = sum(x=="0" for x in left_charges)
            r_quenched = sum(x=="0" for x in right_charges)
            products = [(substrate[0][:cleavage_site], l_charged, l_quenched, "c%i" % cleavage_site),
                        (substrate[0][cleavage_site:], r_charged, r_quenched, "z%i" % (len(substrate[0]) - cleavage_site))]
            product_charges = [left_charges, right_charges]
    return (reaction_time, products, product_charges)


def fragment(precursor, tmax, intsy_dict, multiple_ETD=False):
    """
    Returns one chain of fragmentation reactions of one precursor molecule
    from time 0 to tmax.

    Returns
    -----
        A list of tuples (m, e, l), where m is a molecule, e is the time
        of entry of molecule into sample and l is the time when the
        molecule left the sample.
    """
    assert precursor[0][0] == '*', "No N-terminal amino group ('*') on the precursor molecule."
    # Sampling the charge distribution:
    chargeable_sites = [i for i in range(0, len(precursor[0])) if precursor[0][i] in BASIC_AA]
    charge_sites = list(rd.choice(chargeable_sites, precursor[1], replace=False))
    precursor_charges = ''.join('+' if i in charge_sites else '-' for i in range(len(precursor[0])))
    sample = [(precursor, precursor_charges, 0)]  #  (molecule, charge distribution, entry time)
    trajectory = []  # molecule, time of entry, time of leave   
    while sample:
        molecule, molecule_charges, entry_time = sample.pop()
        if entry_time < tmax:        
            reaction_time, products, charges = reaction(molecule,
                                                        precursor,
                                                        molecule_charges,
                                                        intsy_dict,
                                                        multiple_ETD)
            if reaction_time == -1:
                leave_time = tmax
            else:   
                leave_time = entry_time + reaction_time
            trajectory.append((molecule, entry_time, leave_time))
            sample.extend([(x, c, leave_time) for x, c in zip(products, charges)])
        #elif molecule[1] == 0:
        #    trajectory.append((molecule, entry_time, -1))
    return trajectory


def iterate_over_products(precursor, x0, tmax, intsy_dict, multiple_ETD=False):
    """
    An iterator over results of fragmentation reaction of x0 precursor
    molecules from time 0 to tmax.
    """
    for _ in range(int(x0)):
        for m in fragment(precursor, tmax, intsy_dict, multiple_ETD):
            yield m


# Dictionary output format
def simulate(precursor, n=10000, ETD=1., PTR=1., ETnoD=1., t=1., return_sink=False, multiple_ETD = True, density = True):
    """
    Returns a dictionary with densities of products of fragmentation
    reaction at a given time. 

    Parameters
    -----
    precursor: tuple
        Precursor molecule: (Seq, p, q).
    n: int
        Number of input molecules.
    t: float
        Time at which the state of the sample is returned.
    etd, ptr, etnod: float
        Reaction intensities.
    return_sink: bool
        Should non-observable molecules be returned?
    multiple_ETD: bool
        Can ETD fragmentation occur more than once?
    density: bool
        If true, product densities will be returned; Otherwise, numbers of molecules.

    Value
    -----
    out: dict
        A dictionary indexed by product molecules, storing their proportions.
    
    Details
    -----
        The numbers of simulated product molecules are normalized by the initial
        number of precursor molecules (n), i.e. it is assumed that x_0 = 1. 
    """
    np.random.seed()
    intsy_dict = {'ETD': ETD, 'ETnoD': ETnoD, 'PTR': PTR}
    #molecules = create_molecule_graph(precursor).keys() + [precursor]
    #if not return_sink:
    #    molecules = filter(lambda x: x[1] != 0, molecules)
    #products = dict([(m, 0.) for m in molecules])
    products = dict()
    skeleton = iterate_over_products(precursor,
                                     n,
                                     t+1.,
                                     intsy_dict,
                                     multiple_ETD)
    for m, entry_time, leave_time in skeleton:
        if m[1] > 0 or return_sink:
            if entry_time <= t and leave_time > t:
                # using try-except, we do not need to additionally iterate
                # over the dictionary to check if there's a proper key
                # when we increment the value for an existing key
                try:
                    products[m] += 1.
                except KeyError:
                    products[m] = 1.
    if density:
        product_sum = float(sum(products.itervalues()))
        # Normalization to get density:
        for m in products:
            products[m] /= product_sum
    return products


# Tabular format
def skeleton_to_table(precursor, skeleton, tmax, h=0.01):
    """
    Converts a fragmentation reaction in a skeleton notation
    into tabular notation. Internal fragments are not listed in the table.
    """
    molecules = create_molecule_graph(precursor).keys() + [precursor]
    nb_of_molecules = len(molecules)
    table = np.array([[0. for _ in range(nb_of_molecules+1)] for _ in range(int(tmax/h) + 1)])
    for i in range(0, int(tmax/h) + 1):
        t = i*h
        table[i, 0] = t
    for m, entry_time, leave_time in skeleton:
        i1 = int(ceil(entry_time/h))
        i2 = int(leave_time/h)
        if m in molecules:
            table[i1:(i2+1), 1 + molecules.index(m)] += 1.
    return (molecules, table)


# Tabular output format
def reaction_summary(precursor, x0, tmax, intsy_dict, multiple_ETD=False, h=0.01):
    """
    Returns results of fragmentation reaction of x0 precursor molecules
    in tabular format, with time interval from 0 to tmax divided into 
    parts of length h.
    """
    m, t = skeleton_to_table(precursor,
                             iterate_over_products(precursor,
                                                   x0,
                                                   tmax,
                                                   intsy_dict,
                                                   multiple_ETD),
                             tmax,
                             h)
    return (m, t)


def noisify(data, sd, density = True):
    """
    Add gaussian noise to data.
    :param data: dict
        As returned from simulate. Should be unnormalized.
    :param sd: float
        Standard deviation of noise.
    :param density: bool
        If true, product numbers will be renormalized after noise addition.
    :return: dict
        Simulation results in dictionary format with gaussian noise added.
    """
    data = data.copy()
    N = ss.norm.rvs(loc=0, scale = sd, size = len(data))
    for i, k in enumerate(data):
        data[k] += N[i]*data[k]
        data[k] = max(data[k], 0.)
    if density:
        product_sum = float(sum(data[k] for k in data))
        for k in data:
            data[k] /= product_sum
    return data


def filter_data(data, p, density = True):
    """
    Randomly remove molecules from data (i.e. assign 0. to random keys).
    :param data: dict
        As returned from simulate.
    :param p: float
        Approximate proportion of keys to be assigned 0.
    :param density: bool
        If true, product numbers will be renormalized after filtering.
    :return: dict
    """
    data = data.copy()
    for k in data:
        if ss.uniform.rvs() <= p:
            data[k] = 0.
    if density:
        product_sum = float(sum(data[k] for k in data))
        for k in data:
            data[k] /= product_sum
    return data


if __name__=='__main__':
    intsy_dict = {'PTR': 1., 'ETnoD': 1., 'ETD': 1.}
    P = ('*ARPWV', 2, 0, 'precursor')
    # ETD_list = [0] * (len(P[0])-1)
    trajectory = fragment(P, 4, intsy_dict)
    table = skeleton_to_table(P, trajectory, 4)
    skeleton = []
    print "Computing"
    for _ in range(100):
        skeleton.extend(fragment(P, 4, intsy_dict, multiple_ETD=True))
    print "Finished"
    m, table = skeleton_to_table(P, skeleton, 4)
    print "Parsed"
    d = simulate(P, multiple_ETD=True, return_sink=True)
    #m1, table1 = reaction_summary(P, 10000, 4, intsy_dict)
    #print "Parsed2"
