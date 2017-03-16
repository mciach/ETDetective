# -*- coding: utf-8 -*-
"""
Computation of mean numbers of ETD reaction products.
The functions for reaction graph (including reaction intensities) are moved to reaction_graph.py.
"""
import sympy
# from matplotlib import pyplot
import numpy as np
from scipy.stats import binom as binom_rv
from scipy.special import binom as binom_coef
from shared import SINK, CHARGE_RADIUS, P, CHARGE_FACTOR, BASIC_AA
from reaction_graph import get_ancestors

# Produkt = (Sekwencja, Protony, Elektrony) = (Seq, p, q)
# SINK jest jedynym produktem w którym Seq = ''.  
# SINK oznacza dowolną cząsteczkę w której p = q.


def sum_parameters(param):
    """
    Sums the equation parameters which have the same exponent and polynomial term.
    (a, ni, bi) and (c, ni, bi) become (a + c, ni, bi).
    
    Parameters
    -----------
    param: list
        list of tuples (Ai, ni, Bi)

    Returns
    ----------
    out: list
        list of tuples (Ci, ni, Bi).
    """
    # Dictionary of (ni, Bi): Ai
    Ai = {} 
    for x in param:
        try:
            Ai[x[1:]] += x[0]
        except KeyError:
            Ai[x[1:]] = x[0]
    return [(Ai[x], x[0], x[1]) for x in Ai]


def hypergeom(k, M, n, N):
    """Scipy's hypergeom gives wrong results sometimes."""
    return binom_coef(n, k)*binom_coef(M-n, N-k)/binom_coef(M, N)


def update_coefficients(coefs, p_in, p_out):
    """
    Returns a list of updated coefficients. The list represents the coefficients of a node
    obtained from one tuple of coefficients of an ancestral node.

    Parameters
    ----------
    coefs: tuple of floats or sympy.symbol objects
        Ancestral coefficients. A length 3 tuple (A, n, B).
    p_in: sympy.symbol object or float
        In-flow intensity (i.e. intensity of reaction from ancestor to current node)
    p_out:
        Out-flow intensity, i.e. sum of intensities of reactions of the current node
    """
    A, n, B = coefs
    updated_coefs = []
    if p_out == B:
        updated_coefs.append((A*p_in/(n+1), n+1, p_out))
    else:
        for k in range(n+1):
            updated_coefs.append((A*p_in*sympy.prod((i for i in range(k+1, n+1)))*((-1)**(n-k))/((p_out - B)**(n-k+1)),
                                  k,
                                  B))
        updated_coefs.append((A*p_in*sympy.prod((i for i in range(1, n+1)))*((-1)**(n+1))/((p_out - B)**(n+1)),
                              0,
                              p_out))
    return updated_coefs


def get_params(product, precursor, x_0, intensity_dict, cleavage_probabilities, parameter_dict, verbose=False):
    """
    Updates the parameter dictionary and returns the differential equation
    parameters for the given product.

    Parameters
    ----------
    product: tuple
        Product molecule for which the parameters are to be computed.
        Represented as a length 3 tuple (Seq, p, q, P), where Seq is the product sequence,
        p is the number of charged protons, q is the number of quenched protons (gained in ETnoD) and
        P is the subsequence position ("L" for left fragments, "R" for right fragments,
        "P" for precursor sequence).
    precursor: tuple
        The ETD reaction precursor. A tuple (Seq, p, q).
    x_0: float
        The starting number of precursor molecules.
    PTR: sympy.symbol object
        The PTR reaction intensity.
    ETnoD: sympy.symbol object
        The ETnoD reaction intensity.
    ETD: sympy.symbol object
        The base ETD reaction intensity. Different cleavages are assumed to have equal probability.
        The distribution of protons and electrons follows a binomial distribution with probability
        proportional to the length of a product.
    parameter_dict: dictionary
        The dictionary of pre-computed equation parameters, indexed by molecules (i.e. tuples (Seq, p, q)).
        It may be changed during runtime of the function.
        The parameters are stored as lists of tuples (A_i, n_i, B_i) such that the sum over i of A_i*t**{n_i}*exp(-t*B_i)
        is the number of molecules of given product at time t.

    Value
    ----------
    return: list
        A list of tuples (A_i, n_i, B_i), which represent the equation for the number of product molecules.
        The resulting equation is sum over i of A_i*t**{n_i}*exp(-t*B_i).
    """
    global CHARGE_FACTOR
    if verbose:
        print "get_params invoked for", product
    PTR = intensity_dict['PTR']
    ETnoD = intensity_dict['ETnoD']
    ETD = intensity_dict['ETD']
    if product == precursor:
        # B_i of precursor = outflow intensity = charge*(PTR + ETnoD + ETD)
        par = [(x_0, 0, (PTR + ETnoD + ETD)*CHARGE_FACTOR(precursor[1]))]
        # insert the above list of parameters to parameter_dict
        parameter_dict[product] = par
        return par
    elif product in parameter_dict:
        return parameter_dict[product]
    else:  # Compute the parametes and update the dictionary.
        outflow_intensity = (PTR + ETnoD + ETD)*CHARGE_FACTOR(product[1])
        if verbose:
            print "Outflow:", outflow_intensity
        # The direct ancestor molecules and intensities of corresponding reactions:
        ancestors, intensities = get_ancestors(product,
                                               precursor,
                                               intensity_dict,
                                               cleavage_probabilities)
        if verbose:
            print "Current molecule ancestors:"
            print ancestors

        # Recursively compute the parameters
        # (note that this procedure memoizes by updating parameter_dict)
        parameters = [get_params(anc,
                                 precursor,
                                 x_0,
                                 intensity_dict,
                                 cleavage_probabilities,
                                 parameter_dict,
                                 verbose) for anc in ancestors]
        if verbose:
            print "Computed parameters before update:"
            print parameters

        # Update the parameters
        new_parameters = []
        for i, anc_param_list in enumerate(parameters):
            for j, param in enumerate(anc_param_list):
                # extend the list with a list of updated parameters
                # intensities[i] = inflow intensity from i-th ancestral molecule
                if verbose:
                    print "Param:", param, "Intens:", intensities[i], "New param:", update_coefficients(param,
                                                                                                        intensities[i],
                                                                                                        outflow_intensity)
                new_parameters.extend(update_coefficients(param, intensities[i], outflow_intensity))

                # previous versions:
                # parameters[i][j] = (param[0]*intensities[i]/(outflow_intensity - param[1]), param[1])
                # alternatively, use sympy without simplification:   
                #parameters[i][j] = (param[0]*intensities[i]/sympy.Add(outflow_intensity, -param[1], evaluate=False), param[1])

        # Flatten the parameter list
        # parameters = [t for a_p_l in parameters for t in a_p_l]

        # Append the new (outflow) parameter
        # parameters.append((-sum(p[0] for p in parameters), outflow_intensity))

        # Sum the intensities with the same exponents and polynomials
        new_parameters = sum_parameters(new_parameters)
        if verbose:
            print "Updated parameters:"
            print new_parameters
        # Update the parameter dictionary and return the parameters
        # (note that this uses the pointer, i.e. modifies the dictionary)
        parameter_dict[product] = new_parameters
        return new_parameters


def evaluate_sympy_dict(parameter_dict, tmax, x0=1000., etd=1, etnod=1, ptr=1, h=0.1):
    """Returns a dictionary of the product proportions given a dictionary of parameters."""
    values = {'T': [tt*h for tt in range(int(tmax/h))]}
    t = sympy.symbols('t')
    for key, value in parameter_dict.iteritems():
        a = sum(tt[0]*(t**tt[1])*sympy.exp(-tt[2]*t) for tt in value)
        f = lambda time: a.subs([('etd', etd), ('etnod', etnod), ('ptr', ptr), ('x0', x0), ('t', time)]).evalf()
        values[key] = [f(tt*h) for tt in range(int(tmax/h))]
    return values


# def visualize(value_dict, legend=True):
#     """Plots the products."""
#     T = value_dict['T']
#     for key, value in value_dict.iteritems():
#         if key != 'T':
#             pyplot.plot(T, value, label=key)
#     if legend:
#         pyplot.legend()
#     pyplot.show()
    

def save_evaluation(value_dict, save_path):
    """Saves the evaluation in a csv file."""
    T = value_dict['T']
    keys = [x for x in value_dict.keys() if x!='T']
    keys_to_write = [x[0] + '_' + str(x[1]) + '_' + str(x[2]) for x in value_dict.keys() if x!='T']
    with open(save_path, 'w') as h:
        h.write('T, ' + ', '.join(keys_to_write) + '\n')
        for i, t in enumerate(T):
            h.write(str(t))
            for k in keys:
                h.write(', ' + str(value_dict[k][i]))
            h.write('\n')


# Main function:
def mean_values(precursor, t=1., ETD=1., PTR=1., ETnoD=1.,
                cleavage_probabilities=None, return_sink=False, proportions=True):
    """
    Returns a dictionary of proportions of products of fragmentation
    with given reaction intensities.

    Parameters
    -----
    precursor: tuple
        Precursor molecule. Represented as a length 3 tuple (Seq, p, q, P), where Seq is the product sequence, \
        p is the number of charged protons, q is the number of quenched protons (gained in ETnoD) and
        P is the subsequence position ("L" for left fragments, "R" for right fragments,
        "P" for precursor sequence).
    t: float
        Time at which the reaction is evaluated.
    etd, ptr, etnod: float
        Reaction intensities.
    cleavage_probabilities: iterable
        An iterable of length one shorter than the precursor sequence. Has to sum to 1.
        The i-th value is the probability that a cleavage occurs between i-th and i+1-th residue, given that
        ETD reaction occurs. If None, uniform probability is assumed.
    return_sink: bool
        Should non-observable molecules be included in returned dictionary?

    Value
    -----
    out: dict
        A dictionary indexed by molecules (Seq, p, q) with product absolute intensities (assuming one initial molecule)
        or proportions as values.
    """
    global SINK
    param_dict = dict()
    intsy_dict = {'ETD': ETD, 'PTR': PTR, 'ETnoD': ETnoD}
    value_dict = dict()
    if not cleavage_probabilities:
        cleavage_probabilities = [1. / (len(precursor[0])-1)]*(len(precursor[0])-1)
    get_params(product=SINK,
               precursor=precursor,
               x_0=1.,
               intensity_dict=intsy_dict,
               cleavage_probabilities=cleavage_probabilities,
               parameter_dict=param_dict,
               verbose=False)
    for mol in param_dict:
        if return_sink or mol[1] != 0:
            x = sum(p[0]*(t**p[1])*np.exp(-t*p[2]) for p in param_dict[mol])
            value_dict[mol] = x
    if proportions:
        molecule_number = sum(value_dict[k] for k in value_dict)
        if abs(molecule_number) > 1e-12:
            for k in value_dict:
                value_dict[k] /= molecule_number
        else:
            for k in value_dict:
                value_dict[k] = 0.
    return value_dict
    

if __name__=="__main__":
    ptr, etnod, etd, x0, t = sympy.symbols('ptr etnod etd x0 t')
    intsy_dict = {'PTR': ptr, 'ETnoD': etnod, 'ETD': etd}
    intsy_dict2 = {'PTR': 1., 'ETnoD': 1., 'ETD': 1.}
    get_ancestors(SINK, P, intsy_dict, [1. / (len(P[0])-1)]*(len(P[0])-1))
##    print "Obtaining parameters for (A, 1, 0)"

##    param_dict = dict()
##    get_params(('A', 1, 0), ('AKVGK', 2, 0), x0, intsy_dict, param_dict)
##    a = sum(tt[0]*(t**tt[1])*sympy.exp(-tt[2]*t) for tt in param_dict[('A', 1, 0)])
##    a_eval = lambda time: a.subs([('etd', 1), ('etnod', 1), ('ptr', 1), ('x0', 10), ('t', time)]).evalf()
##
##    #vd = evaluate_sympy_dict(param_dict, 3)
##    param_dict2= dict()
##    #get_params(('A', 1, 0), ('AKVGK', 2, 0), 1, 1, 1, 10, 1, param_dict2)                                                         
##    #b_eval = sum(t[0]*(time**t[1])*sympy.exp(-t[2]) for t in param_dict2[('A', 1, 0)])
##
    P1 = ('ARNWV', 2, 0, "precursor")
    v = mean_values(P1, 1., 1., 1., 1.)
    
    param_dict = dict()
    get_params(SINK, P1, 10., intsy_dict, [1. / len(P1[0]) for _ in range(len(P1[0]))], param_dict)
    higher_order = [(y, x) for y in param_dict for x in param_dict[y] if x[1] != 0]

    
    t2 = mean_values(('ARNWV', 2, 0, "precursor"), 2., 1., 1., 1.)
    i2 = mean_values(('ARNWV', 2, 0, "precursor"), 1., 2., 2., 2.)
