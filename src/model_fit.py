# -*- coding: utf-8 -*-
from meanfield import mean_values
from simulations import simulate
from bayes_opt import BayesianOptimization
import numpy as np
from scipy.optimize import fmin_bfgs, fmin_l_bfgs_b
from shared import BASIC_AA

def make_target_function(precursor, data, min=False, use_null_observations=True):
    """
    Returns a function that takes reaction parameters and returns
    the goodness of fit between the mean model and the data (currently -SSE).

    Parameters
    -----
    precursor: tuple
        The reaction precursor molecule in form (Seq, p, q).
    data: dict
        A dictionary indexed by observable product molecules (i.e. p != 0),
        representing a state of the sample.
        The values are the proportions of given product molecule in the sample.
    min: bool
        Should the optimal point be a minimum of the target function (i.e. multiply result by -1)?
        By default it is a maximum.
    use_null_observations: bool
        If false, molecules that are too smal to be observed in a spectrometer will not be used in fitting.
    Value
    -----
    out: function
        target(ETD, PTR, ETnoD, t=1).
    """
    # To avoid pointer issues:
    data = data.copy()

    def target(ETD, PTR, ETnoD, cleavage_probabilities, t=1.):
        """
        Return goodness of fit to data for given set of parameters
        :param ETD_list: tuple
            Tuple of cleavage intensities. The i-th value is the intensity of ETD cleavage
            between i-th and i+1-th residue.
        :param PTR: float
        :param ETnoD: float
            Log-intensities of respective side reactions
        :param t:
            Time of evaluation
        :return: float
            Logarithm of sum of squared errors.
        """
        #print "T"
        # print "Evaluating target at"
        # print PTR, ETnoD, ' '.join(ETD_list)
        # ETD_list = np.exp(logETD_list)
        assert abs(sum(cleavage_probabilities)-1.) < 1e-04, "Probabilities do not sum to 1."
        # total_ETD = sum(ETD_list)
        # cleavage_probabilities = [ei/total_ETD for ei in ETD_list]
        v = mean_values(precursor,
                        t,
                        ETD,
                        PTR,
                        ETnoD,
                        cleavage_probabilities)
        #  Including molecules predicted but not observed
        missing_keys = set(v).difference(data)
        for k in missing_keys:
            if use_null_observations or len(k[0]) > 1:
                data[k] = 0.
        molecules = set(v) & set(data)
        if min:
            sgn = 1
        else:
            sgn = -1
        if not molecules:
            raise ValueError(
                "No molecules in the data! (Perhaps improper dictionary keys? Or maybe Total Annihilation occured?)")
        else:
            return sgn*np.log(sum((v[m] - data[m])**2 for m in molecules))
            # return -sum((v[m] - data[m])**2 for m in v)
    return target


def bfgs_arg_to_meanfield_arg(x, precursor_sequence_length, unbreakable_bonds):
    """
    Converts a list of arguments as passed to bfgs target function into reaction intensities
    and cleavage probabilities.
    :param x: list
        A list representing an argument of BFGS target function, containing
        reaction log-intensities (see bfgs_target docstring for details).
        It has to be of length 1 + precursor_sequence_length - len(unbreakable_bonds).
    :param unbreakable_bonds: list
        A list of integers representing unbreakable bonds in the precursor molecule. An integer i means
        that the bond between i-th and i+1-th aminoacid (counting from 0) can not be broke.
    :return: tuple
        (ETD, PTR, ETnoD, cleavage_probabilities)
    """
    PTR = np.exp(x[0])
    ETnoD = np.exp(x[1])
    ETD_list = np.exp(x[2:])
    ETD = sum(ETD_list)
    nb_of_unbreakable_bonds = len(unbreakable_bonds)
    cleavage_probabilities = [1./(nb_of_unbreakable_bonds)] * (precursor_sequence_length - 1)
    for i in unbreakable_bonds:
        cleavage_probabilities[i] = 0.
    if ETD > 1e-18:
        shift = 0
        for i in range(precursor_sequence_length-1):
            if i in unbreakable_bonds:
                shift += 1
            else:
                cleavage_probabilities[i] = ETD_list[i - shift] / ETD
    if abs(sum(cleavage_probabilities) - 1) > 1e-04:
        raise RuntimeError("Probabilities do not sum to 1.")
    return (ETD, PTR, ETnoD, cleavage_probabilities)


def make_bfgs_target_function(precursor, data, unbreakable_bonds, sink_penalty=False):
    """
    Returns a function that takes reaction parameters and returns
    the goodness of fit between the mean model and the data (currently -SSE).

    Parameters
    -----
    precursor: tuple
        The reaction precursor molecule in form (Seq, p, q, label).
    data: dict
        A dictionary indexed by observable product molecules (i.e. p != 0),
        representing a state of the sample.
        The values are the proportions of given product molecule in the sample.
    sink_penalty: bool
        If true, predicted intensities of fully discharged (i.e. non-observable) molecules
         will be included in the target function value, thus lowering the returned overall
         intensity.
    Value
    -----
    out: function
        bfgs_target(x).
    """
    # To avoid pointer issues:
    data = data.copy()

    def bfgs_target(x):
        """
        Return goodness of fit to data for given set of parameters
        :param x: list
            A list of reaction logintensities: logPTR, logETnoD, logETDn, where logETDn is equal to
            log(ETD*Pn) for Pn being the probability of cleavage between n'th and n+1'th aminoacid.
            The probabilities have to correspond only to the breakable bonds.
        :return: float
            Logarithm of sum of squared errors.
        """
        #print "T"
        # print "Evaluating target at"
        # print PTR, ETnoD, ' '.join(ETD_list)
        # ETD_list = np.exp(logETD_list)
        if len(x) != 2 + len(precursor[0]) - 1 - len(unbreakable_bonds):  # 2 + len(breakable_bonds)
            raise ValueError("Wrong number of parameters: %i" % len(x))
        ETD, PTR, ETnoD, cleavage_probabilities = bfgs_arg_to_meanfield_arg(x, len(precursor[0]), unbreakable_bonds)
        # total_ETD = sum(ETD_list)
        # cleavage_probabilities = [ei/total_ETD for ei in ETD_list]
        v = mean_values(precursor=precursor,
                        ETD=ETD,
                        PTR=PTR,
                        ETnoD=ETnoD,
                        cleavage_probabilities=cleavage_probabilities,
                        t=1.,
                        return_sink=sink_penalty,
                        proportions=True)
        #  Including molecules predicted but not observed
        missing_keys = set(v).difference(data)
        for k in missing_keys:
            if len(k[0]) > 1:
                data[k] = 0.
        molecules = set(v) & set(data)
        # if sink_penalty:
        #     # Renormalizing proportions so that they sum up to 1 on observed molecules to fit the data
        #     observed_sum = sum(v[k] for k in v if k[1] != 0)
        #     if abs(observed_sum) > 1e-12:
        #         for k in v:
        #             v[k] /= observed_sum
        if not molecules:
            raise ValueError(
                "No molecules in the data! (Improper dictionary keys?)")
        else:
            return np.log(sum((v[m] - data[m])**2 for m in molecules if m[3] != 'z11'))
            # return sum((v[m] - data[m])**2 for m in v)
    return bfgs_target


def bfgs_callback(target):
    def cl(x):
        print "Current point (PTR, ETnoD, ETDs):"
        print ' '.join(map(str, x))
        print "Target function value:"
        print target(x)
    return cl


def fit_model_to_data(precursor,
                      breakable_bonds,
                      data,
                      ETD,
                      PTR,
                      ETnoD,
                      cleavage_probabilities,
                      mean_total_annihilation=10.,
                      sink_penalty = False,
                      verbose = False,
                      **kwargs):
    """
    Returns a dictionary of fitted parameters.
    Default reaction time is 1. To fit over time, add 't' to bounds.

    Parameters
    -----
    precursor: tuple
        The reaction precursor molecule in form (Seq, p, q).
    breakable_bonds: list
        A list of integers. An integer i means that the bond between i-th and i+1-th aminoacid can be cleaved.
    data: dict
        A dictionary indexed by observable product molecules (i.e. p != 0),
        representing a state of the sample.
        The values are the proportions of given product molecule in the sample.
    ETD, PTR, ETnoD: float
    cleavage_probabilities: list
        Starting values for the optimizing function. The list of cleavage probabilities has to sum to one,
        and the probabilities of breaking an unbreakable bond have to be zero.
    """

    unbreakable_bonds = [i for i in range(len(precursor[0]) - 1) if i not in breakable_bonds]
    assert len(set(breakable_bonds + unbreakable_bonds)) == len(precursor[0])-1, "Wrong number of breakable bonds."
    assert len(cleavage_probabilities) == len(precursor[0]) - 1, "Wrong number of cleavage probability parameters"
    for i in range(len(cleavage_probabilities)):
        assert i in breakable_bonds or cleavage_probabilities[i] == 0., "Non-zero probability of breaking an unbreakable bond."
    assert abs(sum(clpr for i, clpr in enumerate(cleavage_probabilities) if i in breakable_bonds) - 1) < 1e-12, "Probabilities of cleaving breakable bonds do not sum to 1."
    assert mean_total_annihilation >= 0., "Negative intensity of Mean Total Annihilation!"
    bfgs_target = make_bfgs_target_function(precursor, data, unbreakable_bonds, sink_penalty=sink_penalty)
    # optimizer_bounds = [(None, None) for _ in range(len(breakable_bonds)+2)]
    optimizer_bounds = [(-24, np.log(4*mean_total_annihilation)) for _ in range(len(breakable_bonds)+2)]
    if verbose:
        callback = bfgs_callback(bfgs_target)
    starting_point = [PTR, ETnoD] + [ETD*clpr for i, clpr in enumerate(cleavage_probabilities) if i not in unbreakable_bonds]
    starting_point = np.log(starting_point)
    if len(starting_point) < len(breakable_bonds) + 2:
        raise ValueError("Wrong number of parameters (too few)")
    elif len(starting_point) > len(breakable_bonds) + 2:
        raise ValueError("Wrong number of parameters (too many)")
    if verbose:
        print "Target function on starting point:"
        callback(starting_point)
    print "Optimizing."
    if verbose:
        estimates, value, info = fmin_l_bfgs_b(bfgs_target,
                                               starting_point,
                                               approx_grad=True,
                                               bounds=optimizer_bounds,
                                               callback=callback)
    else:
        estimates, value, info = fmin_l_bfgs_b(bfgs_target,
                                               starting_point,
                                               approx_grad=True,
                                               bounds=optimizer_bounds)
    if verbose:
        print "Target function on estimated intensities:"
        callback(estimates)
    ETD, PTR, ETnoD, cleavage_probabilities = bfgs_arg_to_meanfield_arg(estimates, len(precursor[0]), unbreakable_bonds)
    assert abs(sum(cleavage_probabilities) - 1) < 1e-04, "Estimated cleavage probabilities do not sum to 1!"
    return [ETD, PTR, ETnoD, cleavage_probabilities]


def print_comparison(data, prediction):
    key_set = set(data) | set(prediction)
    print "Molecule, Data, Prediction"
    for k in key_set:
        print k,
        try:
            print data[k],
        except:
            print -1,
        try:
            print prediction[k]
        except:
            print -1


if __name__=="__main__":
    # P = ('ARNWV', 2, 0)
    # P = ('*RPKPQQFFGLM*', 3, 0, "precursor")
    P = ('*MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQ', 6, 0, 'precursor')
    # P = ("MSDLKATVSETPQGFHVEGYEKIEYDFTFVDGVFDVNNPGLANCYKKWGRVLAVTDKNIF", 6, 0)
    breakable_bonds = [i for i in range(len(P[0])-1) if P[0][i+1] != 'P']
    unbreakable_bonds = [i for i in range(len(P[0])-1) if P[0][i+1] == 'P']
    print "Precursor charge: %i protons on %i basic residues" % (P[1], sum(1 for x in P[0] if x in BASIC_AA))
    print "%i breakable bonds out of %i total bonds" % (len(breakable_bonds), len(P[0])-1)
    multiple_ETD = True

    #cleavage_probabilities = [1./(len(P[0]) - 1) for _ in range(len(P[0])-1)]
    #intsy_dict = {'ETD': .2, 'PTR': .4, 'ETnoD': .4}
    intsy_dict = {'ETD': 0.057099646757, 'PTR': 0.0484388705821, 'ETnoD': 0.0840269645109}
    param_dict = {'logETD': np.log(intsy_dict['ETD']),
                  'logETnoD': np.log(intsy_dict['ETnoD']),
                  'logPTR': np.log(intsy_dict['PTR']),
                  'cleavage_probabilities': [1./len(breakable_bonds) if i in breakable_bonds else 0. for i in range(len(P[0])-1)]}
    print "Simulating the data..."
    data = simulate(P, n=10000, multiple_ETD=multiple_ETD, **intsy_dict)
    prediction = mean_values(P, **intsy_dict)
    target = make_target_function(P, data, use_null_observations=False)
    print "Actual goodness of fit:",
    print target(cleavage_probabilities = param_dict['cleavage_probabilities'], **intsy_dict)
    print ''
    # # Parameters for bayesian optimization:
    # fit_param_dict = {'init_points':10,
    #                   'n_iter': 20,
    #                   'xi': 0.0,  # xi = 0.0 => exploitation, xi = 0.1 => exploration
    #                   'acq': 'ei'}
    # print "Optimizer parameters:"
    # print fit_param_dict
    # fit = fit_model_to_data(P,
    #                         data,
    #                         **fit_param_dict)

    # bfgs_target eats a sequence which has logPTR and logETnoD on two first coordinates
    # and log(ETDn) = log(ETD) + log(Pn) on the rest of coordinates, where Pn is the n'th probability of cleavage.
    bfgs_target = make_bfgs_target_function(precursor=P, data=data, unbreakable_bonds=unbreakable_bonds)
    callback = bfgs_callback(bfgs_target)
    print "Target function on optimal point:"
    callback([param_dict['logPTR'], param_dict['logETnoD']] + [np.log(clpr)+param_dict['logETD'] for i, clpr in enumerate(param_dict['cleavage_probabilities']) if i in breakable_bonds])
    PTR = 1.
    ETnoD = 1.
    ETD = 1.
    cleavage_probabilities = [1./len(breakable_bonds) if i in breakable_bonds else 0. for i in range(len(P[0])-1)]
    ETD, PTR, ETnoD, cleavage_probabilities = fit_model_to_data(precursor=P,
                                                                breakable_bonds=breakable_bonds,
                                                                data=data,
                                                                ETD=ETD,
                                                                PTR=PTR,
                                                                ETnoD=ETnoD,
                                                                cleavage_probabilities=cleavage_probabilities,
                                                                verbose=True)


    print "Actual vs fitted parameters:"
    # for x in intsy_dict:
    #     print str(x) + ':', intsy_dict[x], 'vs', fit['max_params'][x]
    print "PTR:", intsy_dict['PTR'], 'vs', PTR
    print "ETnoD:", intsy_dict['ETnoD'], 'vs', ETnoD
    print "ETD:", intsy_dict['ETD'], 'vs', ETD
    print "Cleavage probabilities:"
    for i, clpr in enumerate(cleavage_probabilities):
        print "C%i:" % i, 1./len(breakable_bonds) if i in breakable_bonds else 0., clpr




    # print sum((intsy_dict[x] - fit['max_params'][x])**2 for x in intsy_dict))**0.5
    # print sum((intsy_dict[x] - estimates[i])**2 for i, x in enumerate(['ETD', 'PTR', 'ETnoD']))**0.5

    # n=500

    # bfgs dla zbyt malych wartosci np [0.1 0.1 0.1] zle chyba aproksymuje gradient i od razu przeskakuje pik