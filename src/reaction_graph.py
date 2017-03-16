# -*- coding: utf-8 -*-
"""
Functions to create and traverse the reaction graph of ETD, i.e. relations between molecules
(ancestor/daughter molecule) and intensities of reactions leading from ancestor to daughter.
"""
from shared import SINK, CHARGE_RADIUS, P, BASIC_AA, CHARGE_FACTOR
import sympy
from scipy.stats import binom
#import graph_tool.all as gt


def hypergeom_pmf_iterator(N, k, n):
    """
    Iterator over values of the hypergeometric probability mass function.
    The subsequent values represent the probabilities over the support, i.e.
    the set Cx = {max(0, n-f), ..., min(k, n)} where f = N - k is the number of failures in the population.
    :param N: int
        Population size
    :param k: int
        Number of successes in population
    :param n:
        Number of trials.
    :return: float
    """
    # Initialization
    f = N - k
    prod = lambda sequence: reduce(lambda x, y: x*y, sequence, 1.)  # empty sequence returns 1.
                                                                    # This happens e.g. when n = 0.
    # Fairly efficient computation of the probability of "full failure":
    if n <= f:
        pr = prod((float(f-i)/(N-i) for i in range(n)))
    else:
        pr = prod((float(n-i)/(N-i) for i in range(f)))
    yield pr
    for x in range(max(0, n-f)+1, min(k, n)+1):
        pr *= (k-x+1.)/x
        pr *= (n-x+1.)/(f-n+x)
        yield pr


def iterate_over_ptr_probabilities(p1, q1, p, q, p0):
    """
    Helper function to iterate over probabilities of numbers of PTR reactions on a sequence fragment.
    The i-th value is the probability that there were i - 1 - p1 - q1 reactions on the given fragment (represented
    as the product molecule). Counting starts from i = p1 + q1 + 1.
    :param p1: int
    :param q1: int
        Charged and quenched protons on the product
    :param p: int
    :param q: int
        On the substrate
    :param p0: int
        Initial precursor charging
    :return: tuple
        Tuple (Ai, Bi), where Bi is the i-th probability, and Ai is the multiplier to get i-th probability
        from i-1-th value (so that B(i+1) = Ai*Bi).
    """
    # p0 = precursor[1]
    # p = substrate[1]
    # q = substrate[2]
    # p1 = product[1]
    # q1 = product[2]
    p2 = p - p1
    q2 = q - q1
    # Initialisation: i = p1 + q1 + 1
    A = 1.
    B = reduce(lambda x, y: x*y, (float(p-i)/(p0-q-i) for i in range(p1)))
    # yield (A, B)
    for x in range(p1 + q1 + 1, p0 - p2 - q2 + 2):
        # B = B(x)
        # A = A(x)
        B *= A
        A = (1. + float(p1) / (x - q1 - p1)) * (1. - float(p2) / (p0 - x - q2 + 1.))
        yield (A, B)


def proton_distribution_probability(product, substrate, precursor):
    """
    WRONG!!!

    Returns the probability of observing given charge on the product after ETD reaction given substrate and precursor.
    :param product: tuple
    :param substrate: tuple
    :param precursor: tuple
        Respective molecules, represented as (Seq, p, q, label).
    :return: float
    """
    global BASIC_AA
    p = substrate[1]
    q = substrate[2]
    p1 = product[1]
    p2 = p - p1
    q1 = product[2]
    q2 = q - q1
    p0 = precursor[1]
    nu = len(substrate[0])
    nu1 = len(product[0])
    #nu = sum(aa in BASIC_AA for aa in substrate[0])
    #nu1 = sum(aa in BASIC_AA for aa in product[0])
    nu2 = nu - nu1
    # Distribution over initial charging on product sequence:
    # Hypergeometric with population nu, successes nu1, trials p0
    XA1 = hypergeom_pmf_iterator(N=nu, k=nu1, n=p0)
    XA1_0 = max(0, p0-nu2)
    PTR = iterate_over_ptr_probabilities(p1=p1, q1=q1, p=p, q=q, p0=p0)
    PTR_0 = p1+q1+1
    kmin = max(XA1_0, PTR_0)
    kmax = min(nu1, p0-p2-q2+1)
    # Moving past the "null" summands
    for i in range(max(0, kmin - PTR_0 - 1)):
        PTR.next()
    for i in range(max(0, kmin - XA1_0 - 1)):
        XA1.next()
    pr = 0.
    for i in range(kmin, kmax+1):
        A, B = PTR.next()
        C = XA1.next()
        pr += (float(A)*float(nu2)/nu + float(nu1)/nu)*B*C
    #pr = sum((float(A)*float(nu2)/nu + float(nu1)/nu)*B*C for i, (A, B), C in enumerate(zip(PTR, XA1)) if i + kmin < kmax)
    return pr


def INTSY(type, product, substrate, precursor, intensity_dict, cleavage_probabilities=None):
    """
    Returns the intensity that a substrate undergoes a given type of reaction, yielding
    a given product, in an ETD reaction that starts from the given precursor.
    Note that the precursor is not necessarily the direct ancestor of the product or the molecule.

    Parameters
    -----
    type: str
        Name of the reaction (ETD, ETnoD or PTR).
    product: tuple
        Product molecule. Represented as a length 3 tuple (Seq, p, q, P), where Seq is the product sequence,
        p is the number of charged protons, q is the number of quenched protons (gained in ETnoD) and
        P is the subsequence position ("L" for left fragments, "R" for right fragments,
        "P" for precursor sequence).
    substrate: tuple
        Reaction substrate molecule.
    precursor: tuple
        The fragmentation precursor molecule. It can't have neutralized electrons. Has to start with a '*' symbol,
        representing the N-terminal amino group.
    intensity_dict: dict
        Dictionary of reaction intensities. The ETD entry is the total intensity of ETD,
        regardless where the cleavage occurs (this is specified by cleavage_probabilities).
    cleavage_probabilities: iterable
        An iterable of length one shorter of the precursor sequence. Has to sum to 1.
        The i-th value is the probability that a cleavage occurs between i-th and i+1-th residue, given that
        ETD reaction occurs. This variable needs to be specified only if type == "ETD".
    """
    global CHARGE_RADIUS, CHARGE_FACTOR
    PTR = intensity_dict['PTR']
    ETnoD = intensity_dict['ETnoD']
    ETD = intensity_dict['ETD']
    charge_factor = CHARGE_FACTOR(substrate[1])
    substrate_bases = sum(a in BASIC_AA for a in substrate[0])
    # Note that even if ETD occurs, the sequences of substrate and product might be equal (loss of amino group)
    if substrate[0] == precursor[0] and substrate[3] != product[3]:
        # ETD or HTR
        product_len = len(product[0])
        if type == 'ETD':
            assert product[3] != precursor, "ETD product labelled as precursor"
            if 'c' in product[3]:
                assert product[0] == substrate[0][:product_len], \
                    "Product sequence labeled as c-ion, but is not a proper prefix of substrate"
                # c-ion, cleavage between len(product[0]) - 1 and len(product[0])
                cleavage_prob_factor = cleavage_probabilities[product_len-1]
            elif 'z' in product[3]:
                assert product[0] == substrate[0][-product_len:], \
                    "Product sequence labeled as z-ion, but is not a proper suffix of substrate"
                # z-ion, cleavage between -len(product[0]) - 1 and -len(product[0])
                cleavage_prob_factor = cleavage_probabilities[-product_len]
            else:
                raise ValueError("Product sequence is not a suffix nor a prefix of substrate sequence.")
            product_bases = sum(a in BASIC_AA for a in product[0])
            other_bases = substrate_bases - product_bases
            min_charged = max(0, substrate[1]-1-other_bases)
            max_charged = min(product_bases, substrate[1]-1)
            min_quenched = max(0, substrate[1]-product[1]+substrate[2]-1-other_bases)
            max_quenched = min(product_bases-product[1], substrate[2])
            if product[1] < min_charged or product[2] < min_quenched:
                return 0.
            elif product[1] > max_charged or product[2] > max_quenched:
                return 0.
            else:
                charged_iterator = hypergeom_pmf_iterator(N=substrate_bases,
                                                          k=product_bases,
                                                          n=substrate[1] - 1)
                quenched_given_charged_iterator = hypergeom_pmf_iterator(N=substrate_bases - substrate[1] + 1,
                                                                         k=product_bases - product[1],
                                                                         n=substrate[2])
                charged_prob = charged_iterator.next()
                quenched_prob = quenched_given_charged_iterator.next()
                # Moving to the proper probability value:
                # Initially charged_prob = P(charged = min_charged)
                for i in range(product[1] - min_charged):
                    charged_prob = charged_iterator.next()
                for i in range(product[2] - min_quenched):
                    quenched_prob = quenched_given_charged_iterator.next()
                proton_prob_factor = charged_prob*quenched_prob
                return float(charge_factor) * ETD * cleavage_prob_factor * proton_prob_factor
        else:
            raise ValueError("Unknown reaction (code 1)")
    elif substrate[3] == product[3]:
        if type == 'PTR':
            return charge_factor * PTR
        elif type == 'ETnoD':
            return charge_factor * ETnoD
        else:
            raise ValueError("Unknown reaction (code 2)")
    elif substrate[0] != precursor[0] and substrate[0] != product[0]:
        raise ValueError("ETD can't occur twice!")


def get_ancestors(product, precursor, intensity_dict, cleavage_probabilities):
    """
    Returns ancestor molecules of a given product
    and corresponding reaction intensities.

    Parameters
    ----------
    product: tuple
        Product molecule. Represented as a length 3 tuple (Seq, p, q, P), where Seq is the product sequence,
        p is the number of charged protons, q is the number of quenched protons (gained in ETnoD) and
        P is the ion type (cn, zn or precursor, where n is the ion length).
    precursor: tuple
        Precursor molecule. Has to start with a '*' symbol,
        representing the N-terminal amino group.
    intensity_dict: dict
        Dictionary of reaction intensities. The ETD entry is the total intensity of ETD,
        regardless where the cleavage occurs (this is specified by cleavage_probabilities).
    cleavage_probabilities: iterable
        An iterable of length one shorter than the precursor sequence. Has to sum to 1.
        The i-th value is the probability that a cleavage occurs between i-th and i+1-th residue, given that
        ETD reaction occurs.

    Returns
    ---------
    out: tuple of lists
        Returns two lists. The first one contains the ancestor molecules,
        i.e. molecules with one more proton (PTR ancestor) or one less
        electron (ETnoD ancestor) or with sequence of the precursor
        molecule (ETD ancestor). The ETD ancestors contain any number
        of protons which is greater or equal than that of the product
        molecule and less or equal than that of the precursor molecule.
        The second list contains corresponding reaction intensities.
        If product == SINK, then all the precursor sequences' prefixes
        and suffixes with charge p - q = +1 are returned.

    Details
    ----------
        -
    """
    global SINK, INTSY, CHARGE_RADIUS
    precursor_bases = sum(a in BASIC_AA for a in precursor[0])
    product_bases = sum(a in BASIC_AA for a in product[0])
    dual_bases = precursor_bases - product_bases
    if product[0] not in precursor[0]:
        raise ValueError("Sequence of product is not a substring of precursor.")
    if precursor[2] != 0:
        raise ValueError("Precursor can't have neutralized protons (q)")
    if product[1] > precursor[1]:
        raise ValueError("Product ion can't have higher charge (p) than precursor")
    if product[1] + product[2] > precursor[1] + precursor[2]:
        raise ValueError("Product ion can't have more protons (p + q) than precursor")
    # elif product[1] > 1 + len(product[0])/CHARGE_RADIUS:
    #     raise ValueError("Overcharged product (max number of protons = %i)" % (1 + len(precursor[0])/CHARGE_RADIUS))
    if precursor[1] > precursor_bases:
        raise ValueError("Overcharged precursor (%i charges on %i possible sites)" % (precursor[1], precursor_bases))
    if product[1] + product[2] > product_bases:
        raise ValueError("Overcharged product (%i charges on %i possible sites)"%(product[1] + product[2], product_bases))
    if '*' not in precursor[0]:
        raise ValueError("No N-terminal amino group ('*') in the precursor sequence.")
    if precursor[3] != 'precursor':
        raise ValueError("Precursor not labeled as precursor.")
    if 'c' == product[3][0] and len(product[0]) != int(product[3][1:]):
        raise ValueError("Improper labelling of c-ion: %i aminoacids labeled as %s" % (len(product[0]), product[3]))
    if 'z' == product[3][0] and len(product[0]) != int(product[3][1:]):
        raise ValueError("Improper labelling of z-ion: %i aminoacids labeled as %s" % (len(product[0]), product[3]))
    if product[3] == 'precursor':
        assert product[0] == precursor[0], "Product sequence labeled as precursor molecule but sequences differ."
    ancestors = []
    intensities = []
    # PTR = intensity_dict['PTR']
    # ETnoD = intensity_dict['ETnoD']
    # ETD = intensity_dict['ETD']
    prec_len = len(precursor[0])

    # sink has to be treated separately, because it returns all the
    # suffixes and prefixes
    if product == SINK:
        # SINK is a pseudo-node with 0 inflow.
        # Ancestors of SINK are all the molecules with charge 0.
        # The number of quenched protons on a fragment has to be strictly smaller than the precursor charge,
        # because each fragment is an ETD reaction product.
        # PREFIXES:
        for i in xrange(1, prec_len):
            substrate_bases = sum(a in BASIC_AA for a in precursor[0][:i])
            for q in xrange(0, min(substrate_bases, precursor[1])):
                substrate = (precursor[0][:i], 0, q, "c%i" % i)
                ancestors.append(substrate)
                intensities.append(0.)
        # SUFFIXES:
        for i in xrange(1, prec_len):
            substrate_bases = sum(a in BASIC_AA for a in precursor[0][-i:])
            for q in xrange(0, min(precursor[1], substrate_bases)):
                substrate = (precursor[0][-i:], 0, q, "z%i" % i)
                ancestors.append(substrate)
                intensities.append(0.)
        # PRECURSOR:
        for q in xrange(0, min(precursor[1]+1, precursor_bases+1)):
            substrate = (precursor[0], 0, q, "precursor")
            ancestors.append(substrate)
            intensities.append(0.)

    else: # product != SINK
        # ETD could occur only if product and precursor have different labels
        # Note that they can still have same sequences because of loss of N-terminal amino group
        if product[3] != precursor[3]:
            pmin = product[1] + 1  # minimal substrate charge
            pmax = min(precursor[1]-product[2],  # quenched on product had to be quenched on substrate
                       product[1] + dual_bases + 1,  # no overcharging of paired product ion
                       precursor_bases)  # no overcharging of the full sequence
            qmin = product[2]
            qmax = min(precursor[1] - product[1] - 1,  # charged had to be charged, plus reacting proton
                       product[2] + dual_bases)        # no overcharging of paired ion

            for p in range(pmin, pmax+1):
                for q in range(qmin, min(precursor[1] - p + 1, qmax + 1)):
                    substrate = (precursor[0], p, q, "precursor")
                    etd_intsy = INTSY('ETD', product, substrate, precursor, intensity_dict, cleavage_probabilities)
                    ancestors.append(substrate)
                    intensities.append(etd_intsy)

        # PTR occurs if the number of protons on the product is smaller than the one
        # on the precursor and there is place for another charge on the substrate.
        # If the substrate is a fragment, then there has to be additional charge availiable for ETD to occur
        # higher on the graph (a fragment can't take all the charges).
        if (product[3] == precursor[3] and product[1] + product[2] < min(precursor[1], product_bases)) or \
            (product[3] != precursor[3] and product[1] + product[2] < min(precursor[1] - 1, product_bases)):
            substrate = (product[0], product[1] + 1, product[2], product[3])
            ptr_intsy = INTSY('PTR', product, substrate, precursor, intensity_dict)
            ancestors.append(substrate)
            intensities.append(ptr_intsy)

        # ETnoD on a molecule could always happen if it has quenched protons
        if product[2] > 0:
            substrate = (product[0], product[1] + 1, product[2] - 1, product[3])
            etnod_intsy = INTSY('ETnoD', product, substrate, precursor, intensity_dict)
            ancestors.append(substrate)
            intensities.append(etnod_intsy)
    return (ancestors, intensities)


def create_molecule_graph(precursor):
    """Creates the reaction product graph. This is a graph of molecules, not the reaction graph,
    which means that there is an edge from X to Y iff Y is a product and X is a substrate of a reaction.
    A reaction graph, on the other hand, should contain molecules and transitions as vertices,
    since one molecule can give several products in a single reaction."""
    global SINK
    ancestor_dict = {}
    idict = {'PTR': 1, 'ETnoD': 1, 'ETD': 1}
    def update_ancestor_dict(x, d):
        """closure to create a dictionary of ancestors of x
        d = dict of ancestors
        i = dict of intensities"""
        if x == precursor:
            pass
        elif x in d:
            pass
        else:
            anc, intsy = get_ancestors(x, precursor, idict, [1./(len(precursor[0]))]*len(precursor[0]))
            d[x] = anc
            for a in anc:
                update_ancestor_dict(a, d)
    update_ancestor_dict(SINK, ancestor_dict)
    # dopisac cos w stylu
    # for a in ancestor_dict:
    #   dodaj wezel(a)
    # for a, k in anc_dict.iteritems()
    #   dodaj krawedz(k -> a)
    return ancestor_dict


def test_hypergeom():
    """
    Compare mine with scipy.stats.
    :return:
    """
    from scipy.stats import hypergeom
    import time
    # M = populacja, N = liczba prób, n = liczba sukcesów, k = punkt
    pop = 100
    trials = 50
    successes = 30
    failures = pop - successes
    kmin = max(0, trials - failures)
    kmax = min(trials, successes)
    ss_list = [hypergeom.pmf(k=k, N=trials, M=pop, n=successes) for k in range(kmin, kmax + 1)]
    my_list = list(hypergeom_pmf_iterator(N=pop, k=successes, n=trials))
    expect = trials * float(successes) / pop
    iterations = 100
    ss_results = [0 for _ in range(iterations)]
    ss_start_time = time.time()
    for iter in range(iterations):
        ss_expect = sum(i * hypergeom.pmf(k=i, N=trials, M=pop, n=successes) for i in range(kmin, kmax + 1))
        ss_results[iter] = ss_expect
    ss_end_time = time.time()
    ss_time = ss_end_time - ss_start_time
    my_results = [0 for _ in range(iterations)]
    my_start_time = time.time()
    for iter in range(iterations):
        my_expect = sum((i + kmin) * pr for i, pr in enumerate(hypergeom_pmf_iterator(N=pop, k=successes, n=trials)))
        my_results[iter] = my_expect
    my_end_time = time.time()
    my_time = my_end_time - my_start_time
    print "Average scipy error:"
    print sum(abs(r - expect) for r in ss_results) / iterations
    print "Average my implementation error:"
    print sum(abs(r - expect) for r in my_results) / iterations
    print "Scipy execution time:"
    print ss_time
    print "My implementation execution time:"
    print my_time


def test_ptr_iterator():
    from scipy.stats import hypergeom
    p1 = 3
    q1 = 2
    p = 3
    q = 2
    p2 = p - p1
    q2 = q - q1
    p0 = 8
    # M = populacja, N = liczba prób, n = liczba sukcesów, k = punkt
    ss_ptr = [hypergeom.pmf(k=p1, M=p0-q, N=p, n=x-1-q1) for x in range(p1+q1+1, p0-p2-q2+2)]
    my_ptr = list(iterate_over_ptr_probabilities(p=p, q=q, p1=p1, q1=q1, p0=p0))
    print ss_ptr
    print my_ptr


if __name__ == "__main__":
    # ptr, etnod, etd, x0, t = sympy.symbols('ptr etnod etd x0 t')
    # P = ('*RPKPQQFFGLM', 3, 0, "precursor")
    # intsy_dict = {'PTR': ptr, 'ETnoD': etnod, 'ETD': etd}
    # get_ancestors(('WK', 1, 0, 'z2'), ('*ARNWK', 3, 0, 'precursor'), intsy_dict, [1./5]*5)
    # get_ancestors(('*RPKPQQFFGLM', 1, 0, "precursor"),  ('*RPKPQQFFGLM', 3, 0, "precursor"), intsy_dict, [1./(len(P[0])-1)]*(len(P[0])-1))
    # a = create_molecule_graph(P)
    # #test_hypergeom()
    # #test_ptr_iterator()
    # P = ("*ARNWV", 3, 0)
    # S = ("ARNWV", 2, 1)
    # p = ("ARN", 1, 0)
    # print proton_distribution_probability(p, S, P)

    intsy_dict = {'ETD':0.092827067677180949, 'PTR':0.0030914034270024767, 'ETnoD': 0.0034352983680447685}
    cleavage_probabilities = [ 0.13469298,  0.,  0.14727094,  0.,  0.07671245,
                               0.12521983,  0.09810842,  0.08939107,  0.07087194,
                               0.14621228,  0.06916717,  0.04235293]
    prod = ('RPKPQQFFGLM*', 1, 0, 'z12')
    prec = ('*RPKPQQFFGLM*', 3, 0, 'precursor')
    get_ancestors(prod, prec, intsy_dict, cleavage_probabilities)
