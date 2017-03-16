"""
A script to estimate reaction intensities from an input data file.
"""

import sys, getopt, os, time
import numpy as np
from src.model_fit import fit_model_to_data, make_target_function, make_bfgs_target_function
from src.meanfield import mean_values
from warnings import warn
import multiprocessing as mproc

def fit_thread(threadID, precursor, breakable_bonds, data, jobs, results, job_counter, print_lock):
    """
    Multiprocessing interface fot optimizing.
    :param threadID: int
    :param jobs: multiprocessing.queue
        Queue with starting points in form [PTR, ETnoD, ETD0, ETD1, ..., ETDn], where the
        coordinates are the intensities of corresponding reactions (ETDi = cleavage after i-th aminoacid).
    :param results: multiprocessing.queu
    :param job_counter: multiprocessing.value
    :param print_lock: multiprocessing.lock
    :param bounds: list
    :return: None
    """
    global bfgs_disp, I, sink_penalty
    unbreakable_bonds = [i for i in range(len(precursor[0])-1) if i not in breakable_bonds]
    bfgs_target = make_bfgs_target_function(precursor, data, unbreakable_bonds, sink_penalty=sink_penalty)
    print_lock.acquire()
    print "Thread %i started." % threadID
    print_lock.release()
    while not jobs.empty():
        try:
            starting_point = jobs.get(True, 1)
        except:
            continue
        ETD = starting_point[0]
        PTR = starting_point[1]
        ETnoD = starting_point[2]
        cleavage_probabilities = starting_point[3:]
        assert len(cleavage_probabilities) == len(precursor[0])-1, "Improper number of cleavage probabilities:" \
                                                                   "%i probabilities for %i bonds"%(
                                                                   len(cleavage_probabilities),
                                                                   len(precursor[0])-1)
        assert abs(sum(cleavage_probabilities) - 1) < 1e-04, "Cleavage probabilities do not sum to 1."
        bfgs_args = [np.log(PTR),
                     np.log(ETnoD)]+[np.log(clpr*ETD) for i, clpr in enumerate(cleavage_probabilities) if i in breakable_bonds]
        initial_target_value = bfgs_target(bfgs_args)
        print_lock.acquire()
        print "Starting optimization on thread %i" % threadID
        print "Initial intensities:"
        print starting_point
        print "Initial intensities of ETD, PTR and ETnoD:"
        print "(%f, %f, %f), " % (ETD, PTR, ETnoD)
        print_lock.release()
        try:
            ETD, PTR, ETnoD, cleavage_probabilities = fit_model_to_data(precursor,
                                                                        breakable_bonds,
                                                                        data,
                                                                        ETD,
                                                                        PTR,
                                                                        ETnoD,
                                                                        cleavage_probabilities,
                                                                        I)
            if abs(sum(cleavage_probabilities)-1) > 1e-04:
                raise RuntimeError("Optimized cleavage probabilities do not sum to 1.")
            fit = [ETD, PTR, ETnoD] + cleavage_probabilities
            bfgs_args = [np.log(PTR),
                         np.log(ETnoD)]+[np.log(clpr*ETD) for i, clpr in enumerate(cleavage_probabilities) if i in breakable_bonds]
            optimal_target_value = bfgs_target(bfgs_args)
        except Exception as e:
            print_lock.acquire()
            warn("An exception occured: %s." % e.message)
            print_lock.release()
            optimal_target_value = 'err'
        else:
            results.put(fit)
        print_lock.acquire()
        with job_counter.get_lock():
            job_counter.value += 1
            print "Thread %i finished job number %i." % (threadID, job_counter.value)
        print "Obtained results:"
        print "ETD\tPTR\tETnoD:"
        print '\t'.join(map(str, np.round([ETD, PTR, ETnoD], 6)))
        print "Cleavage probabilities:"
        print '\t'.join(str(np.round(clpr, 4)) for clpr in cleavage_probabilities)
        print "Initial target function value:"
        print initial_target_value
        print "Optimal target function value:"
        print optimal_target_value
        print_lock.release()

help_text="""NAME
    get_estimates.py

SYNOPSIS
    get_estimates.py [OPTIONS] FILE DIRECTORY

DESCRIPTION
    Estimates reaction intensities from data in FILE.
    Saves the estimates and predicted spectrum in DIRECTORY.
    If a previous estimate exists for data in FILE, it is replaced by the new one.
    FILE is a tab-separated file with columns for product subsequence, active protons, neutral protons, intensities.
    The first line has to contain the precursor molecule.

OPTIONS
    -h
        Print this message
    -d: int
        Numbers of point on initial grid per dimention (default 6)
    -t: int
        Maximum number of threads (default 4)
    -s
        Turn on penalizing for decharged molecules
    -v
        Verbose mode

"""

pts = 6
threads = 4
bfgs_disp = 0
sink_penalty = False

if len(sys.argv) == 1:
    print help_text
    quit()
else:
    opts, args = getopt.getopt(sys.argv[1:], "d:t:hsv")
    infile = args[0]
    outdir = args[1]
    for opt, arg in opts:
        if opt == "-h":
            print help_text
            quit()
        elif opt == "-d":
            pts = int(arg)
        elif opt == "-t":
            threads = int(arg)
        elif opt == "-v":
            bfgs_disp = 2
        elif opt == '-s':
            sink_penalty = True
basename = os.path.basename(infile)
basename = os.path.splitext(basename)[0]
if not os.path.isdir(outdir):
    os.mkdir(outdir)

# read the datafile
print "Dataset:", basename
with open(infile) as handle:
    data = handle.readlines()
    data = [d.split('\t') for d in data]

# check for header
try:
    int(data[0][1])
except ValueError:
    data = data[1:]

# convert to proper types
for i, d in enumerate(data):
    data[i] = [str(d[0]).upper(), int(d[1]), int(d[2]), str(d[3]), float(d[4])]

# basic data checking
# data[0][0] = '*'+data[0][0]
precursor = tuple(data[0][:4])

print "Precursor molecule and intensity:"
print precursor, data[0][4]
pr_seq_len = len(precursor[0])
breakable_bonds = [i for i in range(len(precursor[0])-1) if precursor[0][i+1] != 'P']
unbreakable_bonds = [i for i in range(len(precursor[0])-1) if precursor[0][i+1] == 'P']
assert precursor[2] == 0, "Neutralized protons on precursor."
for i, d in enumerate(data[1:]):
    assert d[1] < precursor[1], "Charge of %s equal to precursor charge" % str(d)
    assert d[0] in precursor[0], "Product sequence not a substring of precursor sequence"
    # if d[3][0] == 'c' or d[3] == 'precursor':
    #     data[i+1][0] = '*'+data[i+1][0]
    if d[3][0] == 'c':
        data[i+1][3] = 'c%i' % len(d[0])
    elif d[3][0] == 'z':
        data[i+1][3] = 'z%i' % len(d[0])

# parse the data
# substr_labels = []
# for d in data:
#     if precursor[0] == d[0]:
#         substr_labels.append("P")
#     elif precursor[0].index(d[0]) == 0:
#         substr_labels.append("L")
#     else:
#         substr_labels.append("R")
# data=dict((tuple(d[:3] + [substr_labels[i]]), d[3]) for i, d in enumerate(data))
data = dict((tuple(d[:4]), d[4]) for d in data)
print "Example product molecules and intensities:"
for i in range(5):
    print data.keys()[i], data[data.keys()[i]]
# normalize the data
norm_factor = float(sum(data[k] for k in data))
for k in data:
    data[k] /= norm_factor

# optimize
I = sum(1./i for i in range(1, precursor[1]+1))
print "Intensity of Mean Total Annihilation:"
print I


# # bayes
# bounds = {'PTR': [0, 2*I], 'ETD': [0, 2*I], 'ETnoD': [0, 2*I]}
# n_iter = 20
# init_points = 50
# acq='ei'
# xi=0.0
# fit_model_to_data(precursor, data, bounds, init_points=init_points, n_iter=n_iter, acq=acq, xi=xi)

# bfgs
results_queue = mproc.Queue()
fit_jobs = mproc.Queue()
finished_jobs = mproc.Value('i', 0)
print_lock = mproc.Lock()
for x in range(pts):
    for y in range(pts):
        for z in range(pts):
            xv = 0.34 * (x + 1.) * I / pts  # * 0.34 # optional so that the sum of parameters is at most around max_I
            yv = 0.34 * (y + 1.) * I / pts
            zv = 0.34 * (z + 1.) * I / pts
            # start_point = [PTR, ETnoD, ETD] + cleavage_probabilities

            start_point = [xv, yv, zv] + [1./len(breakable_bonds) if i in breakable_bonds else 0. for i in range(pr_seq_len-1)]
            fit_jobs.put(start_point)

print "Going to perform %i optimizations." % pts ** 3
print "Penalizing for neutralization:", sink_penalty
processes = []
for i in range(threads):
    processes.append(mproc.Process(target=fit_thread, args=(i, # threadID
                                                            precursor,
                                                            breakable_bonds,
                                                            data,
                                                            fit_jobs, # jobs queue
                                                            results_queue, # results queue
                                                            finished_jobs, # job counter
                                                            print_lock))) # printing lock

print "Starting %i threads." % threads
for i in range(threads):
    processes[i].start()

for i in range(threads):
    processes[i].join()
    print_lock.acquire()
    print "Thread %i finished." % i
    print_lock.release()
print "Optimization finished."

fit_results = []
while not results_queue.empty():
    result = results_queue.get_nowait()
    fit_results.append(result)
fit_results = np.array(fit_results)

# print "Computing means of fitted intensities from the interquartile range."
# lower_bounds, upper_bounds = np.percentile(fit_results, [25, 75], axis=0)
# in_IQR = (fit_results <= upper_bounds)*(fit_results >= lower_bounds)
# fit_means = np.array([np.mean(fit_results[in_IQR[:, i], i]) for i in range(fit_results.shape[1])])
# fit_stds = np.array([np.std(fit_results[in_IQR[:, i], i]) for i in range(fit_results.shape[1])])
# print ''
# print "Means (std's) of reaction intensity estimates after outlier removal:"
# print 'PTR\tETnoD\tETD[n]'
# print '\t'.join(("%.4f (%.4f)" % (fit_means[i], fit_stds[i]) for i in range(3)))
# print ''
# print "Computing theoretical spectrum from mean estimated intensities after outlier removal."
# total_ETD = sum(fit_means[2:])
# cleavage_probabilities = [x/total_ETD for x in fit_means[2:]]
# v = mean_values(precursor, t=1.,
#                 ETD=total_ETD,
#                 PTR=fit_means[0],
#                 ETnoD=fit_means[1],
#                 cleavage_probabilities=cleavage_probabilities)
target = make_bfgs_target_function(precursor, data, unbreakable_bonds, sink_penalty=sink_penalty)
bfgs_args = [[np.log(fit[1]), np.log(fit[2])] + [np.log(fit[0]*f) for i, f in enumerate(fit[3:]) if i in breakable_bonds] for fit in fit_results]
target_values = [target(bfarg) for bfarg in bfgs_args]
best_value = min(target_values)
print "Lowest target function value:"
print best_value
best_fit = fit_results[target_values.index(best_value)]  # totally ineffective way to do this but whatevs
print "Obtained reaction intensity estimates (ETD, PTR, ETnoD):"
print '\t'.join(map(str, best_fit[:3]))
print "Obtained cleavage probabilities:"
print '\t'.join(str(clpr) for clpr in best_fit[3:])

print "Computing theoretical spectrum from best estimates."
ETD = best_fit[0]
PTR = best_fit[1]
ETnoD = best_fit[2]
cleavage_probabilities = best_fit[3:]
v = mean_values(precursor, t=1.,
                ETD=ETD,
                PTR=PTR,
                ETnoD=ETnoD,
                cleavage_probabilities=list(cleavage_probabilities))
vs = mean_values(precursor, t=1.,
                 ETD=ETD,
                 PTR=PTR,
                 ETnoD=ETnoD,
                 cleavage_probabilities=list(cleavage_probabilities),
                 return_sink = True)
percent_neutralized = sum(vs[k] for k in vs if k[1] == 0)
print "Percentage of neutralized molecules: %.2f%%" % (100*np.round(percent_neutralized, 4))

print "Saving the predicted spectrum to:"
print os.path.join(outdir, 'predicted_spectra', basename+'_predicted.tsv')
keys = set(v) | set(data)
if not os.path.isdir(os.path.join(outdir, 'predicted_spectra')):
    os.mkdir(os.path.join(outdir, 'predicted_spectra'))
with open(os.path.join(outdir, 'predicted_spectra', basename+'_predicted.tsv'), 'w') as handle:
    handle.write('subsequence\tactive\tneutralized\tfragment_type\treal_intsy\tpredicted\n')
    for k in keys:
        try:
            v_k = v[k]
        except KeyError:
            v_k = 0.
        try:
            d_k = data[k]
        except KeyError:
            d_k = 0.
        handle.write('\t'.join(map(str, list(k)+[d_k, v_k]))+'\n')

print "Saving the optimization results to:"
print os.path.join(outdir, 'intensity_estimates', basename + '_estimates.tsv')
if not os.path.isdir(os.path.join(outdir, 'intensity_estimates')):
    os.mkdir(os.path.join(outdir, 'intensity_estimates'))
with open(os.path.join(outdir, 'intensity_estimates', basename + '_estimates.tsv'), 'w') as handle:
    handle.write('ETD\tPTR\tETnoD\t' + '\t'.join('CLPR%i' % i for i in range(pr_seq_len - 1)) + '\n')
    for fit in fit_results:
        handle.write('\t'.join(map(str, fit))+'\n')

print "Appending the estimated values to the summary file:"
print os.path.join(outdir, 'Intensity estimates.tsv')
#to_write = "\t".join([basename] + [str(l[i]) for i in range(3) for l in (fit_means, fit_stds)])
to_write = "\t".join([basename] + map(str, best_fit))
# If the file doesn't exist, create it and write a header
if not os.path.isfile(os.path.join(outdir, 'Intensity estimates.tsv')):
    with open(os.path.join(outdir, 'Intensity estimates.tsv'), 'w') as h:
        h.write("Dataset\tETD\tPTR\tETnoD\t" + '\t'.join('CLPR%i' % i for i in range(pr_seq_len - 1)) + '\n')
        h.write(to_write + '\n')
else:
    with open(os.path.join(outdir, 'Intensity estimates.tsv'), 'r') as h:
        processed_data = map(str.strip, h.readlines())
    # removing old results for the same dataset:
    processed_data = [l for l in processed_data if (l and (basename != l.split()[0]))]
    processed_data.append(to_write)
    with open(os.path.join(outdir, 'Intensity estimates.tsv'), 'w') as h:
        h.write('\n'.join(processed_data) + '\n')

