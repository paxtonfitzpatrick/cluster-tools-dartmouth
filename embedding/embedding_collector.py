import os
import sys
import pickle
from shutil import copy2
from time import sleep, localtime, asctime
import numpy as np
from os.path import join as opj
from scipy.spatial.distance import pdist, cdist
from embedding_config import config

try:
    from tqdm import trange
    range_ = trange
except ModuleNotFoundError:
    range_ = range

passed_args = sys.argv[1:]

if not passed_args:
    n_seeds = 5000
    orig_order = False
elif passed_args[0].isdigit():
    n_seeds = int(passed_args.pop(0))

else:
    n_seeds = 5000

if passed_args and 'orig' in passed_args[0]:
    orig_order = True
else:
    orig_order = False

print(f'searching {n_seeds} seeds...')

events_dir = opj(config['datadir'], 'events', 'episodes')

if orig_order:
    embeddings_dir = opj(config['datadir'], 'embeddings_orig_order')
    figures_dir = opj(config['datadir'], 'figures_orig_order')
    optimized_dir = opj(config['datadir'], 'optimized_orig_order')
else:
    embeddings_dir = opj(config['datadir'], 'embeddings')
    figures_dir = opj(config['datadir'], 'figures')
    optimized_dir = opj(config['datadir'], 'optimized')

print(f'searching {embeddings_dir}...')

if not os.path.isdir(optimized_dir):
    os.mkdir(optimized_dir)

# Define some functions
##################################################


def r2z(r):
    with np.errstate(invalid='ignore', divide='ignore'):
        return 0.5 * (np.log(1 + r) - np.log(1 - r))


# def spatial_similarity(embedding, original_pdist, emb_metric, z_transform=False):
#     emb_pdist = pdist(embedding, emb_metric)
#     if emb_metric == 'correlation':
#         emb_pdist = 1 - emb_pdist
#         if z_transform:
#             emb_pdist = r2z(emb_pdist)
#             original_pdist = r2z(original_pdist)
#     return 1 - pdist((emb_pdist, original_pdist), 'correlation')[0]


def spatial_similarity(embedding, original_pdist, emb_metric):
    """
    computes correlation between pairwise euclidean distance in embedding space
    and correlation distance in original space
    """
    emb_pdist = pdist(embedding, emb_metric)
    if emb_metric == 'correlation':
        emb_pdist = 1 - emb_pdist
    return 1 - pdist((emb_pdist, original_pdist), 'correlation')[0]


# source: https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
def _segments_intersect2d(a1, b1, a2, b2):
    s1 = b1 - a1
    s2 = b2 - a2

    s = (-s1[1] * (a1[0] - a2[0]) + s1[0] * (a1[1] - a2[1])) / (-s2[0] * s1[1]
                                                                + s1[0] * s2[1])
    t = (s2[0] * (a1[1] - a2[1]) - s2[1] * (a1[0] - a2[0])) / (-s2[0] * s1[1]
                                                               + s1[0] * s2[1])

    if (s >= 0) and (s <= 1) and (t >= 0) and (t <= 1):
        return True
    else:
        return False


def n_intersections(x):
    intersections = 0
    for i in np.arange(x.shape[0] - 1):
        a1 = x[i, :]
        b1 = x[i + 1, :]
        for j in np.arange(i + 2, x.shape[0] - 1):
            a2 = x[j, :]
            b2 = x[j + 1, :]

            if _segments_intersect2d(a1, b1, a2, b2):
                intersections += 1
    return intersections


def dispersion_dist(emb):
    # normalize to fit in unit square
    scaled = ((emb - emb.min(0))*2 / (emb.max(0) - emb.min(0))) - 1
    # center = np.array([.5, .5])
    avg_point = scaled.mean(0)
    return cdist(np.atleast_2d(avg_point), scaled, 'euclidean').mean()


##################################################

rectypes = ['atlep1', 'delayed', 'atlep2', 'arrdev']

ep_events_pdists = {}
for rectype in rectypes:
    ep = 'atlep1' if rectype == 'delayed' else rectype
    ep_events = np.load(opj(events_dir, f'{ep}_events.npy'))
    ep_events_pdists[rectype] = 1 - pdist(ep_events, 'correlation')

# wait for jobs to finish
# ready = False
# while not ready:
#     ready = True
#     if any(len(os.listdir(opj(embeddings_dir, rt))) < n_seeds for rt in rectypes):
#         print('Waiting for jobs to finish...')
#         ready = False
#         sleep(300)

print(f'Started: {asctime(localtime())}')
results = {rectype: {} for rectype in ep_events_pdists.keys()}
for rectype, res in results.items():
    emb_rtdir = opj(embeddings_dir, rectype)
    print(f'optimizing {rectype}...')
    dispersion = np.full((n_seeds,), np.nan)
    intersections = np.full((n_seeds,), np.nan)
    similarity_euc = np.full((n_seeds,), np.nan)
    similarity_corr = np.full((n_seeds,), np.nan)

    for np_seed in range_(n_seeds):
        if orig_order:
            f_name = str(np_seed)
        else:
            f_name = f'np{np_seed}_umap{0}'
        fpath = opj(emb_rtdir, f'{f_name}.p')
        try:
            with open(fpath, 'rb') as f:
                ep_emb = pickle.load(f)['episode']

        except FileNotFoundError:
            print(f'File not found: {rectype}/{f_name}')
            continue

        dispersion[np_seed] = dispersion_dist(ep_emb)
        intersections[np_seed] = n_intersections(ep_emb)
        similarity_euc[np_seed] = spatial_similarity(ep_emb,
                                                     ep_events_pdists[rectype],
                                                     'euclidean')
        similarity_corr[np_seed] = spatial_similarity(ep_emb,
                                                      ep_events_pdists[rectype],
                                                      'correlation')

    # print(f'{rectype}_dispersion:', np.where(np.isnan(dispersion)))
    # print(f'{rectype}_intersections:', np.where(np.isnan(intersections)))
    # print(f'{rectype}_similarity_euc:', np.where(np.isnan(similarity_euc)))
    # print(f'{rectype}_similarity_corr:', np.where(np.isnan(similarity_corr)))
    np.save(opj(optimized_dir, f'{rectype}_dispersion.npy'), dispersion)
    np.save(opj(optimized_dir, f'{rectype}_intersections.npy'), intersections)
    np.save(opj(optimized_dir, f'{rectype}_similarity_euc.npy'), similarity_euc)
    np.save(opj(optimized_dir, f'{rectype}_similarity_corr.npy'), similarity_corr)

#
#     dist_np = np.where(dispersion > np.nanpercentile(dispersion, 90))[0]
#     res['dispersion'] = dist_np
#
#     intersect_np = np.where(intersections < np.nanpercentile(intersections, 90))[0]
#     res['intersections'] = intersect_np
#
#     sim_euc_np = np.where(similarity_euc > np.nanpercentile(similarity_euc, 90))[0]
#     res['similarity_euc'] = sim_euc_np
#
#     sim_corr_np = np.where(similarity_corr > np.nanpercentile(similarity_corr, 90))[0]
#     res['similarity_corr'] = sim_corr_np
#
# with open(opj(optimized_dir, 'optimal_seeds.p'), 'wb') as f:
#     pickle.dump(results, f)
#
# for rectype, opts in results.items():
#     emb_dir = opj(embeddings_dir, rectype)
#     fig_dir = opj(figures_dir, rectype)
#
#     for seeds in opts.values():
#         for seed in seeds:
#             if orig_order:
#                 fname = str(seed)
#             else:
#                 fname = f'np{seed}_umap0'
#             emb_src = opj(emb_dir, f'{fname}.p')
#             emb_dest = opj(optimized_dir, f'{rectype}_{seed}.p')
#             fig_src = opj(fig_dir, f'{fname}.pdf')
#             fig_dest = opj(optimized_dir, f'{rectype}_{seed}.pdf')
#             try:
#                 copy2(emb_src, emb_dest)
#             except FileNotFoundError:
#                 print(f'File not found: {rectype}/{fname}')
#                 pass
#             try:
#                 copy2(fig_src, fig_dest)
#             except FileNotFoundError:
#                 print(f'File not found: {rectype}/{fname}')
#                 pass

print(f'Ended: {asctime(localtime())}')
