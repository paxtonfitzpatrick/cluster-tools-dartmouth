import os
from os.path import join as opj
from embedding_config import config

try:
    from tqdm import trange
    range_ = trange
except ModuleNotFoundError:
    range_ = range

embedding_dir = opj(config['datadir'], 'embeddings')
figure_dir = opj(config['datadir'], 'figures')
rectypes = ['atlep1', 'delayed', 'atlep2', 'arrdev']

for rectype in rectypes:
    print(f'cleaning up {rectype}')
    emb_dir = opj(embedding_dir, rectype)
    fig_dir = opj(figure_dir, rectype)
    embs = sorted(os.listdir(emb_dir))
    figs = sorted(os.listdir(fig_dir))

    for np_seed in range_(200):
        seedembs = [f for f in embs if f.startswith(f'np{np_seed}_umap')]
        seedfigs = [f for f in figs if f.startswith(f'np{np_seed}_umap')]
        if len(seedembs) > 1:
            print(f'saving {seedembs[0]}')
            for emb in seedembs[1:]:
                os.remove(opj(emb_dir, emb))
        if len(seedfigs) > 1:
            print(f'saving {seedfigs[0]}')
            for fig in seedfigs[1:]:
                os.remove(opj(fig_dir, fig))
