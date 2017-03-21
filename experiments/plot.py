from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


results_dir = Path('results')
results_dir.mkdir(exist_ok=True)

# Performance plot
for scale in [3, 4]:
    for test_set in ['Set5', 'Set14']:
        time = []
        psnr = []
        model = []
        for save_dir in Path('.').glob(f'*-sc{scale}'):
            if 'bicubic' not in save_dir.stem:
                model += [save_dir.stem.rsplit('-', 1)[0].upper()]
                metrics_file = save_dir / f'test/{test_set}/metrics.csv'
                metrics = pd.read_csv(str(metrics_file), index_col='name')
                time += [metrics.time.average]
                psnr += [metrics.psnr.average]
        plt.figure()
        plt.scatter(time, psnr)
        for x, y, s in zip(time, psnr, model):
            plt.text(x, y, s)
        plt.xlabel('Run Time (seconds)')
        plt.ylabel('PSNR (db)')
        plt.savefig(str(results_dir / f'performance-sc{scale}-{test_set}.png'))
        plt.close()
