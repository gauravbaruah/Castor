import scipy.stats as st
import numpy as np
import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="computes stats for performance figures after 100 runs with random seeds")
    ap.add_argument("performance_file")
    args = ap.parse_args()

    times = np.zeros(100)
    epochs = np.zeros(100)
    maps = np.zeros(100)
    mrrs = np.zeros(100)

    with open(args.performance_file) as pf:        
        for i in range(100):
            line = pf.readline()
            time, epoch, map, mrr = line.strip().split()[1:]
            minutes, seconds = [float(r) for r in time.split(':')]
            times[i] = 60*minutes + seconds
            epochs[i] = float(epoch)
            maps[i] = float(map)
            mrrs[i] = float(mrr)

    for series in [times, epochs, maps, mrrs]:
        print('median:\t{:.5f}'.format(np.median(series)))
        print('mean:\t{:.5f}'.format(np.mean(series)))
        ci = st.t.interval(0.95, 99, loc=np.mean(series), scale=st.sem(series))
        print('conf.int:\t{:.5f}, {:.5f}'.format(ci[0], ci[1]))
