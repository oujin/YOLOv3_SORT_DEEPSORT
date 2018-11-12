from sort import *


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument(
        '--display',
        dest='display',
        help='Display online tracker output (slow) [False]',
        action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # all train
    sequences = [
        'PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'ETH-Bahnhof',
        'ETH-Sunnyday', 'ETH-Pedcross2', 'KITTI-13', 'KITTI-17', 'Venice-2',
        'ADL-Rundle-6', 'ADL-Rundle-8'
    ]
    args = parse_args()
    display = args.display
    phase = 'train'
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3)  # used only for display
    if (display):
        if not os.path.exists('yolo_v3/mot_benchmark'):
            print('''
                  ERROR: mot_benchmark link not found!

                  Create a symbolic link to the MOT benchmark
                  (https://motchallenge.net/data/2D_MOT_2015/#download).
                E.g.:

                  $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark

                ''')
            exit()
        plt.ion()
        fig = plt.figure()

    if not os.path.exists('sort/results'):
        os.makedirs('sort/results')

    for seq in sequences:
        mot_tracker = Sort()  # create instance of the SORT tracker
        seq_dets = np.loadtxt(
            'sort/data/%s/det.txt' % (seq), delimiter=',')  # load detections
        with open('sort/results/{}.txt'.format(seq), 'w') as out_file:
            print("Processing {}.".format(seq))
            for frame in range(int(seq_dets[:, 0].max())):
                frame += 1  # detection and frame numbers begin at 1
                dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                dets[:, 2:4] += dets[:, 0:2]
                total_frames += 1

                if (display):
                    ax1 = fig.add_subplot(111, aspect='equal')
                    fn = 'yolo_v3/mot_benchmark/{}/{}/img1/{}.jpg'.format(
                        phase, seq,
                        str(frame).zfill(6))
                    im = io.imread(fn)
                    ax1.imshow(im)
                    plt.title(seq + ' Tracked Targets')

                start_time = time.time()
                trackers = mot_tracker.update(dets)
                cycle_time = time.time() - start_time
                total_time += cycle_time

                for d in trackers:
                    print(
                        '{:d},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},1,-1,-1,-1'.
                        format(frame, d[4], d[0], d[1], d[2] - d[0],
                               d[3] - d[1]),
                        file=out_file)
                    if (display):
                        d = d.astype(np.int32)
                        ax1.add_patch(
                            patches.Rectangle((d[0], d[1]),
                                              d[2] - d[0],
                                              d[3] - d[1],
                                              fill=False,
                                              lw=3,
                                              ec=colours[d[4] % 32, :]))
                        ax1.set_adjustable('box-forced')

                if (display):
                    fig.canvas.flush_events()
                    plt.draw()
                    ax1.cla()

    print("Total Tracking took: {:.3f} for {:d} frames or {:.1f} FPS".format(
        total_time, total_frames, total_frames / total_time))
    if (display):
        print("Note: to get real runtime results run"
              "without the option: --display")
