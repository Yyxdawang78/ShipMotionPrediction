import os
import multiprocessing


def pro_run(run_order):
    os.system(run_order)


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=1)
    train_or_test = ['test']#, 'train']
    poses = ['z-s']#,'y' , 'p', 'r', 'x-s', 'y-s'
    ori_or_trend = ['ori', 'residual']#]
    norms = ['minmax', 'no', 'z-score']#
    optimizers = ['pso', 'adagrad', 'rmsprop', 'adadelta', 'adam', 'sgd']#]

    for train_test in train_or_test:
        for pos in poses:
            for ori_residual in ori_or_trend:
                for norm in norms:
                    for optimizer in optimizers:
                        data_file = ori_residual + '_' + norm +'.txt'
                        run_order = 'python main.py --train_test ' + train_test + ' --pyr ' + pos + ' --ori ' + ori_residual +  ' --data_file ' + data_file +' --norm ' + norm + ' --optimizer ' + optimizer# + ' &'
                        pool.apply_async(pro_run, (run_order, ))
    pool.close()
    pool.join()
    print('process run success')
