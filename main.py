import os
import argparse
import datetime
import tensorflow as tf
from tool.predict_ship_LSTM import *
from tool.keras_pso.models import load_model


def Initiate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='ori_data.txt', help='input data file')
    parser.add_argument('--look_back', type=int, default=500, help='length to look back')
    parser.add_argument('--look_ahead', type=int, default=300, help='length to look ahead')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--big_small', type=str, default='big', help='big train OR small train')
    parser.add_argument('--epochs', type=int, default=1, help='train epochs')
    parser.add_argument('--num_particles', type=int, default=2, help='pso particles number')
    parser.add_argument('--psomaxiter', type=int, default=2, help='train maxiter with pso optimizer each epoch')
    parser.add_argument('--pyr', type=str, default='p', help='pitch or yaw or roll')
    parser.add_argument('--train_test', type=str, default='train', help='train or test')
    parser.add_argument('--optimizer', type=str, default='pso', help='adam sgd adagrad rmsprop adadelta pso')
    parser.add_argument('--norm', type=str, default='minmax', help='data normlization')
    parser.add_argument('--ori', type=str, default='ori', help='original or trend')
    parser.add_argument('--adam', type=bool, default=True, help='original or trend')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = Initiate()
    os.environ['CUDA_VISIBLE_DEVICES']='0,1'
    tf.device('/gpu:0,1')

    run_time = datetime.date.today()
    run_time = '2018-12-05'
    h5_file = args.big_small + 'models/' + str(run_time) + '_' + args.pyr + '_' + args.ori + '_' + args.norm + '_' + args.optimizer + '.h5'
    # if not os.path.exists(h5_file):
    #     run_time = '2018-12-04'
    #     h5_file = args.big_small + 'models/' + str(run_time) + '_' + args.pyr + '_' + args.ori + '_' + args.norm + '_' + args.optimizer + '.h5'
    # h5_pso_file = args.big_small + 'models/' + str(run_time) + '_' + args.pyr + '_' + args.ori + '_' + args.norm + '_pso.h5'
    print(h5_file)
    if not os.path.exists(args.big_small + 'models/'):
        os.makedirs(args.big_small + 'models/')
    cutoff1 = 77000
    cutoff2 = 44000

    data_preprocess(args.pyr, args.ori, args.norm, args.data_file, args.look_ahead, args.look_back, cutoff1, cutoff2)
    model = build_model(args.look_ahead, args.look_back)
    model.summary()
    if args.train_test == 'train':
        # if args.optimizer=='adam':
        #     model, loss = train_optimize(pyr=args.pyr, model=model, batch_size=args.batch_size, h5_file=h5_file, epochs=args.epochs, optimizer=args.optimizer)
        #     hist = optimize_model(err_best_g=loss, num_particles=args.num_particles, psomaxiter=args.psomaxiter, epochs=args.epochs, batch_size=args.batch_size, model=model, h5_file=h5_pso_file, optimize=True, adam=False)
        # else:
        #     model, loss = train_optimize(pyr=args.pyr, model=model, batch_size=args.batch_size, h5_file=h5_file, epochs=args.epochs, optimizer=args.optimizer)
        #     # hist = optimize_model(err_best_g=loss, num_particles=args.num_particles, psomaxiter=args.psomaxiter, epochs=args.epochs, batch_size=args.batch_size, model=model, h5_file=h5_pso_file, optimize=True, adam=False)
        if args.adam==True:
            adam = True
        else:
            adam=False
        model, loss = train_optimize(pyr=args.pyr, model=model, batch_size=args.batch_size, h5_file=h5_file, epochs=args.epochs, optimizer=args.optimizer, adam=adam, num_particles=args.num_particles, psomaxiter=args.psomaxiter)
    else:
        model = load_model(h5_file)
        pred = test_model(args.pyr, model, args.norm, args.ori, args.optimizer)
        print(pred)
