'''
This is the main code of the ECAPATDNN project, to define the parameters and build the construction
'''

import argparse, glob, os, torch, warnings, time

import yaml
from torch.utils.data import DataLoader

from NASModel import NASModel
from tools import *
from dataLoader import train_loader


def init_eer(score_path):
    errs = []
    with open(score_path) as file:
        lines = file.readlines()
        for line in lines:
            parteer = line.split(',')[-1]
            parteer = parteer.split(' ')[-1]
            parteer = parteer.replace('%', '')
            parteer = float(parteer)
            errs.append(parteer)
    return errs


def read_config(args):
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
        for key, item in config.items():
            setattr(args, key, item['value'])

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NAS_SEARCH_trainer")
    # Training Settings
    parser.add_argument('--config',
                        type=str,
                        default="config_nas_search.yml",
                        help='Configuration file')

    # Initialization
    warnings.simplefilter("ignore")
    # torch.multiprocessing.set_start_method('spawn', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

    args = parser.parse_args()
    args = read_config(args)
    args = init_args(args)

    # Define the data loader
    trainloader = train_loader(**vars(args))
    trainLoader = DataLoader(trainloader, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.n_cpu,
                             drop_last=True)

    # Search for the exist models
    modelfiles = glob.glob('%s/model_0*.model' % args.model_save_path)
    modelfiles.sort()

    # Only do evaluation, the initial_model is necessary
    if args.eval:
        s = NASModel(**vars(args))
        print(f"Model {args.initial_model} loaded from previous state!", flush=True)
        s.load_parameters(args.initial_model)
        EER, minDCF = s.eval_network(eval_list=args.eval_list, eval_path=args.eval_path, n_cpu=args.n_cpu)
        print(f"EER {EER: 2.2f}%, minDCF {minDCF:.4f}%", flush=True)
        quit()

    # If initial_model is exist, system will train from the initial_model
    if args.initial_model != "":
        print(f"Model {args.initial_model} loaded from previous state!", flush=True)
        s = NASModel(**vars(args))
        s.load_parameters(args.initial_model)
        epoch = 1
        EERs = []

    # Otherwise, system will try to start from the saved model&epoch
    elif len(modelfiles) >= 1:
        print(f"Model {modelfiles[-1]} loaded from previous state!", flush=True)
        epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
        s = NASModel(**vars(args))
        s.load_parameters(modelfiles[-1])
        EERs = init_eer(args.score_save_path)
    # Otherwise, system will train from scratch
    else:
        epoch = 1
        s = NASModel(**vars(args))
        EERs = []

    score_file = open(args.score_save_path, "a+")

    while True:
        # Training for one epoch
        loss, lr, acc = s.train_network(epoch=epoch, loader=trainLoader)

        # Evaluation every [test_step] epochs
        if epoch % args.test_step == 0:
            s.save_parameters(args.model_save_path + "/model_%04d.model" % epoch, delete=True)

            EERs.append(s.eval_network(eval_list=args.eval_list, eval_path=args.eval_path, n_cpu=args.n_cpu)[0])
            print(time.strftime("%Y-%m-%d %H:%M:%S"),
                  "%d epoch, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%" % (epoch, acc, EERs[-1], min(EERs)), flush=True)
            score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%\n" % (
                epoch, lr, loss, acc, EERs[-1], min(EERs)))
            score_file.flush()
            if EERs[-1] <= min(EERs):
                s.save_parameters(args.model_save_path + "/best.model")

        if epoch >= args.max_epoch:
            quit()

        epoch += 1
