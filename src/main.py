import argparse
import wandb
from torch import optim

import models
import helper
import trainer
import globals
import data_handler
import trainer.evaluation as evaluation


def parse_args():
    parser = argparse.ArgumentParser(description='Annotation tool')
    parser.add_argument('--train', action='store_true')  # for training models
    parser.add_argument('--evaluate', action='store_true')  # for evaluation

    parser.add_argument('--model_name', type=str)
    parser.add_argument('--loss_type', type=str)

    parser.add_argument('--train_folder', type=str)  # explicitly set it in the argument, if wanted
    parser.add_argument('--val_folder', type=str)
    parser.add_argument('--test_folder', type=str)

    parser.add_argument('--train_csv', type=str)
    parser.add_argument('--val_csv', type=str)
    parser.add_argument('--test_csv', type=str)

    parser.add_argument('--cv', type=int)  # e.g. --cv 1 denoting the first fold of cross-validation
    parser.add_argument('--checkpoints_path', type=str)
    parser.add_argument('--line_parse_type', type=int)  # for csv file lines
    parser.add_argument('--save_preds_to', type=str)  # path to save model predictions to

    parser.add_argument('--step', type=int)  # used to load checkpoint for evaluation
    parser.add_argument('--b_size', type=int)  # batch size
    parser.add_argument('--img_size', nargs='+', type=int)
    parser.add_argument('--imread_mode', type=int)
    parser.add_argument('--n_workers', type=int)  # number of workers for data loading
    parser.add_argument('--lr', type=float)  # learning rate
    parser.add_argument('--resume_epoch', type=int)  # epoch to resume from
    parser.add_argument('--resume_step', type=int)  # step to resume from
    parser.add_argument('--train_augments', nargs='+')  # data augmentations, passed as strings

    parser.add_argument('--gpu_id', type=int, default=7)  # -1 for cpu (NOT IMPLEMENTED FOR NOW)
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--eval_step', type=int)
    parser.add_argument('--no_tracker', action='store_true')
    return parser.parse_args()


def check_args(args):
    assert args.model_name is not None, 'model_name should be specified'
    assert args.loss_type is not None, 'loss_type should be specified'
    assert args.train or args.evaluate, 'main script should be run with either --train or --evaluate'


def update_params_with_args(args, params):
    print(f'\n{"=" * 100}\n')
    if args.train_folder is not None:
        params['data']['train_folder'] = args.train_folder
        print(f'train_folder updated to: {args.train_folder}')

    if args.val_folder is not None:
        params['data']['val_folder'] = args.val_folder
        print(f'val_folder updated to: {args.val_folder}')

    if args.test_folder is not None:
        params['data']['test_folder'] = args.test_folder
        print(f'test_folder updated to: {args.test_folder}')

    if args.train_csv is not None:
        params['data']['train_csv'] = args.train_csv
        print(f'train_csv updated to: {args.train_csv}')

    if args.val_csv is not None:
        params['data']['val_csv'] = args.val_csv
        print(f'val_csv updated to: {args.val_csv}')

    if args.test_csv is not None:
        params['data']['test_csv'] = args.test_csv
        print(f'test_csv updated to: {args.test_csv}')

    if args.img_size is not None:
        params['train']['img_size'] = args.img_size
        print(f'img_size updated to: {args.img_size}')

    if args.imread_mode is not None:
        params['data']['imread_mode'] = args.imread_mode
        print(f'imread_mode updated to: {args.imread_mode}')

    if args.train_augments is not None:
        params['train']['augments'] = args.train_augments
        print(f'augments (for training) updated to: {args.train_augments}')

    if args.eval_step is not None:
        params['train']['eval_step'] = args.eval_step
        print(f'eval_step updated to: {args.eval_step}')

    if args.n_epochs is not None:
        params['train']['n_epochs'] = args.n_epochs
        print(f'n_epochs updated to: {args.n_epochs}')

    if args.b_size is not None:
        params['train']['batch_size'] = args.b_size
        print(f'batch_size changed to: {args.b_size}')

    if args.n_workers is not None:
        params['train']['n_workers'] = args.n_workers
        print(f'n_workers changed to: {args.n_workers}')

    if args.lr is not None:
        params['train']['lr'] = args.lr
        print(f'Learning rate changed to: {args.lr}')

    if args.line_parse_type is not None:
        params['data']['line_parse_type'] = args.line_parse_type
        print(f'line_parse_type changed to: {args.line_parse_type}')

    if args.checkpoints_path is not None:
        params['train']['checkpoints_path'] = args.checkpoints_path
        print(f'checkpoints_path changed to: {args.checkpoints_path}')
    print(f'\n{"=" * 100}\n')


def set_globals(gpu_id, params):
    assert gpu_id != -1
    globals.setup_logger(pure_line=False)
    globals.set_current_device(gpu_id)
    globals.params = params  # universal object that would be used in other modules
    globals.logger.info(f'Global device is: {globals.get_current_device()}')
    globals.logger.info(f'Global params are set.\n')


def train_model(args, params):
    # wandb tracker
    if args.no_tracker:
        do_track = False
    else:  # default
        do_track = True
        wandb.init(project='Masking', name=args.model_name, config=params)
        globals.logger.info(f'\033[1mWandb initialized!\033[10m\n')

    # init model
    model = models.init_model(loss_type=args.loss_type)
    model_name = args.model_name

    # init optimizer
    optimizer = optim.Adam(model.parameters())
    lr = params['train']['lr']
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    globals.logger.info(f'\033[1mOptimizer learning rate adjusted to: {lr}\033[0m\n')

    # load model and optimizer for resuming training
    if args.resume_step is not None:
        model, optimizer, _, _ = helper.load_checkpoint(path_to_load=helper.get_paths(model_name)['checkpoints_path'],
                                                        step=args.resume_step,
                                                        model=model,
                                                        optimizer=optimizer,
                                                        resume_train=True)
        globals.logger.info(f'Model and optimizer loaded successfully from step: {args.resume_step}')

    # running with cross-validation
    if args.cv:  # determining files with cross validation
        globals.logger.info(f'\033[1mPerforming cross-validation for fold: {args.cv}\033[0m')
        train_names, val_names = [], []
        for i in range(5):  # only 5-fold is supported now
            file = params['data']['cv_files'][i]
            if args.cv == i + 1:  # add to val
                val_names.extend(helper.read_file_to_list(file))  # read filenames
                globals.logger.info(f'Using {file} for validation')
            else:  # add the rest to train
                train_names.extend(helper.read_file_to_list(file))
                globals.logger.info(f'Using {file} for training')

        # having got the filenames, we use them to get the sub-dataframes
        all_train_list = helper.read_csv_to_list(params['data']['train_csv'])
        val_list = [line for line in all_train_list if line.split(';')[0] in val_names]
        train_list = [line for line in all_train_list if line.split(';')[0] in train_names]

    # usual running
    else:
        # reading csv file used for data loaders
        train_csv, val_csv = params['data']['train_csv'], params['data']['val_csv']  # val_csv could be None
        train_list = helper.read_csv_to_list(train_csv)
        val_list = helper.read_csv_to_list(val_csv) if val_csv is not None else []  # final training, without any validation

    globals.logger.info(f'\033[1mCreated train_list with len: {len(train_list):,}, '
                        f'val_list with len: {len(val_list):,}, '
                        f'total unique images: {len(list(set(train_list + val_list))):,}\033[0m\n')

    # common params for dataset
    common_dataset_params = {
        'data_folder': params['data']['train_folder'],  # same data_folder for both train and cross-val
        'img_size': params['train']['img_size'],
        'imread_mode': params['data']['imread_mode'],
        'line_parse_type': params['data']['line_parse_type'],
        'csv_sep_type': params['data']['csv_sep_type']
    }
    # common params for data loader
    common_data_loader_params = {
        'num_workers': params['train']['n_workers'],
        'batch_size': params['train']['batch_size']
    }

    # init train loader
    globals.logger.info(f'Initializing train data loader...')
    train_dataset_params = {'data_list': train_list, 'augments': params['train']['augments'], **common_dataset_params}
    train_data_loader_params = {'shuffle': True, **common_data_loader_params}
    train_loader = data_handler.init_data_loader(train_dataset_params, train_data_loader_params)

    # init val loader, if wanted
    if len(val_list) == 0:
        val_loader = None  # no val loader
        globals.logger.info(f'\033[1mInitialized train_loader of len: {len(train_loader)}, val_loader is: None \033[10m\n')
    else:
        # init val data loader
        globals.logger.info(f'Initializing validation data loader...')
        val_dataset_params = {'data_list': val_list, **common_dataset_params}
        val_data_loader_params = {'shuffle': False, **common_data_loader_params}
        val_loader = data_handler.init_data_loader(val_dataset_params, val_data_loader_params)
        globals.logger.info(f'\033[1mInitialized train_loader of len: {len(train_loader)}, val_loader of len: {len(val_loader)}\033[10m\n')

    # do training
    trainer.train(model, optimizer, lr, model_name, args.loss_type, train_loader, val_loader,
                  max_epoch=params['train']['n_epochs'],
                  eval_step=params['train']['eval_step'],
                  do_track=do_track,
                  resume_step=args.resume_step,
                  resume_epoch=args.resume_epoch)


def main():
    # read args and check they are appropriate
    args = parse_args()
    check_args(args)

    # read params and update it based on optional args
    params = helper.read_params()
    update_params_with_args(args, params)

    # set globals so they are visible by all modules
    set_globals(args.gpu_id, params)

    if args.train:
        train_model(args, params)  # start training

    elif args.evaluate:
        assert args.step, 'Please specify --step'
        evaluation.evaluate_model(test_csv=params['data']['test_csv'],
                                  model_name=args.model_name,
                                  loss_type=args.loss_type,
                                  step=args.step,
                                  params=params,
                                  save_preds_to=args.save_preds_to)
    else:
        raise NotImplementedError('Please specify the correct tag: --train, --evaluate')


if __name__ == '__main__':
    main()
