import argparse
import wandb
from torch import optim
from tmp import models
import trainer
import globals
import helper
import data_handler


def parse_args():
    parser = argparse.ArgumentParser(description='Annotation tool')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--model_mode', type=str, default='usual')  # 'usual' or 'sep_anns', the latter for separate annotators
    parser.add_argument('--annotator', type=str)
    parser.add_argument('--loss_type', type=str)
    parser.add_argument('--pretraining', type=str)  # which pretraining type to choose
    parser.add_argument('--train_csv', type=str)
    parser.add_argument('--val_csv', type=str)
    parser.add_argument('--cv', type=int)  # e.g. --cv 1 denoting the first fold of cross-validation
    parser.add_argument('--csv_sep_type', type=int)
    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--checkpoints_path', type=str)

    parser.add_argument('--classes_weighted', action='store_true')  # weighted for classes
    parser.add_argument('--samples_weighted', action='store_true')  # weighted for samples, for softmax with distance

    parser.add_argument('--line_parse_type', type=int)  # for csv file lines
    parser.add_argument('--b_size', type=int)  # batch size
    parser.add_argument('--img_size', nargs='+', type=int)
    parser.add_argument('--imread_mode', type=int)
    parser.add_argument('--n_workers', type=int)  # number of workers for data loading
    parser.add_argument('--lr', type=float)  # learning rate
    parser.add_argument('--resume_epoch', type=int)  # epoch to resume from
    parser.add_argument('--resume_step', type=int)  # step to resume from
    parser.add_argument('--augments', nargs='+')  # data augmentations, passed as strings

    parser.add_argument('--gpu_id', type=int, default=7)  # -1 for cpu (NOT IMPLEMENTED FOR NOW)
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--eval_step', type=int)
    parser.add_argument('--no_tracker', action='store_true')
    parser.add_argument('--tags', nargs='+')  # tags for the comet experiment
    parser.add_argument('--prev_exp_id', type=str)
    return parser.parse_args()


def check_args(args):
    assert args.model_name is not None, 'model_name should be specified'
    assert args.loss_type is not None, 'loss_type should be specified'


def update_params_with_args(args, params):
    print(f'\n{"=" * 100}\n')
    if args.img_size is not None:
        params['train']['img_size'] = args.img_size
        print(f'img_size updated to: {args.img_size}')

    if args.imread_mode is not None:
        params['data']['imread_mode'] = args.imread_mode
        print(f'imread_mode updated to: {args.imread_mode}')

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

    if args.csv_sep_type is not None:
        params['data']['csv_sep_type'] = args.csv_sep_type
        print(f'csv_sep_type changed to: {args.csv_sep_type}')

    if args.data_folder is not None:
        params['data']['folder'] = args.data_folder
        print(f'data_folder changed to: {args.data_folder}')

    if args.checkpoints_path is not None:
        params['train']['checkpoints_path'] = args.checkpoints_path
        print(f'checkpoints_path changed to: {args.checkpoints_path}')
    print(f'\n{"=" * 100}\n')


def set_globals(gpu_id, params):
    assert gpu_id != -1
    globals.setup_logger(pure_line=True)
    globals.set_current_device(gpu_id)
    globals.params = params
    globals.logger.info(f'Global device is: {globals.get_current_device()}')
    globals.logger.info(f'Global params are set.\n')


def train_model(args, params):
    # wandb
    if args.no_tracker:
        do_track = False
    else:  # default
        do_track = True
        wandb.init(project='Masking', name=args.model_name, config=params)
        globals.logger.info(f'\033[1mWandb initialized!\033[10m\n')

    # init model
    model = models.init_model(model_name=args.model_name,
                              mode=args.model_mode,
                              loss_type=args.loss_type,
                              pretraining=args.pretraining,
                              checkpoint_file=args.moco_checkpoint_file)
    model_name, model_mode = args.model_name, args.model_mode

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
        train_list, val_list = [], []
        for i in range(5):  # only 5-fold is supported now
            file = params['data']['cv_files'][i]
            if args.cv == i + 1:  # add to val
                val_list.extend(helper.read_csv_to_list(file))
                globals.logger.info(f'Using {file} for for validation')
            else:  # add the rest to train
                train_list.extend(helper.read_csv_to_list(file))
                globals.logger.info(f'Using {file} for for training')
    # usual running
    else:
        if args.annotator is not None:  # getting train and val data lines with annotator
            train_csv = params['data']['train_csv'][args.annotator]
            val_csv = params['data']['val_csv'][args.annotator]
        elif (args.train_csv and args.val_csv) or (args.train_csv and not args.val_csv):  # getting data with explicit filenames
            train_csv = args.train_csv
            val_csv = args.val_csv  # val csv could be None, which means final training without tracking val loss
        else:
            raise NotImplementedError

        train_list = helper.read_csv_to_list(train_csv)
        val_list = helper.read_csv_to_list(val_csv) if val_csv is not None else []

    globals.logger.info(f'\033[1mCreated train_list with len: {len(train_list):,}, '
                        f'val_list with len: {len(val_list):,}, '
                        f'total unique images: {len(list(set(train_list + val_list))):,}\033[0m\n')

    # init data loaders
    data_folder = params['data']['folder']
    line_parse_type = params['data']['line_parse_type']  # used for train csv
    imread_mode = params['data']['imread_mode']
    csv_sep_type = params['data']['csv_sep_type']
    train_loader = data_handler.init_data_loader(dataset_mode='train',
                                                 data_folder=data_folder,
                                                 data_list=train_list,
                                                 line_parse_type=line_parse_type,
                                                 csv_sep_type=csv_sep_type,
                                                 imread_mode=imread_mode,
                                                 n_workers=params['train']['n_workers'],
                                                 batch_size=params['train']['batch_size'],
                                                 img_size=params['train']['img_size'],
                                                 shuffle=True,
                                                 augments=args.augments)

    if len(val_list) == 0:
        val_loader = None  # no val loader
        globals.logger.info(f'\033[1mInitialized train_loader of len: {len(train_loader)}, val_loader is: None \033[10m\n')
    else:
        val_loader = data_handler.init_data_loader(dataset_mode='val',
                                                   data_folder=data_folder,
                                                   data_list=val_list,
                                                   line_parse_type=line_parse_type,
                                                   csv_sep_type=csv_sep_type,
                                                   imread_mode=imread_mode,
                                                   n_workers=params['train']['n_workers'],
                                                   batch_size=params['train']['batch_size'],
                                                   img_size=params['train']['img_size'],
                                                   shuffle=False,
                                                   augments=None)
        globals.logger.info(f'\033[1mInitialized train_loader of len: {len(train_loader)}, val_loader of len: {len(val_loader)}\033[10m\n')

    # do training
    trainer.train(model, optimizer, lr, model_name, model_mode, args.loss_type, train_loader, val_loader,
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

    train_model(args, params)


if __name__ == '__main__':
    main()
