import wandb
from .loss import calc_loss, calc_val_metrics
from . import utility
import globals
import helper
import models


def possibly_track_metric(metric, value, tracker, step=None):
    if tracker is not None:
        tracker.track_metric(metric, value, step)


def track_wandb_metrics(metrics_dict, step):
    for k, v in metrics_dict.items():
        if k in ['all_image_names', 'all_preds', 'min_mae_pos', 'max_mae_pos']:
            continue
        wandb.log({f'val_{k}': v}, step=step)


def train(model, optimizer, lr, model_name, model_mode, loss_type, train_loader, val_loader, max_epoch, eval_step,
          do_track, resume_step=None, resume_epoch=None):
    # preparing for train data
    model_paths = helper.get_paths(model_name)
    epoch = 1 if resume_epoch is None else resume_epoch
    step = 1 if resume_step is None else resume_step + 1

    # training loop
    while epoch <= max_epoch:
        for i_batch, batch in enumerate(train_loader):
            image_batch, targets, labels, ann_inds, int_cancer_weights = utility.extract_batch(batch, loss_type)
            logits = model(image_batch, ann_inds)  # forward uses ann_inds only if model was initialized with 'sep_anns' mode

            loss = calc_loss(loss_type, logits, targets)
            train_loss = loss.item()

            # making gradients zero in each optimization step
            model.zero_grad()
            loss.backward()
            optimizer.step()

            globals.logger.info(f'Epoch: {epoch}, step: {step}, batch: {i_batch + 1} - loss: {train_loss}')
            if do_track:
                wandb.log({'train_loss': train_loss}, step=step)

            # calc val loss and save checkpoint
            if step == 1 or step % eval_step == 0:
                # saving checkpoint
                helper.save_checkpoint(model_paths['checkpoints_path'], step, model, optimizer, train_loss, lr)
                # calc val metrics by loading the saved model in eval mode
                if val_loader is not None:
                    model_for_eval = models.init_and_load_model_for_eval(model_name, model_mode, loss_type, step)
                    val_dict = calc_val_metrics(val_loader, model_for_eval, model_mode, loss_type)
                    # track metrics
                    if do_track:
                        track_wandb_metrics(val_dict, step)
                    globals.logger.info(f'Epoch: {epoch}, step: {step} => evaluation done\n')

            # end of one batch
            step += 1
        # end of one epoch
        epoch = epoch + 1
        globals.logger.info(f'Now epoch increased to {epoch}\n\n')
