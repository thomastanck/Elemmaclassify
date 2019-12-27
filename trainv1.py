import collections
import itertools
import datetime
import json

import torch
import torch.nn as nn
import torch.utils.tensorboard as torchboard

import datautils
import utils

TrainSettings = collections.namedtuple(
        'TrainSettings',
        '''
        numepochs
        batchsize
        testingbatchsize
        loss_every_i_batches
        stats_every_i_batches
        weights_every_i_batches
        save_every_i_batches
        crossval_every_i_batches
        ''')

TrainSettings.__new__.__defaults__ = (
        None,
        torch.get_num_threads() * 2,
        1024,
        1,
        5,
        5,
        50,
        50,
        )

def log_stats(writer, stats, global_counter):
    acc, precision, recall, fscore = datautils.calc_stats(stats)
    writer.add_scalar('Accuracy/train', acc, global_counter)
    writer.add_scalar('Precision/train', precision, global_counter)
    writer.add_scalar('Recall/train', recall, global_counter)
    writer.add_scalar('F-Score/train', fscore, global_counter)

def log_crossval(writer, model, parallelmodel, loss_func, get_crossval, batchsize, global_counter):
    print('(i: {}) Testing cross-validation set'.format(global_counter))
    stats = datautils.Stats()

    batch_iterator = utils.group_into(get_crossval(), batchsize)

    loss = torch.tensor([0.])

    with torch.no_grad():
        count = 0
        for batchnum, batch in enumerate(batch_iterator):
            print('(i: {}) Cross-validation batch #{}'.format(global_counter, batchnum))

            batchx = [point[0] for point in batch]
            batchy = [point[1] for point in batch]

            preds = parallelmodel(batchx)

            for pred, y in zip(preds, batchy):
                stats = datautils.update_stats(stats, bool(pred > 0), y)

            target = torch.tensor([1. if y else 0. for y in batchy])
            loss += loss_func(model, preds, target)
            count += 1

    writer.add_scalar('Loss/train', loss / count, global_counter)

    acc, precision, recall, fscore = datautils.calc_stats(stats)
    writer.add_scalar('Accuracy/crossval', acc, global_counter)
    writer.add_scalar('Precision/crossval', precision, global_counter)
    writer.add_scalar('Recall/crossval', recall, global_counter)
    writer.add_scalar('F-Score/crossval', fscore, global_counter)

def log_weights(writer, model, global_counter):
    for name, weight in model.classifier.named_parameters():
        writer.add_image('weight_{}'.format(name), utils.twodfy(weight), global_counter, dataformats='HW')

def log_experiment_info(save_dir, experiment_name, git_hash, now, experimentsettings, trainsettings):
    info_filename = '{}/trainv1-{}-{}.info'.format(save_dir, git_hash, experiment_name)
    with open(info_filename, 'w') as f:
        json.dump({
            'start_time': str(now),
            'git_hash': str(git_hash),
            'experimentsettings': str(experimentsettings),
            'trainsettings': str(trainsettings),
            'train version': 'trainv1',
            }, f, indent=4, sort_keys=True)

def save_model(model, save_dir, short_git_hash, experiment_name, global_counter):
    model_filename = '{}/trainv1-{}-{}-{}.model'.format(save_dir, short_git_hash, experiment_name, global_counter)
    statedict_filename = '{}/trainv1-{}-{}-{}.statedict'.format(save_dir, short_git_hash, experiment_name, global_counter)
    print('(i: {}) Saving model to {}'.format(global_counter, model_filename))

    torch.save(model.state_dict(), statedict_filename)
    torch.save(model, model_filename)

def train(
        save_dir,
        experiment_name,
        model,
        loss_func,
        opt,
        get_train,
        get_crossval,
        experimentsettings,
        trainsettings=TrainSettings(),
        ):
    """ The big train function with all the settings """

    torch.set_num_threads(8)

    if not utils.is_git_clean():
        raise RuntimeError('Ensure that all changes have been committed!')

    git_hash = utils.git_hash()
    short_git_hash = git_hash[:7]

    # Start time string
    now = datetime.datetime.now()
    timestring = now.strftime('%Y%m%d%a-%H%M%S')

    experiment_name = '{}-{}'.format(experiment_name, timestring)

    # Log experiment information
    log_experiment_info(save_dir, experiment_name, git_hash, now, experimentsettings, trainsettings)

    numepochs = trainsettings.numepochs
    batchsize = trainsettings.batchsize
    testingbatchsize = trainsettings.testingbatchsize
    loss_every_i_batches = trainsettings.loss_every_i_batches
    stats_every_i_batches = trainsettings.stats_every_i_batches
    save_every_i_batches = trainsettings.save_every_i_batches
    weights_every_i_batches = trainsettings.weights_every_i_batches

    parallelmodel = nn.DataParallel(model)
    writer = torchboard.SummaryWriter(comment='trainv1-{}-{}'.format(short_git_hash, experiment_name))

    try:
        epoch_iterator = range(numepochs) if numepochs is not None else itertools.count()
        dataset_iterator = ((epoch, point) for epoch in epoch_iterator for point in get_train())
        batch_iterator = utils.group_into(dataset_iterator, batchsize)
        stats = datautils.Stats()
        global_counter = 0
        prev_epoch = -1

        for batchnum, batch_ in enumerate(batch_iterator):
            epoch = [point[0] for point in batch_][0]
            batch = [point[1] for point in batch_]

            if batchnum % save_every_i_batches == 0:
                save_model(model, save_dir, short_git_hash, experiment_name, global_counter)

            if epoch != prev_epoch:
                prev_epoch = epoch
                log_crossval(writer, model, parallelmodel, loss_func, get_crossval, testingbatchsize, global_counter)

            print('(i: {}) Training batch #{}'.format(global_counter, batchnum))

            batchx = [point[0] for point in batch]
            batchy = [point[1] for point in batch]

            if batchnum % stats_every_i_batches == 0:
                log_stats(writer, stats, global_counter)
                stats = datautils.Stats()

            if batchnum % weights_every_i_batches == 0:
                log_weights(writer, model, global_counter)

            preds = parallelmodel(batchx)

            for pred, y in zip(preds, batchy):
                with torch.no_grad():
                    stats = datautils.update_stats(stats, bool(pred > 0), y)

            global_counter += len(batch)
            target = torch.tensor([1. if y else 0. for y in batchy])
            loss = loss_func(model, preds, target)

            if batchnum % loss_every_i_batches == 0:
                writer.add_scalar('Loss/train', loss, global_counter)

            loss.backward()
            opt.step()
            opt.zero_grad()

    except KeyboardInterrupt:
        pass
