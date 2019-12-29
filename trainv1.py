import collections
import itertools
import datetime
import json

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.tensorboard as torchboard

import datautils
import utils

TrainSettings = collections.namedtuple(
        'TrainSettings',
        '''
        numepochs
        num_procs
        testingbatchsize
        loss_every_i_batches
        stats_every_i_batches
        weights_every_i_batches
        save_every_i_batches
        crossval_every_i_batches
        ''')

TrainSettings.__new__.__defaults__ = (
        -1,
        8,
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

    errloss = torch.tensor(0.)
    regloss = torch.tensor(0.)

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
            losses = loss_func(preds, target)
            errloss += losses[0]
            regloss += losses[1]
            count += 1

    writer.add_scalar('Err-Loss/crossval', errloss / count, global_counter)
    writer.add_scalar('Reg-Loss/crossval', regloss / count, global_counter)

    acc, precision, recall, fscore = datautils.calc_stats(stats)
    writer.add_scalar('Accuracy/crossval', acc, global_counter)
    writer.add_scalar('Precision/crossval', precision, global_counter)
    writer.add_scalar('Recall/crossval', recall, global_counter)
    writer.add_scalar('F-Score/crossval', fscore, global_counter)

def log_weights(writer, model, global_counter):
    for name, weight in model.named_parameters():
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
        rank,
        world_size,
        save_dir,
        experiment_name,
        model,
        loss_func,
        opt,
        get_train,
        get_crossval,
        batch_queue,
        experimentsettings,
        trainsettings={},
        ):
    """ The big train function with all the settings """

    dist.init_process_group('gloo', init_method='env://', world_size=world_size, rank=rank)

    torch.set_num_threads(1)

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

    batchsize = experimentsettings.batchsize

    numepochs = trainsettings.numepochs
    testingbatchsize = trainsettings.testingbatchsize
    loss_every_i_batches = trainsettings.loss_every_i_batches
    stats_every_i_batches = trainsettings.stats_every_i_batches
    save_every_i_batches = trainsettings.save_every_i_batches
    weights_every_i_batches = trainsettings.weights_every_i_batches

    parallelmodel = model

    if rank == 0:
        writer = torchboard.SummaryWriter(log_dir='runs/trainv1-{}-{}'.format(short_git_hash, experiment_name))

    try:
        epoch_iterator = range(numepochs) if numepochs is not -1 else itertools.count()
        dataset_iterator = ((epoch, point) for epoch in epoch_iterator for point in get_train())
        batch_iterator = utils.group_into(dataset_iterator, batchsize)
        stats = datautils.Stats()
        global_counter = 0
        prev_epoch = 0

        for batchnum, batch__ in enumerate(batch_iterator):
            # batch__ = list(batch__)
            # batch_ = list(batch__[rank::world_size])
            # epoch = [point[0] for point in batch__][0]
            # batch = [point[1] for point in batch_]

            dist.barrier()

            if rank == 0:
                batch__ = list(batch__)
                epoch = [point[0] for point in batch__][0]

                for point in batch__:
                    batch_queue.put(point[1])

            # Make sure everything sees something in the queue
            while batch_queue.empty():
                continue

            dist.barrier()

            if rank == 0 and batchnum % save_every_i_batches == 0:
                save_model(model, save_dir, short_git_hash, experiment_name, global_counter)

            if rank == 0 and epoch > prev_epoch:
                prev_epoch = epoch
                log_crossval(writer, model, parallelmodel, loss_func, get_crossval, testingbatchsize, global_counter)

            if rank == 0:
                print('(i: {}) Training batch #{}'.format(global_counter, batchnum))

            if rank == 0 and batchnum % stats_every_i_batches == 0:
                log_stats(writer, stats, global_counter)
                stats = datautils.Stats()

            if rank == 0 and batchnum % weights_every_i_batches == 0:
                log_weights(writer, model, global_counter)

            # batchx = [point[0] for point in batch]
            # batchy = [point[1] for point in batch]
            # preds = parallelmodel(batchx)

            batchx = []
            batchy = []
            preds = []

            while True:
                try:
                    point = batch_queue.get(block=False)
                    batchx.append(point[0])
                    batchy.append(point[1])
                    preds.append(model([point[0]]))
                except:
                    if batch_queue.qsize() == 0 and batch_queue.empty():
                        break

            preds = torch.cat(preds)

            for pred, y in zip(preds, batchy):
                with torch.no_grad():
                    stats = datautils.update_stats(stats, bool(pred > 0), y)

            global_counter += len(batch__)
            target = torch.tensor([1. if y else 0. for y in batchy])
            errloss, regloss = loss_func(preds, target)
            errloss.backward()

            for param in model.parameters():
                param.grad.data *= len(preds)
                dist.all_reduce(param.grad.data)
                param.grad.data /= len(batch__)

            regloss.backward()

            if rank == 0 and batchnum % loss_every_i_batches == 0:
                print('(i: {}) Loss: {:.5f}\t{:.5f}'.format(global_counter, errloss.item(), regloss.item()))
                writer.add_scalar('Err-Loss/train', errloss.item(), global_counter)
                writer.add_scalar('Reg-Loss/train', regloss.item(), global_counter)

            dist.barrier()

            if rank == 0:
                opt.step()
            opt.zero_grad()

    except KeyboardInterrupt:
        pass
