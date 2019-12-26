import collections
import itertools

import torch
import torch.nn as nn
import torch.utils.tensorboard as torchboard

import datautils

TrainSettings = collections.namedtuple(
        'TrainSettings',
        '''
        numepochs
        batchsize
        loss_every_i_batches
        stats_every_i_batches
        weights_every_i_batches
        save_every_i_batches
        crossval_every_i_batches
        ''')

TrainSettings.__new__.__defaults__ = (
        None,
        torch.get_num_threads() * 2,
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

def log_crossval(writer, model, get_crossval, global_counter):
    stats = Stats()

    import torch.multiprocessing as mp

    with torch.no_grad():
        with mp.Pool() as pool:
            results = pool.map(lambda point: (model(point[0]), point[1]), utils.group_into(get_crossval(), 4))

        for pred, y in results:
            stats = update_stats(stats, bool(pred > 0), y)

    acc, precision, recall, fscore = datautils.calc_stats(stats)
    writer.add_scalar('Accuracy/crossval', acc, global_counter)
    writer.add_scalar('Precision/crossval', precision, global_counter)
    writer.add_scalar('Recall/crossval', recall, global_counter)
    writer.add_scalar('F-Score/crossval', fscore, global_counter)

def log_weights(writer, model, global_counter):
    for name, weight in model.classifier.named_parameters():
        writer.add_image('weight_{}'.format(name), utils.twodfy(weight), global_counter, dataformats='HW')

def train(
        save_dir,
        experiment_name,
        model,
        loss_func,
        opt,
        get_train,
        get_crossval,
        trainsettings=TrainSettings(),
        modelsettings=tuple(),
        ):
    """ The big train function with all the settings """

    numepochs, batchsize, loss_every_i_batches, stats_every_i_batches, save_every_i_batches = trainsettings

    parallelmodel = nn.DataParallel(model)
    writer = torchboard.SummaryWriter(comment='trainv1-{}-{}'.format(experiment_name, modelsettings))

    try:
        epoch_iterator = range(numepochs) if numepochs is not None else itertools.count()
        dataset_iterator = itertools.chain(get_train() for epoch in epoch_iterator)
        batch_iterator = utils.group_into(dataset_iterator, batchsize)
        stats = Stats()

        for batchnum, batch in enumerate(batch_iterator):
            if batchnum % save_every_i_batches == 0:
                save_model(model, data_dir, global_counter)

            if batchnum % crossval_every_i_batches == 0:
                log_crossval(writer, model, global_counter)

            loss = torch.tensor([0.])

            batchx = [point[0] for point in batch]
            batchy = [point[1] for point in batch]

            if batchnum % stats_every_i_batches == 0:
                log_stats(writer, stats)
                stats = Stats()

            if batchnum % weights_every_i_batches == 0:
                log_weights(writer, model)

            preds = parallelmodel(batchx)

            for pred, y in zip(preds, batchy):
                with torch.no_grad():
                    stats = update_stats(stats, bool(pred > 0), y)

                loss += loss_func(model, pred, torch.tensor([1. if y else 0.]))
                global_counter += 1

            if batchnum % loss_every_i_batches == 0:
                writer.add_scalar('Loss/train', loss, global_counter)

            loss.backward()
            opt.step()
            opt.zero_grad()

    except KeyboardInterrupt:
        pass
