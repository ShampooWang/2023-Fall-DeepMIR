import os
import sys
import time
import shutil
import numpy as np
import soundfile as sf

import torch

from logger.saver import Saver
from logger import utils
from tqdm import tqdm

from vocoder_eval.evaluate import evaluate
import subprocess
import wandb
from debuzz import debuzz_waves


def render(args, model, path_mel_dir, path_gendir='gen', is_part=False):
    print(' [*] rendering...')
    model.eval()

    # list files
    files = utils.traverse_dir(
        path_mel_dir, 
        extension='npy', 
        is_ext=False,
        is_sort=True, 
        is_pure=True)
    num_files = len(files)
    print(' > num_files:', num_files)

    # run
    with torch.no_grad():
        for fidx in range(num_files):
            fn = files[fidx]
            print('--------')
            print('{}/{} - {}'.format(fidx, num_files, fn))

            path_mel = os.path.join(path_mel_dir, fn) + '.npy'
            mel = np.load(path_mel)
            mel = torch.from_numpy(mel).float().to(args.device).unsqueeze(0)
            print(' mel:', mel.shape)

            # forward
            signal, f0_pred, _, (s_h, s_n) = model(mel)

            # path
            path_pred = os.path.join(path_gendir, 'pred', fn + '.wav')
            if is_part:
                path_pred_n = os.path.join(path_gendir, 'part', fn + '-noise.wav')
                path_pred_h = os.path.join(path_gendir, 'part', fn + '-harmonic.wav')
            print(' > path_pred:', path_pred)
            
            os.makedirs(os.path.dirname(path_pred), exist_ok=True)
            if is_part:
                os.makedirs(os.path.dirname(path_pred_h), exist_ok=True)

            # to numpy
            pred = utils.convert_tensor_to_numpy(signal)
            if is_part:
                pred_n = utils.convert_tensor_to_numpy(s_n)
                pred_h = utils.convert_tensor_to_numpy(s_h)
            
            # save
            sf.write(path_pred, pred, args.data.sampling_rate)
            if is_part:
                sf.write(path_pred_n, pred_n, args.data.sampling_rate)
                sf.write(path_pred_h, pred_h, args.data.sampling_rate)


def test(args, model, loss_func, loader_test, path_gendir='gen', is_part=True):
    print(' [*] testing...')
    print(' [*] output folder:', path_gendir)
    model.eval()

    # losses
    test_loss = 0.
    test_loss_mss = 0.
    test_loss_f0 = 0.
    
    # intialization
    num_batches = len(loader_test)
    os.makedirs(path_gendir, exist_ok=True)
    rtf_all = []

    # run
    anno_path_list, pred_path_list = [], []
    with torch.no_grad():
        for bidx, data in enumerate(tqdm(loader_test)):
            fn = data['name'][0]
            # print('--------')
            # print('{}/{} - {}'.format(bidx, num_batches, fn))

            # unpack data
            for k in data.keys():
                if k != 'name':
                    data[k] = data[k].to(args.device).float()
            # print('>>', data['name'][0])

            # forward
            st_time = time.time()
            signal, f0_pred, _, (s_h, s_n) = model(data['mel'])
            ed_time = time.time()

            # crop
            min_len = np.min([signal.shape[1], data['audio'].shape[1]])
            signal        = signal[:,:min_len]
            data['audio'] = data['audio'][:,:min_len]

            # RTF
            run_time = ed_time - st_time
            song_time = data['audio'].shape[-1] / args.data.sampling_rate
            rtf = run_time / song_time
            # print('RTF: {}  | {} / {}'.format(rtf, run_time, song_time))
            rtf_all.append(rtf)
           
            # loss
            loss, (loss_mss, loss_f0) = loss_func(
                signal, data['audio'], f0_pred, data['f0'])

            test_loss         += loss.item()
            test_loss_mss     += loss_mss.item() 
            test_loss_f0      += loss_f0.item()

            # path
            path_pred = os.path.join(path_gendir, 'pred', f"{bidx}.wav")
            path_anno = os.path.join(path_gendir, 'anno', f"{bidx}.wav")
            if is_part:
                path_pred_n = os.path.join(path_gendir, 'part', f"{bidx}-noise.wav")
                path_pred_h = os.path.join(path_gendir, 'part', f"{bidx}-harmonic.wav")

            # print(' > path_pred:', path_pred)
            # print(' > path_anno:', path_anno)

            os.makedirs(os.path.dirname(path_pred), exist_ok=True)
            os.makedirs(os.path.dirname(path_anno), exist_ok=True)
            if is_part:
                os.makedirs(os.path.dirname(path_pred_h), exist_ok=True)

            # to numpy
            pred  = utils.convert_tensor_to_numpy(signal)
            anno  = utils.convert_tensor_to_numpy(data['audio'])
            if is_part:
                pred_n = utils.convert_tensor_to_numpy(s_n)
                pred_h = utils.convert_tensor_to_numpy(s_h)
            
            # save
            sf.write(path_pred, pred, args.data.sampling_rate)
            sf.write(path_anno, anno, args.data.sampling_rate)
            if is_part:
                sf.write(path_pred_n, pred_n, args.data.sampling_rate)
                sf.write(path_pred_h, pred_h, args.data.sampling_rate)

            anno_path_list.append(path_anno)
            pred_path_list.append(path_pred)
    
    if is_part:
        debuzz_waves(path_gendir)

    result = evaluate(os.path.join(path_gendir, 'anno'), os.path.join(path_gendir, 'pred'))
    print(result)

    # report
    test_loss /= num_batches
    test_loss_mss     /= num_batches
    test_loss_f0      /= num_batches

    # check
    print(' [test_loss] test_loss:', test_loss)
    print(' Real Time Factor', np.mean(rtf_all))
    
    return test_loss, test_loss_mss, test_loss_f0


def train(args, model, loss_func, loader_train, loader_test):
    # saver
    saver = Saver(args)

    wandblogger = wandb.init(
        project="2023-DLMAG-HW2",
        dir=args.env.expdir,
        name=args.env.expdir.split("/")[-1],
        config=args
    )

    # model size
    params_count = utils.get_network_paras_amount({'model': model})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.train.lr)

    # run
    best_loss = np.inf
    num_batches = len(loader_train)
    model.train()
    prev_save_time = -1
    saver.log_info('======= start training =======')
    for epoch in range(args.train.epochs):
        for batch_idx, data in enumerate(tqdm(loader_train)):
            saver.global_step_increment()
            optimizer.zero_grad()

            # unpack data
            for k in data.keys():
                if k != 'name':
                    data[k] = data[k].to(args.device).float()
            
            # forward
            signal, f0_pred, _, _,  = model(data['mel'])

            # loss
            loss, (loss_mss, loss_f0) = loss_func(
                signal, data['audio'], f0_pred, data['f0'])
            
            # handle nan loss
            if torch.isnan(loss):
                raise ValueError(' [x] nan loss ')
            else:
                # backpropagate
                loss.backward()
                optimizer.step()

            wandblogger.log(
                {
                    'train loss step': loss.item(), 
                    'train loss mss step': loss_mss.item(),
                    'train loss f0 step': loss_f0.item(),
                },
                step=saver.global_step
            )

            # log loss
            if saver.global_step % args.train.interval_log == 0:
                saver.log_info(
                    'epoch: {}/{} {:3d}/{:3d} | {} | t: {:.2f} | loss: {:.6f} | time: {} | counter: {}'.format(
                        epoch,
                        args.train.epochs,
                        batch_idx,
                        num_batches,
                        saver.expdir,
                        saver.get_interval_time(),
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                    )
                )
                saver.log_info(
                    ' > mss loss: {:.6f}, f0: {:.6f}'.format(
                       loss_mss.item(),
                       loss_f0.item(),
                    )
                )

                y, s = signal, data['audio']
                saver.log_info(
                    "pred: max:{:.5f}, min:{:.5f}, mean:{:.5f}, rms: {:.5f}\n" \
                    "anno: max:{:.5f}, min:{:.5f}, mean:{:.5f}, rms: {:.5f}".format(
                            torch.max(y), torch.min(y), torch.mean(y), torch.mean(y** 2) ** 0.5,
                            torch.max(s), torch.min(s), torch.mean(s), torch.mean(s** 2) ** 0.5))

                saver.log_value({
                    'train loss': loss.item(), 
                    'train loss mss': loss_mss.item(),
                    'train loss f0': loss_f0.item(),
                })
            
        # validation
        # if saver.global_step % args.train.interval_val == 0:

        # save latest
        # saver.save_models(
        #         {'vocoder': model}, postfix=f'epoch{epoch}')


        # run testing set
        path_testdir_runtime = os.path.join(args.env.expdir, 'runtime_gen')
        test_loss, test_loss_mss, test_loss_f0 = test(
            args, model, loss_func, loader_test,
            path_gendir=path_testdir_runtime)
        saver.log_info(
            ' --- <validation> --- \nloss: {:.6f}. mss loss: {:.6f}, f0: {:.6f}'.format(
                test_loss, test_loss_mss, test_loss_f0
            )
        )
        saver.log_value({
            'valid loss': test_loss,
            'valid loss mss': test_loss_mss,
            'valid loss f0': test_loss_f0,
        })
        wandblogger.log(
            {
                'valid loss': loss.item(), 
                'valid loss mss': loss_mss.item(),
                'valid loss f0': loss_f0.item(),
            },
            step=saver.global_step
        )
        result = evaluate(os.path.join(path_testdir_runtime, 'anno'), os.path.join(path_testdir_runtime, 'pred'))
        wandblogger.log(result, step=saver.global_step)
        print(result)
        
        model.train()

        # save best model
        if test_loss < best_loss:
            saver.log_info(' [V] best model updated.')
            saver.save_models(
                {'vocoder': model}, postfix='best')
            test_loss = best_loss

        saver.make_report()


        

                          
