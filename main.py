"""
Entry point training and testing multi-scene transformer
"""
import argparse
import torch
import numpy as np
import json
import logging
from util import utils
import time
from datasets.CameraPoseDataset import CameraPoseDataset
from models.pose_losses import CameraPoseLoss
from models.pose_regressors import get_model
from os.path import join
import matplotlib.pyplot as plt
import scipy.io as sio



def test(test_dataloader, mean_posit_errs, epoch):
    model.eval()
    stats = np.zeros((len(test_dataloader.dataset), 3))
    pred_poses = np.zeros((len(test_dataloader.dataset), 7))  # store all predicted poses
    targ_poses = np.zeros((len(test_dataloader.dataset), 7))  # store all target poses
    with torch.no_grad():
        for i, minibatch in enumerate(test_dataloader, 0):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)
                gt_scene = minibatch.get('scene')
                minibatch['scene'] = None # avoid using ground-truth scene during prediction

                gt_pose = minibatch.get('pose').to(dtype=torch.float32)

                # Forward pass to predict the pose
                tic = time.time()
                est_pose = model(minibatch).get('pose')
                toc = time.time()

                pred_poses[i, :] = est_pose[0].cpu().data.numpy()
                targ_poses[i, :] = gt_pose[0].cpu().data.numpy()

                # Evaluate error
                posit_err, orient_err = utils.pose_err(est_pose, gt_pose)

                # Collect statistics
                stats[i, 0] = posit_err.item()
                stats[i, 1] = orient_err.item()
                stats[i, 2] = (toc - tic)*1000

        # Record overall statistics
        logging.info("Performance of {} on {}".format(args.checkpoint_path, args.labels_file))
        mean_posit_err = np.nanmean(stats[:, 0])
        mean_posit_errs[epoch-251] = mean_posit_err
        mean_orient_err = np.nanmean(stats[:, 1])
        logging.info("Median pose error: {:.4f}[m], {:.4f}[deg]".format(mean_posit_err, mean_orient_err))
        logging.info("Mean inference time:{:.2f}[ms]".format(np.mean(stats[:, 2])))
        if (epoch-251) == 0: return True
        if mean_posit_err <= min(mean_posit_errs[:epoch-251]): return True
        return False



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model_name",default="fusionnet",
                            help="name of model to create (e.g. posenet, transposenet")
    arg_parser.add_argument("--mode", help="train or eval", required=True)
    arg_parser.add_argument("--dataset_path", help="path to the physical location of the dataset", required=True)
    arg_parser.add_argument("--labels_file", default="./datasets/3Floor/abs_pose.csv_3floor_train.csv", help="path to a file mapping images to their poses", required=True)
    arg_parser.add_argument("--rssi_file", default="./datasets/3Floor/rssi.csv_3floor_train_dense.csv", help="path to a file containing rssis", required=True)
    arg_parser.add_argument("--config_file", help="path to configuration file", default="3Floor_config.json")
    arg_parser.add_argument("--checkpoint_path", 
                            help="path to a pre-trained model (should match the model indicated in model_name")

    args = arg_parser.parse_args()
    utils.init_logger()

    # Record execution details
    logging.info("Start {} with {}".format(args.model_name, args.mode))
    logging.info("Using dataset: {}".format(args.dataset_path))
    logging.info("Using labels file: {}".format(args.labels_file))

    # Read configuration
    with open(args.config_file, "r") as read_file:
        config = json.load(read_file)
    model_params = config[args.model_name]
    general_params = config['general']
    config = {**model_params, **general_params}
    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = config.get('device_id')
    np.random.seed(numpy_seed)
    device = torch.device(device_id)

    # Create the model
    model = get_model(args.model_name, config).to(device)
    # Load the checkpoint if needed
    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device_id))
        logging.info("Initializing from checkpoint: {}".format(args.checkpoint_path))

    if args.mode == 'train':
        # Set to train mode
        model.train()

        # Freeze parts of the model if indicated
        freeze = config.get("freeze")
        freeze_exclude_phrase = config.get("freeze_exclude_phrase")
        if isinstance(freeze_exclude_phrase, str):
            freeze_exclude_phrase = [freeze_exclude_phrase]
        if freeze:
            for name, parameter in model.named_parameters():
                freeze_param = True
                for phrase in freeze_exclude_phrase:
                    if phrase in name:
                        freeze_param = False
                        break
                if freeze_param:
                        parameter.requires_grad_(False)

        # Set the loss
        pose_loss = CameraPoseLoss(config).to(device)
        nll_loss = torch.nn.NLLLoss()

        # Set the optimizer and scheduler
        params = list(model.parameters()) + list(pose_loss.parameters())
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                                  lr=config.get('lr'),
                                  eps=config.get('eps'),
                                  weight_decay=config.get('weight_decay'))
        use_scheduler = config.get("use_scheduler")
        if use_scheduler == True:
            scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                        step_size=config.get('lr_scheduler_step_size'),
                                                        gamma=config.get('lr_scheduler_gamma'))

        # Set the dataset and data loader
        no_augment = config.get("no_augment")
        if no_augment:
            train_transform_baseline = utils.test_transforms.get('baseline')
            train_transform_wifi = utils.test_transforms.get('wifi')
        else:
            train_transform_baseline = utils.train_transforms.get('baseline')
            train_transform_wifi = utils.train_transforms.get('wifi')


        train_dataset = CameraPoseDataset(args.dataset_path, args.rssi_file, args.labels_file, train_transform_baseline, train_transform_wifi, False, 'train')
        train_loader_params = {'batch_size': config.get('batch_size'),
                                  'shuffle': True,
                                  'num_workers': config.get('n_workers')}
        train_dataloader = torch.utils.data.DataLoader(train_dataset, **train_loader_params)
        test_transform_baseline = utils.test_transforms.get('baseline')
        test_transform_wifi = utils.test_transforms.get('wifi')
        test_dataset = CameraPoseDataset(args.dataset_path, 
                                         args.rssi_file.split(args.mode)[0]+"test"+"_dense.csv", 
                                         args.labels_file.split(args.mode)[0]+"test"+args.labels_file.split(args.mode)[1], 
                                         test_transform_baseline, 
                                         test_transform_wifi,
                                         False,
                                         'test')
        test_loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': config.get('n_workers')}
        test_dataloader = torch.utils.data.DataLoader(test_dataset, **test_loader_params)

        # Get training details
        n_freq_print = config.get("n_freq_print")
        n_freq_checkpoint = config.get("n_freq_checkpoint")
        n_epochs = config.get("n_epochs")

        # Train
        checkpoint_prefix = join(utils.create_output_dir('out'),utils.get_stamp_from_log())
        n_total_samples = 0.0
        loss_vals = []
        sample_count = []
        mean_posit_errs = np.zeros((n_epochs - 251))
        for epoch in range(n_epochs):

            # Resetting temporal loss used for logging
            running_loss = 0.0
            n_samples = 0

            for batch_idx, minibatch in enumerate(train_dataloader):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)
                gt_pose = minibatch.get('pose').to(dtype=torch.float32)
                gt_scene = minibatch.get('scene').to(device)
                batch_size = gt_pose.shape[0]
                n_samples += batch_size
                n_total_samples += batch_size

                if freeze: # For TransPoseNet
                    model.eval()
                    with torch.no_grad():
                        transformers_res = model.forward_transformers(minibatch)
                    model.train()

                # Zero the gradients
                optim.zero_grad()

                # Forward pass to estimate the pose
                if freeze:
                    res = model.forward_heads(transformers_res)
                else:
                    res = model(minibatch)

                est_pose = res.get('pose')
                est_scene_log_distr = res.get('scene_log_distr')
                if est_scene_log_distr is not None:
                    # Pose Loss + Scene Loss
                    criterion = pose_loss(est_pose, gt_pose) + nll_loss(est_scene_log_distr, gt_scene)
                else:
                    # Pose loss
                    criterion = pose_loss(est_pose, gt_pose)

                # Collect for recoding and plotting
                running_loss += criterion.item()
                loss_vals.append(criterion.item())
                sample_count.append(n_total_samples)

                # Back prop
                criterion.backward()
                optim.step()

                # Record loss and performance on train set
                if batch_idx % n_freq_print == 0:
                    posit_err, orient_err = utils.pose_err(est_pose.detach(), gt_pose.detach())
                    logging.info("[Batch-{}/Epoch-{}] running camera pose loss: {:.3f}, "
                                 "camera pose error: {:.2f}[m], {:.2f}[deg]".format(
                                                                        batch_idx+1, epoch+1, (running_loss/n_samples),
                                                                        posit_err.mean().item(),
                                                                        orient_err.mean().item()))

            # Scheduler update
            if use_scheduler == True:
                scheduler.step()

            if epoch > 250:
                if test(test_dataloader, mean_posit_errs, epoch):
                    torch.save(model.state_dict(), checkpoint_prefix + '_checkpoint-{}.pth'.format(epoch))

        logging.info('Training completed')
        torch.save(model.state_dict(), checkpoint_prefix + '_final.pth'.format(epoch))

        # Plot the loss function
        loss_fig_path = checkpoint_prefix + "_loss_fig.png"
        utils.plot_loss_func(sample_count, loss_vals, loss_fig_path)

    else: # Test
        # Set to eval mode
        model.eval()

        # Set the dataset and data loader
        transform_baseline = utils.test_transforms.get('baseline')
        transform_wifi = utils.test_transforms.get('wifi')
        test_dataset = CameraPoseDataset(args.dataset_path, args.rssi_file, args.labels_file, transform_baseline, transform_wifi, False, 'test')
        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': config.get('n_workers')}
        test_dataloader = torch.utils.data.DataLoader(test_dataset, **loader_params)

        stats = np.zeros((len(test_dataloader.dataset), 3))
        posit_errs = []
        pred_poses = np.zeros((len(test_dataloader.dataset), 7))  # store all predicted poses
        targ_poses = np.zeros((len(test_dataloader.dataset), 7))  # store all target poses

        with torch.no_grad():
            for i, minibatch in enumerate(test_dataloader, 0):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)
                gt_scene = minibatch.get('scene')
                minibatch['scene'] = None # avoid using ground-truth scene during prediction

                gt_pose = minibatch.get('pose').to(dtype=torch.float32)

                # Forward pass to predict the pose
                tic = time.time()
                est_pose = model(minibatch).get('pose')
                toc = time.time()

                pred_poses[i, :] = est_pose[0].cpu().data.numpy()
                targ_poses[i, :] = gt_pose[0].cpu().data.numpy()

                # Evaluate error
                posit_err, orient_err = utils.pose_err(est_pose, gt_pose)

                # Collect statistics
                stats[i, 0] = posit_err.item()
                posit_errs.append(stats[i, 0])
                stats[i, 1] = orient_err.item()
                stats[i, 2] = (toc - tic)*1000

                logging.info("Pose error: {:.3f}[m], {:.3f}[deg], inferred in {:.2f}[ms]".format(
                    stats[i, 0],  stats[i, 1],  stats[i, 2]))

        # Record overall statistics
        logging.info("Performance of {} on {}".format(args.checkpoint_path, args.labels_file))
        logging.info("Mean pose error: {:.3f}[m], {:.3f}[deg]".format(np.nanmean(stats[:, 0]), np.nanmean(stats[:, 1])))
        logging.info("Mean inference time:{:.2f}[ms]".format(np.mean(stats[:, 2])))

        # create figure object
        fig = plt.figure()
        x = np.vstack((pred_poses[:, 0].T, targ_poses[:, 0].T))
        y = np.vstack((pred_poses[:, 1].T, targ_poses[:, 1].T))
        if "7scenes" in args.dataset_path:
            ax = fig.add_subplot(111, projection='3d')
            z = np.vstack((pred_poses[:, 2].T, targ_poses[:, 2].T))
            for xx, yy, zz in zip(x.T, y.T, z.T):
                ax.plot(xx, yy, zs=zz, c='b')
            ax.scatter(targ_poses[:, 0], targ_poses[:, 1], zs=targ_poses[:, 2], c='g', depthshade=0)
            ax.scatter(pred_poses[:, 0], pred_poses[:, 1], zs=pred_poses[:, 2], c='r', depthshade=0)
        else:
            ax = fig.add_subplot(111)
            ax.plot(x, y, c='b')
            ax.scatter(targ_poses[:, 0], targ_poses[:, 1], c='g')
            ax.scatter(pred_poses[:, 0], pred_poses[:, 1], c='r')
        plt.show()
        checkpoint_prefix = join(utils.create_output_dir('out'),utils.get_stamp_from_log())
        loss_fig_path = checkpoint_prefix + "_results_fig.png"
        fig.savefig(loss_fig_path)

        # create mat object
        # sio.savemat(join('tools', '6fuser_2d2.mat'), {"out": posit_errs})
