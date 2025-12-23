import os
from datetime import datetime
from collections import defaultdict
from itertools import islice
import random
from pathlib import Path
import warnings
import numpy as np
from tqdm import tqdm, trange
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig, OmegaConf

from utils import *
from get_ae import *
from Classifier import MNIST_Classifier
from mutinfo.torch.layers import AdditiveGaussianNoise
import mutinfo.estimators.mutual_information as mi_estimators

warnings.filterwarnings("ignore")
# plt.style.use('dark_background')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32

optimizer_dict = {
    'AdamW': torch.optim.AdamW,
    'Adam': torch.optim.Adam,
    'SGD': torch.optim.SGD
}
activation_dict = {
    'ReLU': nn.ReLU,
    'Tanh': nn.Tanh,
    'Sigmoid': nn.Sigmoid,
    'GELU': nn.GELU,
    'LeakyReLU': nn.LeakyReLU,
}
loss_function_dict = {
    'MSE': nn.MSELoss,
    'CrossEntropy': nn.CrossEntropyLoss
}

@hydra.main(
    config_path='./configs',
    config_name="default",
    version_base="1.3.2" # version_base>=1.2 - for not changing cwd !
)
def main(cfg: DictConfig):
    torch.set_default_dtype(dtype)

    # Seed everything
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Create unique run_name, prepare TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.custom_run_name}_{timestamp}" if cfg.custom_run_name else f"run_{timestamp}"
    log_dir = Path("runs") / run_name
    writer = SummaryWriter(log_dir=log_dir)

    # Save config to TensorBoard as text
    config_text = OmegaConf.to_yaml(cfg)
    writer.add_text("config", config_text)

    print(config_text)

    # Filepaths for Information Plane plot
    dd_figure_paths = {}
    for data_subset_name in ['train', 'eval']:
        figure_dir = Path("figures") / run_name
        figure_dir.mkdir(parents=True, exist_ok=True)
        figure_filename = f"InfoPlane-{run_name}_{data_subset_name}.png"
        dd_figure_paths[data_subset_name] = str(figure_dir / figure_filename)

    """
    Data
    """
    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    # Load dataset
    train = torchvision.datasets.MNIST(
        root=cfg.download_directory, train=True,
        transform=image_transform,
        download=True
    )
    test = torchvision.datasets.MNIST(
        root=cfg.download_directory, train=False,
        transform=image_transform,
        download=True
    )
    train_subset_indices = np.random.choice(a=len(train), size=cfg.train.train_points, replace=False)
    train = torch.utils.data.Subset(train, train_subset_indices)

    """
    More workers make error in L_Y_mi_estimator.estimate(L_compressed, targets, verbose=0)
    """
    train_loader = torch.utils.data.DataLoader(train, batch_size=cfg.train.batch_size, shuffle=True,
                                               num_workers=0,
                                               pin_memory=True,
                                               )
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=cfg.train.test_batch_size, shuffle=False,
                                                  num_workers=0,
                                                  pin_memory=True,
                                                  )
    eval_dataloader = test_dataloader
    train_loader_nonshuffle = torch.utils.data.DataLoader(train, batch_size=cfg.train.batch_size, shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               )
    dd_targets_mi = {}
    dd_targets_mi['eval'] = eval_dataloader.dataset.targets.detach().cpu().numpy()
    # dd_targets_mi['train'] = train_loader_nonshuffle.dataset.targets.detach().cpu().numpy()
    dd_targets_mi['train'] = np.array([train.dataset[i][1] for i in train.indices])

    # Prepare X for MI evaluation
    if cfg.mi.compression == 'smi':
        """
        Project onto vectors on a sphere
        """
        # TODO: move n_projections to yaml config
        projector_X = smi_compressor((train.dataset[0][0]).flatten().shape[0], n_projections=1000)
        dd_X_projected = {}

        X = np.array([train.dataset[i][0].flatten().numpy() for i in train.indices])
        dd_X_projected['train'] = projector_X(X)
        X = np.array([x_[0].flatten().numpy() for x_ in test])
        dd_X_projected['eval'] = projector_X(X)
    else:
        """
        Load autoencoder and compress X
        """
        X_autoencoder = Autoencoder(
            MNIST_ConvEncoder(latent_dim=cfg.ae.L_latent_dim),
            MNIST_ConvDecoder(latent_dim=cfg.ae.L_latent_dim)
        ).to(device)
        # TODO: add training from mutinfo lib
        # For training autoencoder see https://github.com/VanessB/Information-v3
        load_X_autoencoder(
            model_ae=X_autoencoder,
            encoder_path=cfg.ae.encoder_path,
            decoder_path=cfg.ae.decoder_path,
            device=device
        )
        X_autoencoder.agn.enabled_on_inference = False

        dd_X_compressed = {}
        dd_X_compressed['train'] = get_outputs(X_autoencoder.encoder, train_loader_nonshuffle, device).numpy()
        dd_X_compressed['eval'] = get_outputs(X_autoencoder.encoder, eval_dataloader, device).numpy()

        del X_autoencoder

    """
    Model, optimizer, loss
    """
    model_clf = MNIST_Classifier(
        width=cfg.model.width,
        activation_fn=activation_dict[cfg.model.activation],
        sigma=cfg.mi.sigma,
        relative_scale=cfg.mi.agn_relative_scale
    ).to(device)
    # Initialize model
    with torch.no_grad():
        for p in model_clf.parameters():
            p.data = cfg.model.initialization_scale * p.data
    # Create optimizer
    assert cfg.train.optimizer in optimizer_dict, f"Unsupported optimizer choice: {cfg.train.optimizer}"
    optimizer = optimizer_dict[cfg.train.optimizer](
        model_clf.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay
    )
    # Define loss function
    assert cfg.train.loss_function in loss_function_dict
    loss_fn = loss_function_dict[cfg.train.loss_function]()
    # Autoencoders.
    L_autoencoders = dict()
    # Mutual information.
    dd_MI_X_L = {}
    dd_MI_X_L['train'] = defaultdict(list)
    dd_MI_X_L['eval'] = defaultdict(list)
    dd_MI_L_Y = {}
    dd_MI_L_Y['train'] = defaultdict(list)
    dd_MI_L_Y['eval'] = defaultdict(list)
    # Turn off filtering
    filtered_MI_X_L = None
    filtered_MI_L_Y = None
    # Lists for logs
    log_steps = []
    # Training
    steps = 0
    one_hots = torch.eye(10, 10).to(device)
    # # Save variance of the output of layers for debugging
    # dd_df_bul2 = {}

    """
    Train loop!
    """
    with tqdm(total=cfg.train.optimization_steps) as pbar:
        for x, labels in islice(cycle(train_loader), cfg.train.optimization_steps):
            if (steps < cfg.train.n_first_steps_to_log) or (steps % cfg.train.freqLog == 0):
                # Log loss, acc
                # TODO: rewrite to compute everything at once
                train_loss_, train_acc_ = (compute_loss_accuracy(model_clf, train, cfg.train.loss_function, device, N=len(train), batch_size=256))
                test_loss_, test_acc_ = (compute_loss_accuracy(model_clf, test, cfg.train.loss_function, device, N=len(test), batch_size=256))
                # test_loss_ = (compute_loss(model_clf, test, cfg.train.loss_function, device, N=len(test), batch_size=256))
                # train_acc_ = (compute_loss_accuracy(model_clf, train, device, N=len(train)))
                # test_acc_ = (compute_accuracy(model_clf, test, device, N=len(test)))

                # Used later for InfoPlane plotting
                log_steps.append(steps)
                # Log scalars to TensorBoard
                writer.add_scalar("Loss/train", train_loss_, steps)
                writer.add_scalar("Loss/test", test_loss_, steps)
                writer.add_scalar("Acc/train", train_acc_, steps)
                writer.add_scalar("Acc/test", test_acc_, steps)

                # Log WN (L2 of n-th layer weights)
                wn_layers = []
                wn_layers.append(get_float_wn(model_clf.linear_1.parameters()))
                wn_layers.append(get_float_wn(model_clf.linear_2.parameters()))
                wn_layers.append(get_float_wn(model_clf.linear_3.parameters()))
                wn_total = sum(wn_layers)
                for i, wn_layer in enumerate(wn_layers):
                    writer.add_scalar(f"WN/linear_{i+1}", wn_layer, steps)
                writer.add_scalar("WN/total", wn_total, steps)

                pbar.set_description("L: {0:1.1e}|{1:1.1e}. A: {2:2.1f}%|{3:2.1f}%".format(
                    train_loss_,
                    test_loss_,
                    train_acc_ * 100,
                    test_acc_ * 100)
                )
                # Log custom stats
                # Set apply_agn=False for computing num_dead_nrns stats
                eval_outputs = get_layers(model_clf, eval_dataloader, device, apply_agn=False)
                train_outputs = get_layers(model_clf, train_loader_nonshuffle, device, apply_agn=False)
                outputs_dict = {
                    'train': train_outputs,
                    'eval': eval_outputs,
                }

                for data_subset_name, outputs_subset in outputs_dict.items():
                    for layer_name in outputs_subset.keys():
                        logits_layer = outputs_subset[layer_name].numpy()
                        # Entropy logging is not formulated well here
                        # H_layer_data = entropy(logits_layer, base=2, axis=1)
                        # Never activating on the whole data subset
                        num_dead_nrns = (logits_layer.sum(axis=0) == 0).sum()

                        # Log custom stats
                        # writer.add_scalar(f"H_logits_data_mean_{data_subset_name}/{layer_name}", H_layer_data.mean(), steps)
                        # writer.add_scalar(f"H_logits_data_var_{data_subset_name}/{layer_name}", H_layer_data.var(), steps)
                        writer.add_scalar(f"logits_mean_{data_subset_name}/{layer_name}", logits_layer.mean(), steps)
                        writer.add_scalar(f"logits_var_{data_subset_name}/{layer_name}", logits_layer.var(), steps)
                        writer.add_scalar(f"num_dead_nrns_{data_subset_name}/{layer_name}", num_dead_nrns, steps)

                # Get layers outputs for MI estimation
                # Set apply_agn=True (default) for computing MI
                eval_outputs = get_layers(model_clf, eval_dataloader, device)
                train_outputs = get_layers(model_clf, train_loader_nonshuffle, device)
                outputs_dict = {
                    'train': train_outputs,
                    'eval': eval_outputs,
                }
                # ------------------------------------
                # Mutual information estimation start
                for data_subset_name, outputs_subset in outputs_dict.items():
                    for layer_name in outputs_subset.keys():
                        this_L_latent_dim = min(
                            cfg.ae.L_latent_dim,
                            torch.numel(outputs_subset[layer_name]) / outputs_subset[layer_name].shape[0]
                        )
                        # Very simple compression
                        if cfg.mi.compression == 'first_coords':
                            L_compressed = outputs_subset[layer_name].numpy()
                            L_compressed = np.reshape(L_compressed, (L_compressed.shape[0], -1))
                            L_compressed = L_compressed[:, :this_L_latent_dim]
                        # PCA compression
                        elif cfg.mi.compression == 'pca':
                            # get output from a layer
                            L_compressed = outputs_subset[layer_name].numpy()
                            # reshape it
                            L_compressed = np.reshape(L_compressed, (L_compressed.shape[0], -1))
                            # compress the output from the layer with PCA
                            L_compressed = PCA(n_components=this_L_latent_dim, whiten=cfg.mi.pca.whiten_flag).fit_transform(L_compressed)
                        # SMI "compression", i.e. the projection
                        elif cfg.mi.compression == 'smi':
                            L_projected = outputs_subset[layer_name].numpy()
                            smi_compressor_L = smi_compressor(L_projected.shape[1], n_projections=1000)
                            L_projected = smi_compressor_L(L_projected)

                        # TODO: Make compression of intermediate layers with AE working
                        elif cfg.mi.compression == 'autoencoders':
                            raise NotImplementedError
                            print(f"Training an autoencoder for the layer {layer_name}")
                            # Datasets.
                            train_layer = train_outputs[layer_name]
                            test_layer = test_outputs[layer_name]
                            eval_layer = outputs_subset[layer_name]

                            L_train_dataset = torch.utils.data.TensorDataset(train_layer, train_layer)
                            L_test_dataset = torch.utils.data.TensorDataset(test_layer, test_layer)
                            L_eval_dataset = torch.utils.data.TensorDataset(eval_layer, eval_layer)

                            L_train_dataloader = torch.utils.data.DataLoader(L_train_dataset, batch_size=cfg.train.batch_size_train,
                                                                                shuffle=True)
                            L_test_dataloader = torch.utils.data.DataLoader(L_test_dataset, batch_size=cfg.train.batch_size_test,
                                                                            shuffle=False)
                            L_eval_dataloader = torch.utils.data.DataLoader(L_eval_dataset, batch_size=cfg.train.batch_size_test,
                                                                            shuffle=False)

                            # Autoencoder.
                            if layer_name in L_autoencoders.keys():
                                L_autoencoder = L_autoencoders[layer_name]
                            else:
                                print(f"Could not find an autoencoder for the layer {layer_name}.")
                                L_dim = train_layer.shape[1]
                                L_autoencoder = Autoencoder(DenseEncoder(input_dim=L_dim, latent_dim=this_L_latent_dim),
                                                            DenseDecoder(latent_dim=this_L_latent_dim,
                                                                            output_dim=L_dim)).to(device)

                            # Training.
                            L_results = train_autoencoder(L_autoencoder, L_train_dataloader, L_test_dataloader,
                                                            torch.nn.MSELoss(),
                                                            device, n_epochs=L_autoencoder_n_epochs) # todo: fix ae for layers training
                            L_autoencoders[layer_name] = L_autoencoder

                            _baseline_PCA = PCA(n_components=this_L_latent_dim).fit(
                                np.reshape(train_layer, (train_layer.shape[0], -1)))
                            _baseline_layer = _baseline_PCA.inverse_transform(_baseline_PCA.transform(test_layer))
                            baseline_loss = float(torch.nn.functional.mse_loss(test_layer, torch.tensor(_baseline_layer)))

                            print(
                                f"Train loss: {L_results['train_loss'][-1]:.2e}; test loss: {L_results['test_loss'][-1]:.2e}")
                            print(
                                f"Better then PCA: {baseline_loss:.2e} / {L_results['test_loss'][-1]:.2e} = {baseline_loss / L_results['test_loss'][-1]:.2f}")

                            L_compressed = get_outputs(L_autoencoder.encoder, L_eval_dataloader, device).numpy()
                            # L_compressed = PCA(n_components=L_latent_dim).fit_transform(np.reshape(layer, (layer.shape[0], -1)))

                        if cfg.mi.compression == 'smi':
                            # (X,L)
                            X_L_mi_estimator = mi_estimators.MutualInfoEstimator(
                                entropy_estimator_params=cfg.mi.entropy_estimator_params,
                            )
                            mi_slices = Parallel(n_jobs=-1) \
                                (delayed(measure_smi_projection)(X_L_mi_estimator, x_.reshape(-1, 1),l_.reshape(-1, 1)) \
                                    for x_, l_ in zip(dd_X_projected[data_subset_name],L_projected))
                            X_L_mi_ = np.mean(mi_slices, axis=0)
                            dd_MI_X_L[data_subset_name][layer_name].append(X_L_mi_)
                            # (L,Y)
                            L_Y_mi_estimator = mi_estimators.MutualInfoEstimator(
                                Y_is_discrete=True,
                                entropy_estimator_params=cfg.mi.entropy_estimator_params
                            )
                            y_ = dd_targets_mi[data_subset_name]
                            mi_slices = Parallel(n_jobs=-1) \
                                (delayed(measure_smi_projection)(L_Y_mi_estimator, l_.reshape(-1, 1), y_) \
                                    for l_ in L_projected)
                            L_Y_mi_ = np.mean(mi_slices, axis=0)
                            dd_MI_L_Y[data_subset_name][layer_name].append(L_Y_mi_)
                        else:
                            # (X,L)
                            X_L_mi_estimator = mi_estimators.MutualInfoEstimator(
                                entropy_estimator_params=cfg.mi.entropy_estimator_params)
                            X_L_mi_estimator.fit(dd_X_compressed[data_subset_name], L_compressed, verbose=0)
                            # tuple of size (2) : X_L_mi_[0] - value, X_L_mi_[1] - error
                            X_L_mi_ = X_L_mi_estimator.estimate(dd_X_compressed[data_subset_name], L_compressed, verbose=0)
                            dd_MI_X_L[data_subset_name][layer_name].append(X_L_mi_)
                            # (L,Y)
                            L_Y_mi_estimator = mi_estimators.MutualInfoEstimator(
                                Y_is_discrete=True,
                                entropy_estimator_params=cfg.mi.entropy_estimator_params)
                            L_Y_mi_estimator.fit(L_compressed, dd_targets_mi[data_subset_name], verbose=0)
                            # tuple of size (2) : L_Y_mi_[0] - value, L_Y_mi_[1] - error
                            L_Y_mi_ = L_Y_mi_estimator.estimate(L_compressed, dd_targets_mi[data_subset_name], verbose=0)
                            dd_MI_L_Y[data_subset_name][layer_name].append(L_Y_mi_)
                        # Log Mutual Information metrics
                        writer.add_scalar(f"MI(X;L)_{data_subset_name}/{layer_name}", X_L_mi_[0], steps)
                        writer.add_scalar(f"MI(L;Y)_{data_subset_name}/{layer_name}", L_Y_mi_[0], steps)
                        writer.add_scalar(f"MI(X;L)_Errors_{data_subset_name}/{layer_name}", X_L_mi_[1], steps)
                        writer.add_scalar(f"MI(L;Y)_Errors_{data_subset_name}/{layer_name}", L_Y_mi_[1], steps)

                    # Plotting Information Plane
                    figure_path = dd_figure_paths[data_subset_name]
                    plot_MI_planes(dd_MI_X_L[data_subset_name],
                                    dd_MI_L_Y[data_subset_name],
                                    figure_path=figure_path,
                                    filtered_MI_X_L=filtered_MI_X_L, filtered_MI_L_Y=filtered_MI_L_Y, log_steps=log_steps)
                    # Log image to TensorBoard
                    if os.path.exists(figure_path):
                        image = plt.imread(figure_path)
                        # TensorBoard expects image of shape (C, H, W)
                        writer.add_image(f"InfoPlane/{data_subset_name}", image.transpose(2, 0, 1), steps)

                # Mutual information estimation end
                # ------------------------------------

            optimizer.zero_grad()
            # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                y = model_clf(x.to(device))
                if cfg.train.loss_function == 'CrossEntropy':
                    loss = loss_fn(y, labels.to(device))
                elif cfg.train.loss_function == 'MSE':
                    loss = loss_fn(y, one_hots[labels])

            loss.backward()
            optimizer.step()

            writer.add_scalar("Loss/train_optimizer", loss.cpu().detach().item(), steps)

            steps += 1
            pbar.update(1)

    # Finally log Information Plane
    for data_subset_name, figure_path in dd_figure_paths.items():
        if os.path.exists(figure_path):
            image = plt.imread(figure_path)
            writer.add_image(f"InfoPlane_Final/{data_subset_name}", image.transpose(2, 0, 1), steps)

    # Closing TensorBoard
    writer.close()

if __name__ == "__main__":
    main()
