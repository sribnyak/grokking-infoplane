from pathlib import Path

from autoencoders.utils import *
from autoencoders.autoencoder import *

X_autoencoder_path = Path("./autoencoders/").resolve()

def load_X_autoencoder(model_ae: Autoencoder, encoder_path, decoder_path, device):
    model_ae.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    model_ae.decoder.load_state_dict(torch.load(decoder_path, map_location=device))
