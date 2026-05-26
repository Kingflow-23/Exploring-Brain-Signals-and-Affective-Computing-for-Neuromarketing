from train_deep_models import (
    EEG_CNN_Attention,
    EEG_LSTM,
    EEG_TCN,
    EEG_CNN_LSTM,
)

from braindecode.models import EEGNet, Deep4Net, ShallowFBCSPNet, EEGConformer
from config import WINDOW_SIZE

def get_model(model_name, n_chans=62, n_classes=3):

    input_window_seconds = WINDOW_SIZE / 200

    if model_name == "cnn_attention":
        return EEG_CNN_Attention(n_chans, n_classes)

    if model_name == "lstm":
        return EEG_LSTM(n_chans, n_classes)

    if model_name == "tcn":
        return EEG_TCN(n_chans, n_classes)

    if model_name == "cnn_lstm":
        return EEG_CNN_LSTM(n_chans, n_classes)

    if model_name == "eegnet":
        return EEGNet(
            n_chans=n_chans,
            n_outputs=n_classes,
            input_window_seconds=input_window_seconds,
            sfreq=200,final_conv_length="auto"
        )

    if model_name == "deep4net":
        return Deep4Net(
            n_chans=n_chans,
            n_outputs=n_classes,
            input_window_seconds=input_window_seconds,
            sfreq=200,final_conv_length="auto"
        )

    if model_name == "shallowconv":
        return ShallowFBCSPNet(
            n_chans=n_chans,
            n_outputs=n_classes,
            input_window_seconds=input_window_seconds,
            sfreq=200,
        )

    if model_name == "eegconformer":
        return EEGConformer(
            n_chans=n_chans,
            n_outputs=n_classes,
            input_window_seconds=input_window_seconds,
            sfreq=200,
        )

    raise ValueError(f"Unknown model: {model_name}")