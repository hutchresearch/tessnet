from models.cnn import CNN
from models.cnn_2 import CNN2
from models.lstm_3d import LSTM3d
from models.lstm_2d import LSTM2d


def initialize_model(model_cfg, state_dict=None):
    if model_cfg.architecture == 'cnn':
        model = CNN(
            input_channels=model_cfg.input_channels,
            num_classes=model_cfg.num_classes,
            cleaning_blocks=model_cfg.cleaning_blocks,
            backbone_blocks=model_cfg.backbone_blocks,
            backbone_block_convs=model_cfg.backbone_block_convs,
            backbone_time_reduction_factor=model_cfg.backbone_time_reduction_factor,
            backbone_channel_increase_factor=model_cfg.backbone_channel_increase_factor,
            head_hidden_units=model_cfg.head_hidden_units,
            head_hidden_layers=model_cfg.head_hidden_layers,
            conv_crop_size=model_cfg.conv_crop_size,
            head_dropout=model_cfg.head_dropout,
            classifier_head=model_cfg.classifier_head,
            period_head=model_cfg.period_head,
            cont_head=model_cfg.cont_head
        )
    elif model_cfg.architecture == 'cnn_2':
        model = CNN2(
            input_channels=model_cfg.input_channels,
            num_classes=model_cfg.num_classes,
            cleaning_blocks=model_cfg.cleaning_blocks,
            backbone_blocks=model_cfg.backbone_blocks,
            backbone_block_convs=model_cfg.backbone_block_convs,
            backbone_time_reduction_factor=model_cfg.backbone_time_reduction_factor,
            backbone_channel_increase_factor=model_cfg.backbone_channel_increase_factor,
            head_hidden_units=model_cfg.head_hidden_units,
            head_hidden_layers=model_cfg.head_hidden_layers,
            conv_crop_size=model_cfg.conv_crop_size,
            head_dropout=model_cfg.head_dropout,
            classifier_head=model_cfg.classifier_head,
            period_head=model_cfg.period_head
        )
    elif model_cfg.architecture == 'lstm_3d':
        model = LSTM3d(
            classifier_head=model_cfg.classifier_head,
            period_head=model_cfg.period_head,
            num_classes=model_cfg.num_classes,
            re_zero=model_cfg.re_zero
        )
    elif model_cfg.architecture == 'lstm_2d':
        model = LSTM2d(
            classifier_head=model_cfg.classifier_head,
            period_head=model_cfg.period_head,
            num_classes=model_cfg.num_classes,
            re_zero=model_cfg.re_zero
        )
    else:
        raise ValueError

    if state_dict:
        model.load_state_dict(state_dict, strict=True)

    return model
