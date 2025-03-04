import math
import torch
from utils.modules import LinearBlock


class CNN(torch.nn.Module):
    """
    CNN - Pytorch
    """

    def __init__(self,
                 input_channels=2,
                 num_classes=4,
                 cleaning_blocks=2,
                 backbone_blocks=5,
                 backbone_block_convs=1,
                 backbone_time_reduction_factor=2,
                 backbone_channel_increase_factor=2,
                 conv_crop_size=4,
                 head_hidden_units=1024,
                 head_hidden_layers=2,
                 head_dropout=0,
                 classifier_head=True,
                 period_head=True,
                 cont_head=True
                 # temporary fix
                 ):
        """
        Init
            Builds the pytorch model
        """
        super().__init__()
        self.classifier_head = classifier_head
        self.period_head = period_head
        self.cont_head = cont_head
        self.num_classes = num_classes
        self.conv_crop_size = conv_crop_size
        # Make model
        self.cleaning_backbone = CleaningBackbone(cleaning_blocks, input_channels)
        self.conv_backbone = ConvBackbone(input_channels,
                                          backbone_blocks,
                                          backbone_block_convs,
                                          backbone_time_reduction_factor,
                                          backbone_channel_increase_factor)
        # Pool leftover time dimension down to 1
        self.adaptive_pool = torch.nn.AdaptiveAvgPool3d((1, None, None))
        last_layer_channels = self.conv_backbone.block_input_channels(backbone_blocks,
                                                                input_channels,
                                                                backbone_channel_increase_factor)
        head_input_size = conv_crop_size * conv_crop_size * last_layer_channels

        if classifier_head:
            lin_shapes = [head_input_size] + ([head_hidden_units] * head_hidden_layers)
            lin_layers = []
            ls = len(lin_shapes)

            for i in range(1, ls):
                lin_layers.append(LinearBlock(lin_shapes[i-1], lin_shapes[i], drop_p=head_dropout))

            lin_layers.append(torch.nn.Linear(lin_shapes[ls-1], num_classes))

            self.classifier = torch.nn.Sequential(*lin_layers)

        if period_head:
            self.period_head = torch.nn.Linear(head_input_size, 1)
        
        if cont_head:
            self.cont_head = torch.nn.Linear(head_input_size, 1)

    def forward(self, data):
        """
        forward
            Defines how data goes through the model
        """
        batch_size = data.shape[0]
        data = self.cleaning_backbone(data)
        data = self.conv_backbone(data)
        data = self.adaptive_pool(data)
        shape = data.shape
        half_crop = self.conv_crop_size/2
        data = data[:,:,:,
                        math.floor(shape[3]/2-(half_crop-1)):math.floor(shape[3]/2+(half_crop+1)),
                        math.floor(shape[4]/2-(half_crop-1)):math.floor(shape[4]/2+(half_crop+1))]
        data = data.reshape(shape[0],-1)

        # If classifier_head is flase, then a tensor of zeros is returned. 
        if self.classifier_head:
            class_pred = self.classifier(data)

        else:
            class_pred = torch.zeros((batch_size, self.num_classes)).to(data.device)
        
        # If period_head is flase, then a tensor of zeros is returned.
        if self.period_head:
            period_pred = self.period_head(data)
        else:
            period_pred = torch.zeros((batch_size, 1)).to(data.device)
        
        if self.cont_head:
            cont_pred = self.cont_head(data)
        else:
            cont_pred = torch.zeros((batch_size, 1)).to(data.device)

        return class_pred, period_pred, cont_pred


class CleaningBackbone(torch.nn.Module):
    """
    Cleaning Backbone
        Builds the resnet blocks for the front of the model
    """
    def __init__(self, num_cleaning_blocks, input_channels):
        """
        Init
            Stacks the cleaning blocks and returns them
        """
        super().__init__()
        blocks = []
        for _ in range(num_cleaning_blocks):
            blocks.append(CleaningBlock(input_channels))
        self.backbone = torch.nn.Sequential(*blocks)

    def forward(self, data):
        """
        forward
            Defines how data goes through the cleaning backbone
        """
        return self.backbone(data)


class CleaningBlock(torch.nn.Module):
    """
    Cleaning (ResNet Full Pre-Activation) Block
    """
    def __init__(self, num_channels, kernel_size=3, stride=1, padding=1):
        """
        Init
            Makes a single resnet block
        """
        super().__init__()
        self.bn1   = torch.nn.BatchNorm3d(num_channels)
        self.relu  = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv3d(num_channels,
                                     num_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding)
        self.bn2   = torch.nn.BatchNorm3d(num_channels)
        self.conv2 = torch.nn.Conv3d(num_channels,
                                     num_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding)

    def forward(self, data):
        """
        forward
            Defines how data goes through a cleaning block
        """
        residual = data
        data = self.bn1(data)
        data = self.relu(data)
        data = self.conv1(data)
        data = self.bn2(data)
        data = self.relu(data)
        data = self.conv2(data)
        data += residual
        return data


class ConvBackbone(torch.nn.Module):
    """
    Conv Backbone
        The stack of conv blocks that compress time and increase channels
    """

    def __init__(self,
                 input_channels,
                 backbone_blocks,
                 backbone_block_convs,
                 backbone_time_reduction_factor,
                 backbone_channel_increase_factor):
        """
        Init
            Stacks the conv blocks into a backbone
        """
        super().__init__()
        layers = []
        # For each block
        for block_idx in range(backbone_blocks):
            block_input_channels  = self.block_input_channels(block_idx,
                                                                input_channels,
                                                                backbone_channel_increase_factor)
            block_output_channels = self.block_input_channels(block_idx+1,
                                                                input_channels,
                                                                backbone_channel_increase_factor)
            # Add sub blocks, keep number of channels the same
            for _ in range(backbone_block_convs-1):
                layers.append(ConvBlock(input_channels=block_input_channels,
                                        output_channels=block_input_channels,
                                        ))
            # Add time reduction block, increase number of channels
            layers.append(ConvBlock(input_channels=block_input_channels,
                                    output_channels=block_output_channels,
                                    ))
            # Pool to reduce time dimension
            layers.append(torch.nn.MaxPool3d(kernel_size=(backbone_time_reduction_factor,1,1),
                                             stride=(backbone_time_reduction_factor,1,1)))
        self.backbone = torch.nn.Sequential(*layers)

    def forward(self, data):
        """
        forward
            Defines how data goes through a cleaning block
        """
        return self.backbone(data)

    def block_input_channels(self, block_idx, input_channels, backbone_channel_increase_factor):
        """
        block_input_channels
            Formula for easily calculating the number of input/output
            channels for a given block layer
        """
        return input_channels * (backbone_channel_increase_factor ** block_idx)


class ConvBlock(torch.nn.Module):
    """
    Conv Block
        A single time reduction and channel increase block
    """
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size=3, stride=1, padding=1):
        """
        Init
            Builds a single conv backbone block
        """
        super().__init__()
        layers = []
        layers.append(torch.nn.Conv3d(input_channels,
                                        output_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding))
        layers.append(torch.nn.BatchNorm3d(output_channels))
        layers.append(torch.nn.ReLU())
        self.sequential = torch.nn.Sequential(*layers)

    def forward(self, data):
        """
        forward
            Defines how data goes through a conv backbone block
        """
        return self.sequential(data)
