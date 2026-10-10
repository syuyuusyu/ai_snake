from torch import nn

from board_config import DEFAULT_BOARD_HIGH, DEFAULT_BOARD_WIDTH


class SnakeNet(nn.Module):
    def __init__(self, board_width=DEFAULT_BOARD_WIDTH, board_high=DEFAULT_BOARD_HIGH):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.prediction = nn.Sequential(
            nn.Linear(64 * (board_width // 4) * (board_high // 4), 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 4) # 输出4个方向的值，上、下、左、右
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.prediction(x)
        return x
    