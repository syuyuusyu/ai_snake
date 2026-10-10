"""Shared board defaults for training, evaluation, playback and export.

Set defaults here, or override once at an entry point using --board-width and
--board-high. Forward both values to every environment, evaluator and cache.
Observations retain the project's existing (channels, x, y) axis order.
Existing model checkpoints require a compatible observation shape.
"""

DEFAULT_BOARD_WIDTH = 12
DEFAULT_BOARD_HIGH = 12


def validate_board_dimensions(board_width, board_high):
    # The initial snake has four horizontal cells and needs room for food.
    if (not isinstance(board_width, int) or not isinstance(board_high, int)
            or board_width < 4 or board_high < 2):
        raise ValueError("board_width must be an integer >= 4; board_high must be an integer >= 2")


def add_board_arguments(parser):
    parser.add_argument("--board-width", type=int, default=DEFAULT_BOARD_WIDTH)
    parser.add_argument("--board-high", type=int, default=DEFAULT_BOARD_HIGH)


def observation_scale(board_width, board_high):
    validate_board_dimensions(board_width, board_high)
    shortest = min(board_width, board_high)
    # Preserve the existing scaling while meeting NatureCNN's 36-pixel minimum.
    return max(1, (32 + shortest - 1) // shortest, (36 + shortest + 1) // (shortest + 2))
