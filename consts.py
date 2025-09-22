from dataclasses import dataclass

GAME_TOTAL_TICKS = 3 * 60 * 10


@dataclass
class TeamColor:
    RED = "R"
    BLUE = "B"
    NOOP = "N"


@dataclass
class PlayerState:
    GOOD = "A"
    BAD = "D"


@dataclass
class Direction:
    NOOP = "N"
    UP = "U"
    DOWN = "D"
    LEFT = "L"
    RIGHT = "R"


@dataclass
class Terrain:
    PLAIN = "P"
    SPEED_UP = "B"
    SPEED_DOWN = "M"
    OBSTACLE = "I"
    NOOP = "N"
    DESTROYABLE_OBSTACLE = "D"


@dataclass
class ItemType:
    BOMB = "BP"
    POTION = "SP"
    SHOE = "AB"


@dataclass
class Map:
    WIDTH = 28
    HEIGHT = 16
    BLOCK_SIZE = 50
