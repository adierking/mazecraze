#!/usr/bin/env python3

import argparse
import multiprocessing
import random
import os
import sys
import time

from enum import IntEnum
from dataclasses import dataclass
from typing import Self, assert_never

# Number of rows and columns stored in a maze. The actual displayed maze is 25x39
# because the bottom row is always solid and the right row is always empty.
ROWS = 24
COLS = 40

# Direction values
RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3

# Rotation values
CW = 1
CCW = 3

# Maze data offsets from the start of RAM
PF0_BASE = 0x0
PF1_L_BASE = 0xC
PF2_L_BASE = 0x18
PF1_R_BASE = 0x24
PF2_R_BASE = 0x30

# Total size of the packed maze data
PACKED_MAZE_SIZE = 0x3C

# Maximum size of the walk queue
QUEUE_MAX = 17

# Number of cells that must be walked before adding to the walk queue
CELLS_PER_QUEUE = 10

# Maximum number of times to retry the initial walk
MAX_RETRIES = 6

# For lack of a better term, we'll call this the "level table". This is used to select a
# "level" each time the walker rerandomizes. Higher levels have longer hallways, lower
# levels have more twists and turns.
LEVEL_TABLE = [1, 1, 2, 3, 2, 2, 3, 3, 1, 3, 2, 1, 3, 1, 1, 1]

# Upon booting the game, it chooses one of these seeds as the initial seed.
STARTING_SEEDS = [
    0xFF00,
    0x0100,
    0x0200,
    0x0300,
    0x0400,
    0x0500,
    0x0600,
    0x0700,
    0xFE00,
    0x0900,
    0x0A00,
    0x2800,
    0xFC00,
    0xFA00,
    0x0E00,
    0x0F00,
]


def flip(direction: int) -> int:
    """Flip a direction 180 degrees."""
    return direction ^ 2


def rotate(direction: int, rotation: int) -> int:
    """Rotate a direction either clockwise (CW) or counterclockwise (CCW)."""
    return (direction + rotation) & 3


def move(row: int, col: int, direction: int, steps: int) -> tuple[int, int]:
    """Move (row, col) in a direction and return the new (row, col)."""
    # sub_FA1F in the game, except we allow moving more than one step for simplicity.
    if direction == RIGHT:
        return row, col + steps
    elif direction == DOWN:
        return row - steps, col
    elif direction == LEFT:
        return row, col - steps
    elif direction == UP:
        return row + steps, col
    else:
        assert False, f"invalid direction: {direction}"


class Tile(IntEnum):
    """A grid tile."""

    OPEN = 0
    CLOSED = 1
    # The following types are added by us for convenience.
    # They become OPEN when the maze is packed.
    INVALID = -1
    START = 2
    SOLUTION = 3


def pack_bits(bits: list[int], start: int, end: int) -> int:
    """Pack a list of ones and zeros into an integer."""
    assert all(b == 0 or b == 1 for b in bits)
    if end >= start:
        assert len(bits) == end - start + 1
        return sum(b << (start + i) for i, b in enumerate(bits))
    else:
        assert len(bits) == start - end + 1
        return sum(b << (start - i) for i, b in enumerate(bits))


def unpack_bits(x: int, start: int, end: int) -> list[int]:
    """Unpack part of an integer into a list of ones and zeros."""
    if end >= start:
        return [x >> i & 1 for i in range(start, end + 1)]
    else:
        return [x >> i & 1 for i in range(start, end - 1, -1)]


class Random:
    """Maze Craze random number generator."""

    seed: int

    def __init__(self, seed: int) -> None:
        """Initialize a random number generator with the given seed."""
        self.seed = seed & 0xFFFF

    def next(self) -> int:
        """Update the seed and return a random byte."""
        # sub_F8FD in the game
        self.seed += (self.seed * 32772 & 0xFF00) + (self.seed << 2 & 0xFF) + 13849
        self.seed &= 0xFFFF
        return self.seed >> 8


class Maze:
    """
    A 40x24 grid of tiles which can be indexed by (row, column).

    Row 0 is the bottommost row, and column 0 is the leftmost column.
    """

    # We store the grid as a list of tiles rather than the packed format the
    # game actually uses because it's easier to understand and lets us annotate
    # the maze with extra info (e.g. the solution path).
    grid: list[Tile]

    def __init__(self, grid: list[Tile] | None = None) -> None:
        """Initialize a maze, optionally with a predefined grid."""
        self.grid = grid or self.default_grid()
        assert len(self.grid) == ROWS * COLS

    @classmethod
    def default_grid(cls) -> list[Tile]:
        """
        Return a grid with every 2x2 tile set to 0b11.

        Due to how the packing works, even rows will have even cells set and
        odd rows will be solid. Refer to pack()/unpack().
        """
        return cls.unpack(b"\xFF" * PACKED_MAZE_SIZE).grid

    def pack(self) -> bytearray:
        """Pack the maze into the game's in-memory format."""
        data = bytearray(PACKED_MAZE_SIZE)
        for row in range(0, ROWS):
            # Map any non-CLOSED cells to 0 so we end up with only 0 and 1.
            cells = list(map(lambda t: 1 if t == Tile.CLOSED else 0, self.row(row)))

            # Left half
            pf0 = pack_bits(cells[0:4], 4, 7)
            pf1 = pack_bits(cells[4:12], 7, 0)
            pf2 = pack_bits(cells[12:20], 0, 7)

            # Right half
            pf0 |= pack_bits(cells[20:24], 0, 3)
            pf1 |= pack_bits(cells[24:32], 15, 8)
            pf2 |= pack_bits(cells[32:40], 8, 15)

            # Mask off tiles that are ignored in this row
            pf0 &= 0x55 << (row & 1)
            pf1 &= 0xAAAA >> (row & 1)
            pf2 &= 0x5555 << (row & 1)

            # Merge into the data (which may contain tiles from the previous row)
            offset = row // 2
            data[PF0_BASE + offset] |= pf0
            data[PF1_L_BASE + offset] |= pf1 & 0xFF
            data[PF1_R_BASE + offset] |= pf1 >> 8
            data[PF2_L_BASE + offset] |= pf2 & 0xFF
            data[PF2_R_BASE + offset] |= pf2 >> 8
        return data

    @classmethod
    def unpack(cls, data: bytes) -> Self:
        """Unpack a maze from the game's in-memory format."""
        assert len(data) >= PACKED_MAZE_SIZE
        grid: list[Tile] = []
        for row in range(0, ROWS):
            # The left and right are technically stored separately, but loading them
            # both into a 16-bit integer makes the bitmasking convenient.
            offset = row // 2
            pf0 = data[PF0_BASE + offset]
            pf1 = data[PF1_L_BASE + offset] | (data[PF1_R_BASE + offset] << 8)
            pf2 = data[PF2_L_BASE + offset] | (data[PF2_R_BASE + offset] << 8)

            if row & 1 == 1:
                # Odd row - every other tile is a wall
                pf0 |= 0x55
                pf1 |= 0xAAAA
                pf2 |= 0x5555
            else:
                # Even row - every other tile is empty
                pf0 &= 0x55
                pf1 &= 0xAAAA
                pf2 &= 0x5555

            # Left half
            grid.extend(map(Tile, unpack_bits(pf0, 4, 7)))
            grid.extend(map(Tile, unpack_bits(pf1, 7, 0)))
            grid.extend(map(Tile, unpack_bits(pf2, 0, 7)))

            # Right half
            grid.extend(map(Tile, unpack_bits(pf0, 0, 3)))
            grid.extend(map(Tile, unpack_bits(pf1, 15, 8)))
            grid.extend(map(Tile, unpack_bits(pf2, 8, 15)))
        return cls(grid)

    def get(self, row: int, col: int) -> Tile:
        """
        Return the tile at (row, col),
        where row 0 is at the bottom and col 0 is at the left.
        """
        # sub_F9F1 in the game, except we store the maze differently.
        return self.grid[row * COLS + col]

    def set(self, row: int, col: int, tile: Tile) -> None:
        """
        Set the tile at (row, col),
        where row 0 is at the bottom and col 0 is at the left.
        """
        self.grid[row * COLS + col] = tile

    def row(self, index: int) -> list[Tile]:
        """Return the tiles in the given row, where row 0 is at the bottom."""
        return self.grid[index * COLS : index * COLS + COLS]

    def solve(self, row: int, col: int, visited: list[bool] | None = None) -> int:
        """Solve the maze and return the number of moves needed to complete it."""
        # This is a simple recursive solving algorithm.
        # We assume the maze has only one solution and the exit is on the right side.
        if col == COLS - 1:
            return 1
        elif row < 0 or row >= ROWS or col < 0 or col >= COLS:
            return 0
        if visited is None:
            visited = [False] * (ROWS * COLS)
        elif visited[row * COLS + col]:
            return 0
        visited[row * COLS + col] = True
        tile = self.get(row, col)
        if tile == Tile.CLOSED:
            return 0
        length = self.solve(row, col - 1, visited)
        if length == 0:
            length = self.solve(row, col + 1, visited)
            if length == 0:
                length = self.solve(row + 1, col, visited)
                if length == 0:
                    length = self.solve(row - 1, col, visited)
        if length > 0:
            length += 1
            if tile == Tile.OPEN:
                self.set(row, col, Tile.SOLUTION)
        return length

    def print(self) -> None:
        """Print the maze to stdout."""
        chars = " █S•"
        # Row 0 is the bottom, so we must iterate in reverse order.
        for row in reversed(range(0, ROWS)):
            print("".join(chars[cell] for cell in self.row(row)))
        # The wall at the bottom is not stored in the maze data.
        print(chars[1] * (COLS - 1))


class MazeGenerator:
    random: Random
    maze: Maze
    level_mask: int
    player_row: int
    player_col: int
    restart_row: int
    restart_col: int
    max_col: int
    retries: int
    queue_counter: int
    queue: list[tuple[int, int]]

    @dataclass
    class Rerandomize:
        row: int
        col: int
        iters: int

    @dataclass
    class Stop:
        pass

    WalkResult = Rerandomize | Stop

    def __init__(self, random: Random) -> None:
        """Generate a maze using the given random number generator."""
        self.random = random
        self.initial_seed = self.random.seed
        self.maze = Maze()

        # Read the current seed and compute a mask for the level lookup index
        # (see walk()). This varies the difficulty of each maze a bit.
        self.level_mask = (self.random.seed >> 8) & 0xF

        # The player always starts on an even row and the second column.
        self.player_row = self.random_even_row()
        self.player_col = 1
        self.restart_row = self.player_row
        self.restart_col = self.player_col

        # Every entry in the queue MUST be initialized to (player_row, player_col),
        # otherwise some mazes will not generate correctly.
        self.queue = [(self.player_row, self.player_col)] * QUEUE_MAX
        self.queue_idx = 0
        self.queue_counter = CELLS_PER_QUEUE

        # Perform the initial walk from the starting position to create the solution.
        self.max_col = COLS
        self.retries = MAX_RETRIES
        self.walk(self.player_row, self.player_col, iters=1)

        # Do a second walk just to find more cells.
        self.max_col = COLS - 2
        self.walk(self.player_row, self.player_col, iters=80)

        # Walk from each cell in the queue.
        for row, col in self.queue:
            self.walk(row, col, 24)

        # Complete the maze by scanning for unclaimed cells.
        self.complete_maze()

        # Set the starting tile so it shows up with print().
        self.set(self.player_row, self.player_col, Tile.START)

    def walk(
        self,
        row: int,
        col: int,
        iters: int,
        direction: int | None = None,
        level: int | None = None,
    ) -> None:
        # sub_FC68 in the game
        # In the actual game code, the walk function uses tail recursion, but we can't
        # do that in Python so we need to call it iteratively.
        done = False
        while not done:
            if level is None:
                level = LEVEL_TABLE[(self.random.seed >> 8) & self.level_mask]
            if direction is None:
                direction = (self.random.next() & 0xC0) >> 6
            match self.__walk(row, col, iters, direction, level):
                case self.Rerandomize(row=new_row, col=new_col, iters=new_iters):
                    row, col, iters = new_row, new_col, new_iters
                    direction, level = None, None
                case self.Stop():
                    done = True
                case _ as unreachable:
                    assert_never(unreachable)

    def __walk(
        self,
        row: int,
        col: int,
        iters: int,
        direction: int,
        level: int,
    ) -> WalkResult:
        # sub_FC77 in the game
        # Higher level values make the maze have longer hallways because the direction
        # is rerandomized less often.
        for _ in range(0, level):
            # Seems like the game relies on this value underflowing back to 255?
            iters = (iters - 1) & 0xFF
            if iters == 0:
                # Stop if this is not the initial walk.
                if self.max_col < COLS:
                    return self.Stop()

                # Don't rerandomize until we've exhausted our retries.
                self.retries -= 1
                if self.retries == 0:
                    self.retries = 1
                    return self.Rerandomize(
                        self.restart_row, self.restart_col, iters=64
                    )

            # Randomly check all four possible directions to see if we can expand.
            turn = CCW if (self.random.seed & 0x80 != 0) else CW
            for _ in range(0, 4):
                # Move two steps in our direction.
                row, col = move(row, col, direction, 2)

                # Check adjacent cells to see if this cell is already part of the maze.
                test = self.test_adjacent_cells(row, col)
                if test == Tile.CLOSED:
                    # This cell is surrounded, so it has not been added to the maze yet
                    # and we can move into it.
                    break

                # We can't use this cell. Go back and turn.
                row, col = move(row, col, flip(direction), 2)
                direction = rotate(direction, turn)

            if test != Tile.CLOSED:
                # We checked all four directions and couldn't find any cells to add.
                # One adjacent cell must be empty, or else we couldn't have gotten here,
                # so search for it starting in a random direction.
                direction = self.random.next() >> 6
                turn = CCW if (self.random.seed & 0x2000 != 0) else CW
                for _ in range(0, 4):
                    # Move one step in our direction.
                    row, col = move(row, col, direction, 1)

                    # We found a way forward. Restart from here.
                    if self.get(row, col) == 0:
                        row, col = move(row, col, direction, 1)
                        return self.Rerandomize(row, col, iters)

                    # Step back and turn.
                    row, col = move(row, col, flip(direction), 1)
                    direction = rotate(direction, turn)

                assert False, f"No empty cell adjacent to ({row}, {col})"

            # Step back and delete the wall we crossed to get here.
            wall_row, wall_col = move(row, col, flip(direction), 1)
            self.set(wall_row, wall_col, Tile.OPEN)

            # Incentivize getting closer to the exit.
            if col >= self.restart_col:
                self.restart_col = col
                self.restart_row = row
                iters = max(iters, 64)

            # If we found the exit, stop.
            if col >= COLS - 1:
                return self.Stop()

            # Push every 10 spaces onto our queue. We will walk again from these later.
            if self.queue_idx < QUEUE_MAX:
                self.queue_counter -= 1
                if self.queue_counter == 0:
                    self.queue_counter = CELLS_PER_QUEUE
                    self.queue[self.queue_idx] = (row, col)
                    self.queue_idx += 1

        # Start over in a new direction.
        return self.Rerandomize(row, col, iters)

    def complete_maze(self) -> None:
        # Scan for surrounded cells starting in the upper-right corner and add each one
        # to the maze.
        row = ROWS - 2
        col = COLS - 3
        while row >= 0:
            test = self.test_adjacent_cells(row, col)
            if test == Tile.OPEN:
                # This cell is not surrounded, so keep looking.
                col -= 2
                if col < 0:
                    # Wrap around and move two rows down.
                    col = COLS - 3
                    row -= 2
            else:
                # This cell is surrounded, so we need to add it to the maze.
                # Choose a random direction to start looking for the rest of the maze.
                rand = self.random.next()
                direction = rand >> 6
                turn = CW if (rand & 1 != 0) else CCW

                # Keep turning until we find the direction the maze is in.
                test = Tile.CLOSED
                while test != Tile.OPEN:
                    direction = rotate(direction, turn)
                    test_row, test_col = row, col

                    # Scan forwards until we find the maze or go out-of-bounds.
                    test = Tile.CLOSED
                    while test == Tile.CLOSED:
                        test_row, test_col = move(test_row, test_col, direction, 2)
                        test = self.test_adjacent_cells(test_row, test_col)

                # We found the maze. Starting inside it, walk backwards towards our
                # initial cell.
                self.walk(test_row, test_col, 5, direction=flip(direction), level=1)

        # Finally, clear out the exit column.
        for row in range(1, ROWS, 2):
            self.set(row, COLS - 1, Tile.OPEN)

    def test_adjacent_cells(self, row: int, col: int) -> int:
        """Return a bitwise AND of the tiles surrounding (row, col)."""
        # sub_FA44 in the game
        # As an optimization, we return early if any of the adjacent cells is 0.
        up = self.get(row + 1, col)
        if up == 0:
            return 0
        down = self.get(row - 1, col)
        if down == 0:
            return 0
        right = self.get(row, col + 1)
        if right == 0:
            return 0
        left = self.get(row, col - 1)
        return up & down & right & left

    def get(self, row: int, col: int) -> Tile:
        """
        Get the tile at (row, col) if the position is in bounds,
        otherwise return INVALID.
        """
        # sub_F9E3 in the game
        if row >= 0 and row < ROWS - 1 and col > 0 and col < self.max_col:
            return self.maze.get(row, col)
        else:
            return Tile.INVALID

    def set(self, row: int, col: int, tile: Tile) -> None:
        """Set the tile at (row, col)."""
        self.maze.set(row, col, tile)

    def random_even_row(self) -> int:
        """Return a random even row number."""
        # sub_FA67 in the game
        row = ROWS
        while row >= ROWS / 2:
            row = self.random.next() & 0xF
        return row * 2


@dataclass
class MazeResult:
    seed: int
    length: int
    maze: Maze


def generate_mazes_worker(start: int, end: int, solve: bool) -> list[MazeResult]:
    try:
        results = []
        for seed in range(start, end):
            random = Random(seed)
            gen = MazeGenerator(random)
            if solve:
                length = gen.maze.solve(gen.player_row, gen.player_col) // 2
            else:
                length = 0
            results.append(MazeResult(seed, length, gen.maze))
        return results
    except KeyboardInterrupt:
        return []


def generate_all_mazes(jobs: int | None = None, solve: bool = True) -> list[MazeResult]:
    num_threads = jobs or os.cpu_count() or 1
    chunksize = 65536 // num_threads
    print(f"Starting {num_threads} processes with a chunk size of {chunksize}")
    chunks = []
    for i in range(0, 65536, chunksize):
        chunks.append((i, min(65536, i + chunksize), solve))

    mazes: list[MazeResult] = []
    start_time = time.monotonic()
    with multiprocessing.Pool(processes=num_threads) as pool:
        try:
            for results in pool.starmap(generate_mazes_worker, chunks):
                mazes.extend(results)
        except KeyboardInterrupt:
            print("Interrupted")
            pool.terminate()
            return []
    end_time = time.monotonic()
    print(f"Computed {len(results)} mazes in {end_time - start_time}s")
    return mazes


def any_base(s: str) -> int:
    return int(s, base=0)


def command_generate(args: argparse.Namespace) -> int:
    seed = args.seed if args.seed is not None else random.getrandbits(16)
    rng = Random(seed)
    print(f"seed: 0x{rng.seed:0>4x}")
    gen = MazeGenerator(rng)
    length = gen.maze.solve(gen.player_row, gen.player_col) // 2
    print(f"next seed: 0x{rng.seed:0>4x}")
    print(f"solution length: {length}")
    print()
    gen.maze.print()
    return 0


def command_unpack(args: argparse.Namespace) -> int:
    with open(args.path, "rb") as f:
        f.seek(args.offset)
        data = f.read(PACKED_MAZE_SIZE)
    maze = Maze.unpack(data)
    assert maze.pack() == data
    maze.print()
    return 0


def command_search(args: argparse.Namespace) -> int:
    # This is very suboptimal, but it works
    mazes = generate_all_mazes(args.jobs)
    if not mazes:
        return 1
    mazes.sort(key=lambda r: r.length)
    for starting in STARTING_SEEDS:
        print(f"seed 0x{starting:0>4x}:")
        seeds = []
        for maze in mazes:
            if maze.length > 23:
                break
            steps = 0
            random = Random(starting)
            MazeGenerator(random)
            while random.seed != maze.seed:
                random.next()
                steps += 1
            seeds.append((maze.seed, maze.length, steps))
        seeds.sort(key=lambda s: s[2])
        for seed, length, steps in seeds[:5]:
            print(f"    seed: 0x{seed:0>4x}, length: {length}, steps: {steps}")
    return 0


def command_check(args: argparse.Namespace) -> int:
    mazes = generate_all_mazes(args.jobs, solve=False)
    if not mazes:
        return 1
    errors = 0
    mazes.sort(key=lambda m: m.seed)
    with open(args.path, "rb") as file:
        for seed, maze in enumerate(mazes):
            if seed & 0xFFF == 0:
                end_seed = seed | 0xFFF
                print(f"Check: {seed:0>4x}-{end_seed:0>4x}")
            data = file.read(PACKED_MAZE_SIZE)
            packed = maze.maze.pack()
            if packed != data:
                print(f"MISMATCH: seed 0x{seed:0>4x}")
                errors += 1
    if errors == 0:
        print("Success!")
        return 0
    else:
        print(f"Failed: {errors} errors")
        return 1


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="mazecraze.py",
        description="Maze Craze maze generation utility",
    )
    subparsers = parser.add_subparsers(help="subcommand")

    generate_parser = subparsers.add_parser(
        "generate", help="generate and solve a maze"
    )
    generate_parser.add_argument(
        "seed", type=any_base, nargs="?", help="maze seed (random if not specified)"
    )
    generate_parser.set_defaults(func=command_generate)

    unpack_parser = subparsers.add_parser(
        "unpack", help="unpack a maze in a memory dump"
    )
    unpack_parser.add_argument(
        "-o", "--offset", type=any_base, default=0, help="file offset"
    )
    unpack_parser.add_argument("path", type=str, help="path to the file to open")
    unpack_parser.set_defaults(func=command_unpack)

    search_parser = subparsers.add_parser(
        "search",
        help="find the shortest mazes accessible from each starting seed (slow!)",
    )
    search_parser.add_argument(
        "-j", "--jobs", type=int, default=0, help="number of processes to create"
    )
    search_parser.set_defaults(func=command_search)

    check_parser = subparsers.add_parser(
        "check",
        help="generate all possible mazes and compare them against a binary file",
    )
    check_parser.add_argument(
        "-j", "--jobs", type=int, default=0, help="number of processes to create"
    )
    check_parser.add_argument("path", type=str, help="path to the file to read")
    check_parser.set_defaults(func=command_check)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
