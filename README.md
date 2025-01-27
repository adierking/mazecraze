# Maze Craze Algorithm

The algorithm from the Atari 2600 game [Maze Craze](https://en.wikipedia.org/wiki/Maze_Craze),
reverse-engineered and rewritten in Python. Maze Craze was developed by Rick Maurer.

Walgrey used this to make a TAS of the game in 00:05.39:

[![Watch the video](https://img.youtube.com/vi/BYTQkU5iqGg/hqdefault.jpg)](https://youtu.be/BYTQkU5iqGg)

## Files

* **mazecraze.py** - The main python script. Contains the algorithm and some utility subcommands.
* **dumpmazes.lua** - A BizHawk script for dumping all possible mazes from the game.
* **mazes.bin** - Precomputed output from dumpmazes.lua so you don't need to run it yourself.
  Mainly useful for verifying the correctness of the code using the `check` command.

## Usage

To generate and solve the maze corresponding to a seed, use `generate`:

    $ python mazecraze.py generate 0x0400
    seed: 0x0400
    next seed: 0x56df
    solution length: 92

    ███████████████████████████████████████
    █   █•••••••█                       █ █
    █ ███•█████•█ ███████████ █ █ █████ █ █
    █S••••█   █•█ █•••••█     █ █ █   █ █ █
    █ █████ █ █•█ █•███•█ █████ █ █ █ █ █ █
    █ █•••••█ █•••█•█•••█ █     █ █ █ █   █
    █ █•███•█████•█•█•█████████████ █ ███ █
    █ █•█ █•••█  •█•█•••█       █   █   █ █
    ███•█ ███•███•█•███•█ █████ ███████ █ █
    █•••█•••█•••••█•█ █•█     █ █     █ █ █
    █•███•█•███████•█ █•█████ █ █ █████ █ █
    █•█•••█•••••••••█•••█   █ █       █ █ █
    █•█•█ ███████████•█████ █ ███████ █ █ █
    █•••█ █•••••    █•    █         █ █ █ █
    █████ █•███•█ ███•███ ███ ███████ █ █ █
    █     █•█ █•█ █•••█ █     █   █   █   █
    █ █████•█ █•███•███ █████████ █████████
    █ █   █•█ █•••••█       █     █••••••••
    █ █ ███•█ ███████ █████ █ ███ █•███████
    █ █•••••█      •••••••█ █ █ █ █•█ █   █
    █ █•█████████ █•█████•█ █ █ █ █•█ █ █ █
    █ █•••••••••█ █•█•••••█   █   █•  █ █ █
    █ █████████•███•█•█████████████•█ █ █ █
    █         █•••••█•••••••••••••••█   █ █
    ███████████████████████████████████████

To get a list of the closest seeds to each starting seed, use `search`.

In the output, "seed" is the seed of a maze, "length" is the length of the maze's solution, and
"steps" is the number of frames you have to wait for the RNG to get to the seed.

    $ python mazecraze.py search
    Starting 32 processes with a chunk size of 2048
    Computed 2048 mazes in 13.344404000000395s
    seed 0xff00:
        seed: 0xc3e3, length: 23, steps: 57
        seed: 0xa247, length: 20, steps: 837
        seed: 0x4f24, length: 23, steps: 1118
        seed: 0x2c09, length: 23, steps: 1747
        seed: 0xdf40, length: 23, steps: 2626
    ...

To unpack a maze from a memory dump, use `unpack`:

    python mazecraze.py unpack mazes.bin

To check the code against a dumped set of mazes, use `check`:

    python mazecraze.py check mazes.bin

## License

This is all licensed under the WTFPL, because the algorithm is not mine. Do whatever you want to.
