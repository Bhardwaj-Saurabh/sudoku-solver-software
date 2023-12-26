# Sudoku Solver

## Goal of the project

This project creates a Sudoku-solving agent in order to solve diagonal Sudoku puzzles by **Constraint Propagation** and **Search**. A diagonal Sudoku puzzle is identical to traditional Sudoku puzzles with the added constraint that the boxes on the two main diagonals of the board must also contain the digits 1-9 in each cell (just like the rows, columns, and 3x3 blocks). 

## Strategies

**Strategy 1: Elimination**<br />
If a box has a value assigned, then none of the peers of this box can have this value.<br />
**Strategy 2: Only Choice**<br />
If there is only one box in a unit which would allow a certain digit, then that box must be assigned that digit.<br />
**Strategy 3: Naked Twins**<br />
If you have two or more unallocated boxes in a unit and there are only two digits that can go in those two boxes, then those two digits can be eliminated from the possible assignments of all other boxes in the same unit.<br />
**Strategy 4: Search**<br />
Pick a box with a minimal number of possible values. Try to solve each of the puzzles obtained by choosing each of these values, recursively.<br />

## Code

* `solution.py` - My solution for the Diagonal Sudoku Solver.
* `tests/test_solution.py` - Test my solution.
* `PySudoku.py` - Code for Visualization.
* `utils.py` - Some basic functions for the agent.

## Visualization

**Note:** The `pygame` library is required to visualize my solution -- however, the `pygame` module can be troublesome to install and configure. It should be installed by default with the AIND conda environment, but it is not reliable across all operating systems or versions. Please refer to the pygame documentation [here](http://www.pygame.org/download.shtml).

Running `python solution.py` will automatically attempt to visualize my solution, and I use the `assign_value` function (defined in `utils.py`) to track the puzzle solution progress for reconstruction during visuzalization.
