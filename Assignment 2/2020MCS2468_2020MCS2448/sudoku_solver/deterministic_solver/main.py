# -*- coding: utf-8 -*-

"""
Engineer: Andrey Puzanov
This is the main file for the program.
"""

from sudoku_solver.deterministic_solver.game import Sudoku

"""========================================================================="""
""" ENTER SUDOKU TO SOLVE HERE """
"""========================================================================="""

# Easy
# sudoku = [[0,6,4,1,0,3,0,2],
#          [8,0,0,0,1,0,5,0],
#          [0,0,7,0,0,8,3,5],
#          [6,0,3,0,4,0,0,0],
#          [0,5,0,2,0,4,8,0],
#          [0,1,0,7,0,0,6,3],
#          [2,0,0,8,0,7,0,0],
#          [0,4,5,0,0,2,1,0]]

# Not solvable
# sudoku = [[4,0,0,2,1,0,3,0],
#          [0,7,0,0,0,2,0,0],
#          [0,0,2,0,5,0,7,0],
#          [7,0,0,5,0,0,0,3],
#          [6,0,0,0,3,0,0,1],
#          [0,1,0,3,0,8,0,0],
#          [0,0,7,0,0,0,8,0],
#          [0,6,0,8,4,0,2,5]]

# Hard
sudoku = [[0, 0, 0, 0, 0, 3, 0, 4],
          [7, 0, 0, 0, 8, 0, 0, 0],
          [0, 0, 8, 0, 0, 5, 0, 3],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 5],
          [0, 0, 7, 5, 0, 4, 0, 0],
          [0, 0, 1, 0, 0, 7, 3, 0],
          [3, 6, 0, 0, 0, 0, 0, 0]]

sudoku = [[7, 2, 6, 3, 0, 1, 8, 4], [8, 0, 4, 1, 2, 6, 7, 3], [3, 4, 1, 2, 7, 0, 6, 8], [0, 8, 7, 6, 4, 2, 3, 1], [1, 3, 0, 4, 8, 7, 2, 6], [2, 6, 8, 7, 1, 3, 4, 0], [6, 1, 2, 8, 3, 4, 0, 7], [4, 7, 0, 0, 6, 8, 1, 2]]

sudoku = [[8, 0, 0, 2, 0, 5, 4, 0], [5, 6, 4, 0, 8, 0, 0, 2], [2, 0, 8, 5, 6, 4, 0, 0], [4, 0, 6, 0, 2, 0, 5, 0], [0, 8, 0, 6, 4, 0, 2, 5], [0, 2, 5, 0, 0, 6, 0, 8], [0, 5, 2, 8, 0, 0, 6, 4], [6, 4, 0, 0, 5, 2, 8, 0]]
"""========================================================================="""
""" SOLVING SUDOKU """
"""========================================================================="""

if not Sudoku.is_empty(sudoku):

    # Create the initial candidates array for the given Sudoku
    candidates_array = Sudoku.create_candidates_array(sudoku)

    # Solve Sudoku logically (without guesses)
    solved, empty_vals, methods, m_counts = Sudoku.solve_sudoku(candidates_array)

    # Interpret and display the results, rate difficulty
    if empty_vals:
        print('Sudoku is not solvable')
    elif solved:
        print('Sudoku has a unique solution')
        print(candidates_array)
        if 'pointing_pairs' in methods:
            print('Difficulty: Medium')
        elif 'hidden_singles' in methods:
            print('Difficulty: Easy')
        else:
            print('Difficulty: Very Easy')
    else:  # Sudoku cannot be solved logically, have to make guesses
        print('Sudoku might have more than one solution')

        # Solving Sudoku using DFS method
        solved, dfs_depth, solution = Sudoku.solve_sudoku_dfs(candidates_array)

        if not solved:
            print('Sudoku is not solvable')
        else:
            print(solution)
            if dfs_depth < 5:
                print('Difficulty: Hard')
            else:
                print('Difficulty: Very Hard')

else:
    print('You entered an empty Sudoku')

