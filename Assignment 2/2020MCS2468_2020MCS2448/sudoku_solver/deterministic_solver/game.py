
import numpy as np
import re

# Define the 8 boxes an 8x8 sudoku consists of (brick-wall pattern):
# the box name and coordinates of its top-left cell
boxes = {1: (0, 0), 2: (0, 4),
         3: (2, 0), 4: (2, 4),
         5: (4, 0), 6: (4, 4),
         7: (6, 0), 8: (6, 4)}


class Sudoku(object):

    def __init__(self):
        pass

    # The function returns top-left coordinate of the box a cell belongs to
    @staticmethod
    def which_box(x, y):
        if y < 4:
            if x < 2:
                return 1
            elif x < 4:
                return 3
            elif x < 6:
                return 5
            else:
                return 7
        else:
            if x < 2:
                return 2
            elif x < 4:
                return 4
            elif x < 6:
                return 6
            else:
                return 8

    # The function inserts a new known value in candidates array and deletes
    # that value from other cells in the row, column, and the box it belongs to
    @staticmethod
    def insert_value(x, y, val, cand_arr):
        made_progress = False
        for i in range(0, cand_arr.shape[1]):
            if not i == y:
                if str(val) in cand_arr[x, i]:
                    cand_arr[x, i] = re.sub(str(val), '', cand_arr[x, i])
                    made_progress = True
        for i in range(0, cand_arr.shape[0]):
            if not i == x:
                if str(val) in cand_arr[i, y]:
                    cand_arr[i, y] = re.sub(str(val), '', cand_arr[i, y])
                    made_progress = True
        x_start, y_start = boxes[Sudoku.which_box(x, y)]
        for i in range(x_start, x_start + 2):
            for j in range(y_start, y_start + 4):
                if not (i == x and j == y):
                    if str(val) in cand_arr[i, j]:
                        cand_arr[i, j] = re.sub(str(val), '', cand_arr[i, j])
                        made_progress = True
        cand_arr[x, y] = val
        return made_progress, Sudoku.has_empty_values(cand_arr)

    # The function updates the solution when a cell in candidates array is left
    # with only one possible value and reports if progress was made
    @staticmethod
    def find_naked_singles(cand_arr):
        for i in range(0, cand_arr.shape[0]):
            for j in range(0, cand_arr.shape[1]):
                if len(cand_arr[i, j]) == 1:
                    mp, tmp = Sudoku.insert_value(i, j, cand_arr[i, j], cand_arr)
        return mp, Sudoku.has_empty_values(cand_arr)

    # The function looks for naked pairs (also triples, quads, etc), updates
    # the candidates array and reports if progress was made
    @staticmethod
    def find_naked_pairs(cand_arr):
        made_progress = False
        for key in boxes:  # the last one
            for pp_size in range(2, 8):
                x_start, y_start = boxes[8]
                for i in range(x_start, x_start + 2):
                    for j in range(y_start, y_start + 4):
                        coord_arr = []
                        if len(cand_arr[i, j]) == pp_size:
                            val = cand_arr[i, j]
                            for ii in range(x_start, x_start + 2):
                                for jj in range(y_start, y_start + 4):
                                    if cand_arr[ii, jj] == val:
                                        coord_arr.append((ii, jj))
                            if len(coord_arr) == pp_size:
                                for c in val:
                                    for iii in range(x_start, x_start + 2):
                                        for jjj in range(y_start, y_start + 4):
                                            if c in cand_arr[iii, jjj] and not (iii, jjj) in coord_arr:
                                                made_progress = True
                                                cand_arr[iii, jjj] = re.sub(c, '', cand_arr[iii, jjj])
        return made_progress, Sudoku.has_empty_values(cand_arr)

    # The function finds hidden singles, updates the candidates array and returns
    # true if hidden singles were in fact found
    @staticmethod
    def find_hidden_singles(cand_arr):
        made_progress = False
        for key in boxes:
            x_start, y_start = boxes[key]
            for i in range(x_start, x_start + 2):
                for j in range(y_start, y_start + 4):
                    if len(cand_arr[i, j]) > 1:
                        for c in cand_arr[i, j]:
                            found = False
                            for ii in range(x_start, x_start + 2):
                                for jj in range(y_start, y_start + 4):
                                    if not (ii == i and jj == j):
                                        if c in cand_arr[ii, jj]:
                                            found = True
                            if found == False:
                                made_progress = True
                                Sudoku.insert_value(i, j, c, cand_arr)
        return made_progress, Sudoku.has_empty_values(cand_arr)

    # The function returns True if the Sudoku is solved
    @staticmethod
    def is_solved(cand_arr):
        result = True
        for i in range(0, cand_arr.shape[0]):
            for j in range(0, cand_arr.shape[1]):
                if not len(cand_arr[i, j]) == 1:
                    result = False
        return result

    # The function checks if the candidates arrays has empty values
    @staticmethod
    def has_empty_values(cand_arr):
        result = False
        for x in np.nditer(cand_arr):
            if x == '':
                result = True
        return result

    # This function checks if user is trying to solve an empty sudoku
    @staticmethod
    def is_empty(sudoku):
        result = True
        for index, val in np.ndenumerate(sudoku):
            if val != 0:
                result = False
        return result

    # The function finds a cell with the least number of candidates
    @staticmethod
    def fewest_candidates_cell(cand_arr):
        smallest_length = 8
        for index, val in np.ndenumerate(cand_arr):
            if len(val) > 1:
                if len(val) < smallest_length:
                    smallest_length = len(val)
                    x, y = index
        return x, y

    # This function finds pointing pairs, updates the candidates array and
    # reports if progress was made
    @staticmethod
    def find_pointing_pairs(cand_arr):
        made_progress = False
        ## reducing row candidates
        for key in boxes:
            x_start, y_start = boxes[key]
            for j in range(y_start, y_start + 4):
                if len(cand_arr[x_start, j]) > 1:
                    for c in cand_arr[x_start, j]:
                        found_pp = True
                        if c in cand_arr[x_start + 1, j] and len(cand_arr[x_start + 1, j]) > 1:
                            for ii in range(x_start, x_start + 2):
                                for jj in range(y_start, y_start + 4):
                                    if not jj == j:
                                        if c in cand_arr[ii, jj]:
                                            found_pp = False
                            if found_pp == True:
                                for k in range(0, cand_arr.shape[1]):
                                    if not (k == x_start or k == x_start + 1):
                                        if c in cand_arr[k, j]:
                                            made_progress = True
                                            cand_arr[k, j] = re.sub(c, '', cand_arr[k, j])
        ## reducing column candidates
        for key in boxes:
            x_start, y_start = boxes[key]
            for i in range(x_start, x_start + 2):
                for j in range(y_start, y_start + 4):
                    if len(cand_arr[i, j]) > 1:
                        for c in cand_arr[i, j]:
                            found_pp = True
                            for ii in range(x_start, x_start + 2):
                                for jj in range(y_start, y_start + 4):
                                    if ii != i:
                                        if c in cand_arr[ii, jj]:
                                            found_pp = False
                            if found_pp == True:
                                for k in range(0, cand_arr.shape[0]):
                                    if not (k == y_start or k == y_start + 1 or k == y_start + 2 or k == y_start + 3):
                                        if c in cand_arr[i, k]:
                                            made_progress = True
                                            cand_arr[i, k] = re.sub(c, '', cand_arr[i, k])
        ## reducing box candidates based on pairs in columns
        for key in boxes:
            x_start, y_start = boxes[key]
            for i in range(x_start, x_start + 2):
                for j in range(y_start, y_start + 4):
                    if len(cand_arr[i, j]) > 1:
                        for c in cand_arr[i, j]:
                            found_pp_col = True
                            for ii in range(0, cand_arr.shape[0]):
                                if not (ii == x_start or ii == x_start + 1):
                                    if c in cand_arr[ii, j]:
                                        found_pp_col = False
                            if found_pp_col == True:
                                for ii in range(x_start, x_start + 2):
                                    for jj in range(y_start, y_start + 4):
                                        if not jj == j:
                                            if c in cand_arr[ii, jj]:
                                                made_progress = True
                                                cand_arr[ii, jj] = re.sub(c, '', cand_arr[ii, jj])
        ## reducing box candidates based on pairs in rows
        for key in boxes:
            x_start, y_start = boxes[key]
            for i in range(x_start, x_start + 2):
                for j in range(y_start, y_start + 4):
                    if len(cand_arr[i, j]) > 1:
                        for c in cand_arr[i, j]:
                            found_pp_row = True
                            for jj in range(0, cand_arr.shape[1]):
                                if not (jj == y_start or jj == y_start + 1 or jj == y_start + 2 or jj == y_start + 3):
                                    if c in cand_arr[i, jj]:
                                        found_pp_row = False
                            if found_pp_row == True:
                                for ii in range(x_start, x_start + 2):
                                    for jj in range(y_start, y_start + 4):
                                        if not ii == i:
                                            if c in cand_arr[ii, jj]:
                                                made_progress = True
                                                cand_arr[ii, jj] = re.sub(c, '', cand_arr[ii, jj])

        return made_progress, Sudoku.has_empty_values(cand_arr)

    # The function creates a candidates array for the given sudoku and
    # performs the initial placement
    @staticmethod
    def create_candidates_array(sudoku):
        sudoku = np.array(sudoku)
        candidates_array = np.full((8, 8), '12345678')

        for index, val in np.ndenumerate(sudoku):
            if val != 0:
                Sudoku.insert_value(index[0], index[1], val, candidates_array)
        return candidates_array

    # The function solves Sudoku. Returns the list of methods(functions)
    # that were used to solve it (with counts)
    # or reports that the sudoku is not solvable
    @staticmethod
    def solve_sudoku(cand_arr):
        methods = []
        m_counts = []
        count_ns = 0
        count_hs = 0
        count_pp = 0
        while True:
            progress = False
            count_ns = count_ns + 1
            pr_ns, e = Sudoku.find_naked_singles(cand_arr)
            pr_np, e = Sudoku.find_naked_pairs(cand_arr)
            progress = progress or pr_ns
            if 'naked_values' not in methods:
                methods.append('naked_values')
            if (not Sudoku.is_solved(cand_arr) and pr_ns == False and pr_np == False):
                count_hs = count_hs + 1
                pr_hs, e = Sudoku.find_hidden_singles(cand_arr)
                progress = progress or pr_hs
                if 'hidden_singles' not in methods:
                    methods.append('hidden_singles')
                if (not Sudoku.is_solved(cand_arr) and pr_hs == False):
                    count_pp = count_pp + 1
                    pr_pp, e = Sudoku.find_pointing_pairs(cand_arr)
                    progress = progress or pr_pp
                    if 'pointing_pairs' not in methods:
                        methods.append('pointing_pairs')
            if (Sudoku.is_solved(cand_arr) or Sudoku.has_empty_values(cand_arr) or not progress):
                break
        m_counts.append(count_ns)
        m_counts.append(count_hs)
        m_counts.append(count_pp)
        return Sudoku.is_solved(cand_arr), Sudoku.has_empty_values(cand_arr), methods, m_counts

    # This function solves Sudoku using a DFS based guessing method
    @staticmethod
    def solve_sudoku_dfs(cand_arr):
        solved = False
        x, y = Sudoku.fewest_candidates_cell(cand_arr)
        max_depth = 0
        for c in range(0, len(cand_arr[x, y])):
            if not solved:
                nodelist = []
                nodelist = [Sudoku_node(x, y, cand_arr[x, y][1], cand_arr)]
                solved, empty_vals, methods, m_counts = Sudoku.solve_sudoku(nodelist[0].cand_arr)
                solution = nodelist[0].cand_arr
                if not solved:
                    depth = 0
                    val_num = 0
                    while True:
                        if max_depth < depth:
                            max_depth = depth
                        x, y = Sudoku.fewest_candidates_cell(nodelist[depth].cand_arr)
                        nodelist.append(
                            Sudoku_node(x, y, nodelist[depth].cand_arr[x, y][val_num], nodelist[depth].cand_arr))
                        depth = depth + 1
                        solved, empty_vals, methods, m_counts = Sudoku.solve_sudoku(nodelist[depth].cand_arr)
                        if solved:
                            solution = nodelist[depth].cand_arr
                            break
                        if empty_vals:
                            if val_num < len(nodelist[depth].cand_arr[x, y]):
                                val_num = val_num + 1
                            else:
                                val_num = val_num + 1
                                depth = depth - 1
                                if depth == 0 and (val_num - 1) == len(nodelist[depth].cand_arr[x, y]):
                                    break

        return solved, max_depth, solution


# The class defines Sudoku_node used in DFS solving method
class Sudoku_node:

    def __init__(self, x, y, val, cand_arr):
        self.x_coordinate = x
        self.y_coordinate = y
        self.value = val
        self.cand_arr = Sudoku_node.new_cand_arr(x, y, val, cand_arr)

    def new_cand_arr(x, y, val, cand_arr):
        c_a = np.copy(cand_arr)
        Sudoku.insert_value(x, y, val, c_a)
        return c_a

