import torch
from typing import *

def independent_zeros(mat: torch.Tensor) -> Optional[torch.Tensor]:
    r"""
    PROCEDURE TO GET ALL INDEPENDENT / ASSIGNED ZEROS
    1. Select row with the least number of zeros that does not have an assigned zero value
    2. Select column of that selected row with the least number of zeros that does not have
        an assigned zero value
    3. If no selected column, record the corresponding row as a row with an assigned value and skip
    4. Store the row and column
    5. record the row and column as ones with assigned zeros so they do not get retrieved again
    6. Remove the value at index (row, col) in the operated matrix
    7. Repeat steps 1 through 6 until either or rows or columns have been tagged as having an assigned zero

    params (torch.Tensor)
    ----------------------
    Input square matrix representing the row and column reduced cost

    return
    ----------------------
    returns independent / assigned zeros (torch.Tensor)
    """
    m = torch.where(mat==0, 1.0, 0.0)
    selected_rows = torch.zeros(m.shape[0], dtype=torch.bool, device=m.device)
    selected_cols = torch.zeros(m.shape[1], dtype=torch.bool, device=m.device)
    result = []

    while True:
        row_sum = m.sum(dim=1)
        col_sum = m.sum(dim=0)
        if torch.all(selected_rows) or torch.all(selected_cols):
            break

        r = row_sum.argsort()
        r = r[~selected_rows[r]][0]
        c = torch.where(m[r, :] == 1)[0]
        c = c[~selected_cols[c]]
        if c.shape[0] == 0:
            m[r] = 0
            selected_rows[r] = True
            continue
        c = c[col_sum[c].argmin()]
        result.append(torch.tensor([r, c], device=m.device))
        selected_rows[r] = True
        selected_cols[c] = True
        m[r] = 0
        m[:, c] =0

    return torch.stack(result, dim=0)

def get_min_lines_mask(mat: torch.Tensor, assigned_indexes: torch.Tensor) -> torch.Tensor:
    r"""
    PROCEDURE FOR GETTING MINIMUM NUMBER OF HORIZONTAL AND VERTICAL LINES
    1. Mark all rows without assigned zeros
    2. For each marked rows, mark all columns containing a zero in that row
    3. For each marked column, mark the rows that have assigned zeros
    4. Repeat steps 2 to 3 until no new columns can be marked (in our case most ops were vectorized)
    5. Draw lines through all unmarked rows and marked columns.

    params
    ----------------------
    mat (torch.Tensor)
        input cost matrix with some zeros
    assigned_indexes (torch.Tensor)
        (row, col) indexes of assigned zeros

    return 
    ----------------------
    boolean matrix with vertical and horizontal lines, Lines are represented as True values running across 
    rows and columns (torch.Tensor)
    """
    assert mat.ndim == 2 and mat.shape[0] == mat.shape[1]
    assigned_rows, assigned_col = assigned_indexes.T
    rows, _ = torch.arange(
        0, mat.shape[0], device=mat.device, dtype=torch.int64
    ).unsqueeze(dim=0).tile((2, 1))

    marked_rows = rows[~torch.isin(rows, assigned_rows)]
    while True:
        _, marked_cols = torch.where(mat[marked_rows] == 0)
        marked_cols = marked_cols.unique()
        new_marked_row = assigned_rows[torch.isin(assigned_col, marked_cols)]
        if torch.isin(new_marked_row, marked_rows).all():
            break
        marked_rows = torch.cat([marked_rows, new_marked_row], dim=0).unique()

    unmarked_rows = rows[~torch.isin(rows, marked_rows)]
    line_mask = torch.zeros_like(mat, dtype=torch.bool)
    # we denote lines as True values running either vertically or horizontally 
    # across the matrix
    line_mask[unmarked_rows] = True
    line_mask[:, marked_cols] = True
    return line_mask
    
def hungarian_solver(cost_matrix: torch.Tensor) -> Tuple[torch.Tensor, float]:
    r"""
    PROCEDURE FOR SOLVING AN ASSIGNMENT PROBLEM WITH THE HUNGARIAN SOLVER
    1. Given the cost matrix, subtract the minimum value of each row from all rows
    2. Subtract the minimum value of each column from all column
    3. Retrieve all independent zeros (zeros for assignments)
    4. if number of assigned zeros is equal to size of the row / col (square matrix), 
        then return the assgined zero indexes and the corresponding cost
    5. If the number of assigned zeros is less than row / col size, get the minimum number
        of vertical and horizontal lines needed to cover all zeros in the matrix
    6. Get the minimum value that is neither covered by a horizontal nor vertical line
    7. Subtract the value from all uncovered values in the matrix
    8. Add the value to all values at line intersects (where vertical and horizontal lines cross)
    9. If no lines intersect, skip
    9. Repeat steps 3 through 8 until convergence (step 4 is triggered on convergence)

    params
    ----------------------
    cost_matrix (torch.Tensor)
        the matrix that represents the assignment problem

    return 
    ----------------------
    returns a tuple with the optimal assignment indexes (torch.Tensor) and its corresponding cost (float)
    """
    assert cost_matrix.ndim == 2
    cmat = cost_matrix.clone()
    cmat -= cmat.min(1).values.unsqueeze(dim=1)
    cmat -= cmat.min(1).values.unsqueeze(dim=0)
    
    while True:
        assign_indexes = independent_zeros(cmat)
        if assign_indexes.shape[0] == cmat.shape[0]:
            return assign_indexes, cost_matrix[assign_indexes[:, 0], assign_indexes[:, 1]].sum().item()
        lines = get_min_lines_mask(cmat, assign_indexes)
        uncrossed_min = cmat[~lines].min()
        cmat[~lines] -= uncrossed_min
        crossed_rows = torch.where(lines.sum(dim=1) == lines.shape[0])[0]
        crossed_cols = torch.where(lines.sum(dim=0) == lines.shape[0])[0]
        if crossed_rows.shape[0] == 0 or crossed_cols.shape[0] == 0:
            continue
        intersects = torch.cat([
            torch.stack([
                torch.stack([i, j]) for i in crossed_rows
            ]) for j in crossed_cols
        ])
        cmat[intersects[:, 0], intersects[:, 1]] += uncrossed_min