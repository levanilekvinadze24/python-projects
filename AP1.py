import os
import numpy as np
import matplotlib.pyplot as plt


class NormVisualizer:
    def __init__(self, seed: int = 42, plot_dir: str = "plots") -> None:
        # Random generator with fixed seed for reproducible results
        self.rng = np.random.default_rng(seed)

        # Folder where all plots will be saved
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)

        # 2) Data: generate two random 4D vectors
        self.x1 = self.rng.standard_normal(4)
        self.x2 = self.rng.standard_normal(4)

        # Reshape each 4D vector into a 2x2 matrix
        self.M1 = self.x1.reshape((2, 2))
        self.M2 = self.x2.reshape((2, 2))

        # Print data for inspection in the console
        print("x1 =", self.x1)
        print("x2 =", self.x2)
        print("M1 =\n", self.M1)
        print("M2 =\n", self.M2)

        # Differences (these will be used to compute distances)
        self.dx = self.x1 - self.x2   # vector difference
        self.dM = self.M1 - self.M2   # matrix difference

    # -------------------------------
    # 1) Norms
    # -------------------------------
    @staticmethod
    def vec_l1(v: np.ndarray) -> float:
        # L1 norm of a vector: sum of absolute values
        return np.linalg.norm(v, ord=1)

    @staticmethod
    def vec_l2(v: np.ndarray) -> float:
        # L2 norm of a vector: standard Euclidean norm
        return np.linalg.norm(v, ord=2)

    @staticmethod
    def mat_norm_1(M: np.ndarray) -> float:
        # Induced matrix 1-norm: maximum column sum of absolute values
        return np.linalg.norm(M, ord=1)

    @staticmethod
    def mat_norm_2(M: np.ndarray) -> float:
        # Induced matrix 2-norm (spectral norm): largest singular value
        return np.linalg.norm(M, ord=2)

    # -------------------------------
    # 3) Distances between x1, x2 and M1, M2
    # -------------------------------
    def print_distances(self) -> None:
        # Vector distances using L1 and L2 norms
        dist_vec_L1 = self.vec_l1(self.dx)
        dist_vec_L2 = self.vec_l2(self.dx)

        # Matrix distances using 1-norm and spectral 2-norm
        dist_mat_1 = self.mat_norm_1(self.dM)
        dist_mat_2 = self.mat_norm_2(self.dM)

        print("\n--- Distances ---")
        print("Pair A — Vector ‖x1 - x2‖_1:         ", dist_vec_L1)
        print("Pair A — Matrix ‖M1 - M2‖_1:         ", dist_mat_1)
        print("Pair B — Vector ‖x1 - x2‖_2:         ", dist_vec_L2)
        print("Pair B — Matrix ‖M1 - M2‖_2 (spec):  ", dist_mat_2)

    # -------------------------------
    # 4) Vector unit-ball slices in R^4
    # -------------------------------
    def plot_vector_unit_balls(self, grid_size: int = 300, span: float = 2.0) -> None:
        # Reference vector in R^4
        x_ref = self.x1

        # Use only the first two coordinates for plotting (2D plane)
        x_ref_xy = x_ref[:2]

        # Range of values around the reference point in the x[0]-x[1] plane
        x_vals = np.linspace(x_ref_xy[0] - span, x_ref_xy[0] + span, grid_size)
        y_vals = np.linspace(x_ref_xy[1] - span, x_ref_xy[1] + span, grid_size)
        X, Y = np.meshgrid(x_vals, y_vals)

        # Build 4D points: first two coordinates vary, last two are fixed
        P = np.zeros((grid_size, grid_size, 4))
        P[:, :, 0] = X                  # varying x[0]
        P[:, :, 1] = Y                  # varying x[1]
        P[:, :, 2] = x_ref[2]           # fixed x[2]
        P[:, :, 3] = x_ref[3]           # fixed x[3]

        # Difference between grid points and reference vector
        D = P - x_ref

        # L1 and L2 norms of the difference at each grid point
        L1_field = np.linalg.norm(D, ord=1, axis=2)
        L2_field = np.linalg.norm(D, ord=2, axis=2)

        # Masks for points inside the unit ball (distance <= 1)
        mask_L1 = (L1_field <= 1.0)
        mask_L2 = (L2_field <= 1.0)

        # L1 unit-ball slice (in the x[0]-x[1] plane)
        plt.figure(figsize=(6, 6))
        plt.contourf(X, Y, mask_L1, levels=[0.5, 1], alpha=0.4)
        plt.scatter(x_ref_xy[0], x_ref_xy[1], s=50, label='x_ref')
        plt.title("|x - x_ref|_1 ≤ 1 (2D slice of R^4)")
        plt.xlabel("x[0]")
        plt.ylabel("x[1]")
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "vector_L1_slice.png"), dpi=200)
        plt.show()

        # L2 unit-ball slice (in the x[0]-x[1] plane)
        plt.figure(figsize=(6, 6))
        plt.contourf(X, Y, mask_L2, levels=[0.5, 1], alpha=0.4)
        plt.scatter(x_ref_xy[0], x_ref_xy[1], s=50, label='x_ref')
        plt.title("|x - x_ref|_2 ≤ 1 (2D slice of R^4)")
        plt.xlabel("x[0]")
        plt.ylabel("x[1]")
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "vector_L2_slice.png"), dpi=200)
        plt.show()

    # -------------------------------
    # 5) Matrix unit-ball slices in 2x2 matrix space
    # -------------------------------
    def plot_matrix_unit_balls(self, grid_size_M: int = 250, span_M: float = 2.0) -> None:
        # Reference 2x2 matrix
        M_ref = self.M1.copy()

        # Fix the second row of the matrix (these entries do not change)
        base20 = M_ref[1, 0]  # fixed M[1,0]
        base21 = M_ref[1, 1]  # fixed M[1,1]

        # Vary the first row entries around the reference values
        u_vals = np.linspace(M_ref[0, 0] - span_M, M_ref[0, 0] + span_M, grid_size_M)
        v_vals = np.linspace(M_ref[0, 1] - span_M, M_ref[0, 1] + span_M, grid_size_M)
        U, V = np.meshgrid(u_vals, v_vals)

        # Vectorized differences for the induced matrix 1-norm
        diff00 = U - M_ref[0, 0]                        # difference in M[0,0]
        diff01 = V - M_ref[0, 1]                        # difference in M[0,1]
        diff10 = (base20 - M_ref[1, 0]) * np.ones_like(U)  # difference in M[1,0] (constant)
        diff11 = (base21 - M_ref[1, 1]) * np.ones_like(U)  # difference in M[1,1] (constant)

        # Column sums of absolute values for the 1-norm
        colsum0 = np.abs(diff00) + np.abs(diff10)
        colsum1 = np.abs(diff01) + np.abs(diff11)

        # Induced 1-norm is the maximum of column sums
        mask_mat1 = np.maximum(colsum0, colsum1) <= 1.0

        # Spectral norm mask (loop using full norm computation)
        mask_mat2 = np.zeros_like(U, dtype=bool)
        for i in range(grid_size_M):
            for j in range(grid_size_M):
                # Candidate matrix with varying first row and fixed second row
                M_candidate = np.array([[U[i, j], V[i, j]],
                                        [base20,  base21]])
                diffM = M_candidate - M_ref
                # Check if spectral norm of the difference is within the unit ball
                mask_mat2[i, j] = (self.mat_norm_2(diffM) <= 1.0)

        # 1-norm unit-ball slice in matrix space
        plt.figure(figsize=(6, 6))
        plt.contourf(U, V, mask_mat1, levels=[0.5, 1], alpha=0.4)
        plt.scatter(M_ref[0, 0], M_ref[0, 1], s=50, label='M_ref[0,*]')
        plt.title("|M - M_ref|_1 ≤ 1 (2D slice in matrix space)")
        plt.xlabel("matrix[0,0]")
        plt.ylabel("matrix[0,1]")
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "matrix_1norm_slice.png"), dpi=200)
        plt.show()

        # Spectral 2-norm unit-ball slice in matrix space
        plt.figure(figsize=(6, 6))
        plt.contourf(U, V, mask_mat2, levels=[0.5, 1], alpha=0.4)
        plt.scatter(M_ref[0, 0], M_ref[0, 1], s=50, label='M_ref[0,*]')
        plt.title("|M - M_ref|_2 ≤ 1 (spectral; 2D slice)")
        plt.xlabel("matrix[0,0]")
        plt.ylabel("matrix[0,1]")
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "matrix_2norm_slice.png"), dpi=200)
        plt.show()

    # -------------------------------
    # Run everything
    # -------------------------------
    def run(self) -> None:
        # Print distances between the two vectors and the two matrices
        self.print_distances()

        # Plot unit-ball slices in vector space (R^4)
        self.plot_vector_unit_balls()

        # Plot unit-ball slices in 2x2 matrix space
        self.plot_matrix_unit_balls()


if __name__ == "__main__":
    # Create visualizer and run the full workflow
    visualizer = NormVisualizer()
    visualizer.run()
