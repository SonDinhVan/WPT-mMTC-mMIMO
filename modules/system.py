from configs import config
from modules import node, channel

import cvxpy as cp
import numpy as np
from IPython.display import clear_output
from typing import List
from time import time
import matplotlib.pyplot as plt


# Declare some constants
a = config.a_PARAMETER
b = config.b_PARAMETER
c = config.c_PARAMETER
E_max = config.E_MAX
E_th = config.E_TH
noise = config.NOISE_POWER


def non_linear_EH_func(P_in: float) -> float:
    """
    The non-linear energy harvesting function.

    Args:
        P_in (float): The input power

    Returns:
        float: The out power
    """
    return E_max / (1 - c) * (1 / (1 + np.exp(-a * P_in + a * b)) - c)


class System:
    def __init__(self, M: int, N: int, K: int) -> None:
        """
        Initialization.
        Generate a topology including APs and UEs.
        Assign them into the AP_list and UE_list.
        Calculate and assign the slow fading coefs.
        """
        # Radius of the square area
        r = config.RADIUS
        # Number of APs
        self.M = M
        # Number of antennas at each AP
        self.N = N
        # Number of UE
        self.K = K
        # List of AP
        self.AP_list = [
            node.AP(
                x=r * np.random.uniform(-1, 1),
                y=r * np.random.uniform(-1, 1),
                N=N,
            )
            for M in range(M)
        ]
        # List of UE
        self.UE_list = [
            node.UE(
                x=r * np.random.uniform(-1, 1), y=r * np.random.uniform(-1, 1)
            )
            for k in range(K)
        ]
        # Beta matrix
        self.beta = np.zeros(shape=(M, K))
        for m in range(self.M):
            for k in range(self.K):
                self.beta[m][k] = channel.calculate_NLOS_pathloss(
                    tx=self.AP_list[m], rx=self.UE_list[k]
                )
        # U matrix. This should be calculated
        self.U = np.zeros(shape=(M, K, K))
        # Small scale H matrix and channel coef G matrix. This should be
        # generated every timeslot
        # if you want to verify the analysis
        self.H = np.zeros(shape=(M, K, N, 1), dtype="complex")
        self.G = np.zeros(shape=(M, K, N, 1), dtype="complex")
        # Number of pilot sequences
        self.tau_p = None
        # Length of coherence
        self.tau_c = config.TAU_C
        # Number of selected APs for each UE
        self.M0 = None

    def sketch(self) -> None:
        """
        This function plots the network topology, i.e., the locations of APs and UEs.
        """
        for m in range(self.M):
            plt.scatter(
                x=self.AP_list[m].x,
                y=self.AP_list[m].y,
                marker="o",
                color="red",
            )
            plt.text(
                x=self.AP_list[m].x + 0.5,
                y=self.AP_list[m].y + 1.0,
                s="AP " + str(m),
                color="red",
            )

        for k in range(self.K):
            plt.scatter(
                x=self.UE_list[k].x,
                y=self.UE_list[k].y,
                marker="x",
                color="blue",
            )
            plt.text(
                x=self.UE_list[k].x + 0.5,
                y=self.UE_list[k].y,
                s="UE " + str(k),
                color="blue",
            )
            try:
                plt.text(
                    x=self.UE_list[k].x - 0.5,
                    y=self.UE_list[k].y - 2.5,
                    s="Pilot " + str(self.UE_list[k].pilot_index),
                    color="black",
                )
            except Exception as e:
                print(e)

        plt.show()

    def assign_pilot(self, option="random") -> None:
        """
        Assign pilot for each UE.

        Args:
            option (str, optional): Strategy to assign the UE.
            Defaults to 'random'.
            If 'random': Randomly select pilot.
            If 'orthogonal': Use orthogonal pilots.
            If 'same': Use the same pilot.
            If 'greedy': Use the greedy pilot.
            If 'propose': Use the proposed pilot.
        """
        assert option in [
            "random",
            "orthogonal",
            "same",
            "greedy",
            "propose",
        ], "option should be assign: random, orthogonal, same, greedy, propose"
        # Pilot matrix. Each user selects a random pilot sequence as a column
        # from this matrix.
        pilot = np.eye(self.tau_p)
        if option == "random":
            for UE in self.UE_list:
                index = np.random.choice(self.tau_p)
                UE.pilot_index = index
                UE.varphi = pilot[index].reshape(-1, 1)

        elif option == "orthogonal":
            assert (
                not self.tau_p < self.K
            ), "Orthogonal pilots can be used only when tau_p >= K"
            for k in range(self.K):
                self.UE_list[k].pilot_index = k
                self.UE_list[k].varphi = pilot[k].reshape(-1, 1)

        elif option == "same":
            for UE in self.UE_list:
                UE.pilot_index = 0
                UE.varphi = pilot[0].reshape(-1, 1)

        elif option == "greedy":

            def calculate_contamination_k(k: int):
                """
                Calculate the contamination at UE k if it uses a given pilot.
                """
                return np.sum(
                    [
                        [
                            self.tau_p
                            * self.beta[m, j]
                            * np.linalg.norm(
                                np.mat(self.UE_list[j].varphi).getH()
                                * np.mat(self.UE_list[k].varphi)
                            )
                            ** 2
                            for j in range(self.K)
                            if j != k
                        ]
                        for m in range(self.M)
                    ]
                )

            def calculate_sum_contamination():
                """
                Calculate the sum of contamination at all UEs.
                """
                return np.sum(
                    [calculate_contamination_k(k) for k in range(self.K)]
                )

            def sum_contamination_vector(k):
                res = self.tau_p * [None]
                temp = self.UE_list[k].varphi
                for i in range(self.tau_p):
                    self.UE_list[k].varphi = pilot[i].reshape(-1, 1)
                    res[i] = calculate_sum_contamination()
                self.UE_list[k].varphi = temp

                return res

            def assign_pilot(k):
                """
                Assign pilot for UE k
                """
                index = np.argmin(sum_contamination_vector(k))
                self.UE_list[k].varphi = pilot[index].reshape(-1, 1)
                self.UE_list[k].pilot_index = index

            self.assign_pilot("random")
            for k in range(self.K):
                assign_pilot(k)

        elif option == "propose":
            # Create a pilot dictionary to store which UEs are using pilots
            pilot_dict = {i: [] for i in range(self.tau_p)}
            # Assign random pilot
            for k in range(self.K):
                index = np.random.choice(range(self.tau_p))
                pilot_dict[index].append(k)
                self.UE_list[k].varphi = pilot[index].reshape(-1, 1)
                self.UE_list[k].pilot_index = index

            def find_intersec(x, y):
                """
                Get the number of common elements between set x and set y
                """
                return len([e for e in x if e in y])

            def find_average_number_coshared_AP(k, i):
                # Consider UE k and pilot i
                # Assume that UE a, b, c, ... are using pilot i
                # This function find the average number of APs serving UE k
                # that are also serve UE a, b, c, ...
                if len(pilot_dict[i]) == 0:
                    return 0
                else:
                    return np.mean(
                        [
                            find_intersec(
                                self.UE_list[k].AP_index,
                                self.UE_list[j].AP_index,
                            )
                            for j in pilot_dict[i]
                        ]
                    )

            def calculate_sum_contamination(k: int, pilot: np.array):
                """
                Calculate the sum of contamination at UE k if it uses a given
                pilot.
                """
                return np.sum(
                    [
                        [
                            self.tau_p
                            * self.beta[m, j]
                            * np.linalg.norm(
                                np.mat(self.UE_list[j].varphi).getH()
                                * np.mat(pilot)
                            )
                            ** 2
                            for j in range(self.K)
                            if j != k
                        ]
                        for m in range(self.M)
                    ]
                )

            def update_pilot(k):
                # This function update pilot for UE k and update the
                # pilot_dict accordingly
                average_num_shared_AP = [
                    find_average_number_coshared_AP(k, i)
                    for i in range(self.tau_p)
                ]
                # print("UE", k)
                # print(share_AP_info)
                # print(pilot_dict)
                # index of the assigned pilot (possible pilot).
                index = [
                    idx
                    for idx, val in enumerate(average_num_shared_AP)
                    if val == np.min(average_num_shared_AP)
                ]
                # print(index)
                s = [
                    calculate_sum_contamination(k, pilot[i].reshape(-1, 1))
                    for i in index
                ]
                # Update the pilot_dict by removing the old pilot of UE k
                pilot_dict[self.UE_list[k].pilot_index].remove(k)
                index_choice = index[np.argmin(s)]
                # print(index_choice)
                self.UE_list[k].pilot_index = index_choice
                self.UE_list[k].varphi = pilot[index_choice].reshape(-1, 1)
                # Update the pilot_dict by adding the new pilot of UE k
                pilot_dict[index_choice].append(k)
                # print(pilot_dict)

            N_iter = self.K * 2
            count = 0
            k = 0
            while count <= N_iter:
                update_pilot(k)
                k += 1
                if k == self.K:
                    k = 0
                count += 1

    def assign_APs(self) -> None:
        """
        Assign the APs to each UE.
        Each UE selects M0 APs with the largest beta values.
        """
        # Assign AP indices to each UE
        for k in range(self.K):
            beta_arr = self.beta[:, k]
            self.UE_list[k].AP_index = list(
                np.argsort(beta_arr)[self.M - self.M0:]
            )

        # Assign UE indices to each AP
        for AP in self.AP_list:
            AP.UE_index = []
        for k in range(self.K):
            for index in self.UE_list[k].AP_index:
                self.AP_list[index].UE_index.append(k)

    def assign_power_weights(self, option="random") -> None:
        """
        Assign power weights for all APs in the system.
        Option can be assigned "random" or "uniform".
        """
        assert option in [
            "random",
            "uniform",
        ], "option should be assigned: random, uniform"
        if option == "random":
            for AP in self.AP_list:
                AP.generate_random_power_weights()
        if option == "uniform":
            for AP in self.AP_list:
                AP.generate_uniform_power_weights()

    def generate_small_scale_fadings(self) -> None:
        """
        Generate the small scale fading coeficient H and G.
        """
        for m in range(self.M):
            for k in range(self.K):
                self.H[m][k] = channel.generate_h(
                    tx=self.AP_list[m], rx=self.UE_list[k]
                )
                self.G[m][k] = self.beta[m][k] ** 0.5 * self.H[m][k]

    def estimate_gmk(self, m: int, k: int) -> np.array:
        """
        Estimate channel g_mk
        """
        # Generate the noise matrix
        noise_matrix = np.random.normal(
            size=(self.N, self.tau_p)
        ) + 1j * np.random.normal(size=(self.N, self.tau_p))
        noise_matrix *= noise * 0.5
        # The AP m receives
        Y = (
            np.sum(
                [
                    np.sqrt(self.tau_p)
                    * np.sqrt(self.UE_list[k_].P)
                    * self.G[m][k_]
                    * np.mat(self.UE_list[k_].varphi).getH()
                    for k_ in range(self.K)
                ],
                axis=0,
            )
            + noise_matrix
        )

        # The projected of Y onto varphi[k]
        projected_Y = np.mat(Y) * np.mat(self.UE_list[k].varphi)

        # The channel estimation of g_mk
        up = (self.tau_p * self.UE_list[k].P) ** 0.5 * self.beta[m][k]
        down = (
            self.tau_p
            * np.sum(
                [
                    self.UE_list[k_].P
                    * self.beta[m][k_]
                    * (
                        np.mat(self.UE_list[k_].varphi).getH()
                        * np.mat(self.UE_list[k].varphi)
                    )
                    ** 2
                    for k_ in range(self.K)
                ]
            )
            + noise
        )

        return np.array(up / down * projected_Y)

    def estimate_G(self) -> np.array:
        """
        Estimate the channel G

        Returns:
            np.array: Estimated channel G
        """
        G_estimated = np.zeros(
            shape=(self.M, self.K, self.N, 1), dtype="complex"
        )

        for m in range(self.M):
            for k in range(self.K):
                G_estimated[m][k] = self.estimate_gmk(m, k)

        return G_estimated

    def calculate_gamma_mk(self, m: int, k: int) -> float:
        """
        Calculate gamma_{mk}
        """
        return (self.tau_p * self.UE_list[k].P * self.beta[m, k] ** 2) / (
            np.sum(
                [
                    self.tau_p
                    * self.UE_list[j].P
                    * self.beta[m, j]
                    * np.abs(
                        np.mat(self.UE_list[j].varphi).getH()
                        * np.mat(self.UE_list[k].varphi)
                    ).item()
                    ** 2
                    for j in range(self.K)
                ]
            )
            + noise
        )

    def calculate_U(self) -> None:
        """
        Calculate array U shape = (M, K, K).
        Where U[m,k,i] = u_{mki}
        """
        self.U = np.array(
            [
                [
                    [
                        self.tau_p
                        * self.beta[m, k]
                        * np.abs(
                            np.mat(self.UE_list[k].varphi).getH()
                            * np.mat(self.UE_list[i].varphi)
                        ).item()
                        ** 2
                        for i in range(self.K)
                    ]
                    for k in range(self.K)
                ]
                for m in range(self.M)
            ]
        )

    def calculate_f(self, m, k, i) -> float:
        """
        Calculate the norm square term using analysis
        |g_{mk}^H * \hat{g}_{mi} / |\norm{\hat{g}_{mi}}||^2 = f_{mki}(p)
        """
        up = (self.N - 1) * self.U[m, k, i] * self.UE_list[k].P - (
            1 - 1 / self.N
        ) * noise
        down = (
            np.sum(
                [self.U[m, j, i] * self.UE_list[j].P for j in range(self.K)]
            )
            + noise
        )

        return self.beta[m, k] * (1 + up / down)

    def calculate_average_received_power_by_analysis(self, k: int) -> float:
        """
        Given power vector p, and beamforming weights W.
        Calculate the average received power at UE k (in Watts)
        """
        total_p = 0
        for m in range(self.M):
            if len(self.AP_list[m].UE_index) > 0:
                f_mk = np.array(
                    [
                        self.calculate_f(m, k, i)
                        for i in self.AP_list[m].UE_index
                    ]
                )
                total_p += (
                    self.AP_list[m].P
                    * np.mat(f_mk)
                    * np.mat(self.AP_list[m].w)
                )

        return total_p.item()

    def calculate_average_accumulated_power_by_simulation(
        self, k: int
    ) -> float:
        """
        Calculate the average accumulated power using simulation (in Watts).

        Args:
            k (int): Index of UE
        """
        # verify the bound
        res = 0
        N_loop = 100
        W = self.get_beamforming_weight_W()
        for j in range(N_loop):
            if j % 10 == 0:
                clear_output(wait=True)
                print("Processing ", j / N_loop * 100, "%")
            sum = 0
            for m in range(self.M):
                if len(self.AP_list[m].UE_index) > 0:
                    for i in self.AP_list[m].UE_index:
                        self.generate_small_scale_fadings()
                        g_mk = self.G[m, k]
                        g_mi_hat = self.estimate_gmk(m=m, k=i)
                        sum += (
                            self.AP_list[m].P
                            * W[m, i]
                            * np.abs(
                                np.mat(g_mk).getH()
                                * np.mat(g_mi_hat)
                                / np.linalg.norm(g_mi_hat)
                            )
                            ** 2
                        )

            res += (1 - self.tau_p / self.tau_c) * non_linear_EH_func(
                sum
            ) - self.tau_p / self.tau_c * self.UE_list[k].P

        return res.item() / N_loop

    def calculate_upper_bound_UE_k_using_simulation_and_analysis(
        self, k: int
    ) -> List:
        """
        This function calculates the lower bound of the accumulated power at
        the UE k

        Args:
            k (int): Index of the UE .

        Returns:
            List: lower bound of UE k using analysis, simulation (in Watts).
        """
        res = 0
        N_loop = 100
        for j in range(N_loop):
            if j % 10 == 0:
                clear_output(wait=True)
                print("Processing ", j / N_loop * 100, "%")
            sum = 0
            for m in range(self.M):
                if len(self.AP_list[m].UE_index) > 0:
                    exp_vec = []
                    for i in self.AP_list[m].UE_index:
                        self.generate_small_scale_fadings()
                        g_mk = self.G[m, k]
                        g_mi_hat = self.estimate_gmk(m=m, k=i)
                        exp_vec.append(
                            np.mat(g_mk).getH()
                            * np.mat(g_mi_hat)
                            / np.linalg.norm(g_mi_hat)
                            * np.random.normal(0, 1)
                        )

                    exp_vec = np.array(exp_vec)
                    sum += (
                        np.sqrt(self.AP_list[m].P)
                        * np.mat(exp_vec)
                        * np.sqrt(np.mat(self.AP_list[m].w))
                    )
            res += np.abs(sum) ** 2

        # Simulation result
        sim = (
            (self.tau_c - self.tau_p) * non_linear_EH_func(res / N_loop)
            - self.tau_p * self.UE_list[k].P
        ) / self.tau_c
        # Analysis result
        analysis = (
            (self.tau_c - self.tau_p)
            * non_linear_EH_func(
                self.calculate_average_received_power_by_analysis(k)
            )
            - self.tau_p * self.UE_list[k].P
        ) / self.tau_c

        return sim.item(), analysis

    def calculate_upper_bound_of_mean_accumulated_power_using_simulation_and_analysis(
        self,
    ) -> List:
        """
        This function calculates the lower bound of the mean accumulated power
        using simulation and analysis

        Returns:
            List: Upper bound calculated by simulation and analysis (in Watts)
        """
        sim = 0
        analysis = 0
        for k in range(self.K):
            (
                s1,
                a1,
            ) = self.calculate_upper_bound_UE_k_using_simulation_and_analysis(
                k
            )
            sim += s1
            analysis += a1

        return sim / self.K, analysis / self.K

    def construct_vector_f_mk(self, m: int, k: int) -> np.array:
        """
        Construct the vector f_{mk}

        Args:
            m (int): AP index
            k (int): UE index

        Returns:
            np.array: Vector f_{mk}
        """
        assert (
            len(self.AP_list[m].UE_index) > 0
        ), "AP {} does not serve any UEs".format(m)
        return np.array(
            [self.calculate_f(m, k, i) for i in self.AP_list[m].UE_index]
        )

    def calculate_upper_bound_UE_k_using_analysis(self, k: int) -> float:
        """
        Calculate the accumulated power at UE k (in Watt)
        """
        return (1 - self.tau_p / self.tau_c) * non_linear_EH_func(
            self.calculate_average_received_power_by_analysis(k)
        ) - self.tau_p / self.tau_c * self.UE_list[k].P

    def calculate_upper_bound_of_mean_accumulated_power(self) -> float:
        """
        Calculate average accumulated power of all UEs.

        Returns:
            float: Power in W
        """
        return np.mean(
            [
                self.calculate_upper_bound_UE_k_using_analysis(k)
                for k in range(self.K)
            ]
        )

    def get_beamforming_weight_W(self) -> np.array:
        """
        This function create a matrix W, where its element, i.e. W[m, k]
        refers to the power weight that AP m uses for AP k.

        Returns:
            np.array: W
        """

        # Find the weight used by an AP m for UE k
        def get_w_mk(m, k):
            try:
                index = self.AP_list[m].UE_index.index(k)
                return self.AP_list[m].w[index, 0]
            except:
                return 0

        return np.array(
            [[get_w_mk(m, k) for k in range(self.K)] for m in range(self.M)]
        )

    def solve_W_by_fixing_P(self, tol=10**-3, max_iter=10) -> List:
        """
        Find the optimal value of W by fixing P. Note that this function will
        change the current setting of W to optimal W, i.e., each AP will be
        assigned new weights w_{m,i} after performing this function.

        Args:
            tol (float, optional): Tolerance. Defaults to 10**-3.
            max_iter (int, optional): Maximum number of iteration.
                Defaults to 10.

        Returns:
            x: optimal x
            z: optimal z
        """

        # Constant parameters
        D = (self.tau_c - self.tau_p) * E_max / (1 - c) / self.tau_c
        F = D * c + E_th
        p = np.array([self.UE_list[i].P for i in range(self.K)]).reshape(-1, 1)
        vector_f = [
            [
                self.construct_vector_f_mk(m=m, k=k)
                if len(self.AP_list[m].UE_index) > 0
                else None
                for k in range(self.K)
            ]
            for m in range(self.M)
        ]

        # Initiate values for the case when CVX fails right at the beginning
        # This rarely happens
        self.assign_power_weights("uniform")
        p_in = np.array(
            [
                self.calculate_average_received_power_by_analysis(k)
                for k in range(self.K)
            ]
        ).reshape(-1, 1)
        x_ = 1.0 / (1 + np.exp(-a * p_in + a * b))
        z_ = a * p_in
        W_ = [AP.w for AP in self.AP_list]

        # Initiate a value of alpha
        alpha = np.ones((self.K, 1))

        # Previous and current optimal value
        pre_optimal_val = -1000
        current_optimal_val = 0

        # Trial order
        trial = 0

        # Variables
        W = [
            cp.Variable((len(AP.UE_index), 1))
            if len(AP.UE_index) > 0
            else None
            for AP in self.AP_list
        ]
        x = cp.Variable((self.K, 1))
        y = cp.Variable((self.K, 1))
        z = cp.Variable((self.K, 1))
        # Introduce a slack variable t to control the feasibility of the
        # optimization problem
        t = cp.Variable((self.K, 1), nonneg=True)

        while (trial < max_iter) and not np.isclose(
            pre_optimal_val, current_optimal_val, rtol=tol
        ):
            trial += 1
            # Assign the current optimal value to previous one
            pre_optimal_val = current_optimal_val

            # Constraints
            constr = []
            for w in W:
                if w is not None:
                    # Constraint 21: C2
                    constr += [cp.sum(w) <= 1]
                    # Constraint 21: C4
                    constr += [w >= 0]
            # Constraint 22: C2
            constr += [z + y >= a * b]
            # Constraint 22: C4
            constr += [x - self.tau_p / self.tau_c / D * p >= F / D]
            # Constraint 22: C3
            for k in range(self.K):
                sum = 0
                for m in range(self.M):
                    if len(self.AP_list[m].UE_index) > 0:
                        sum = (
                            sum + a * self.AP_list[m].P * vector_f[m][k] @ W[m]
                        )
                constr += [sum >= z[k]]
            # Constraint 25: C1
            constr += [
                cp.exp(y) + 1 <= 2 * alpha - cp.multiply(x, alpha**2) + t
            ]

            # Objective function
            f = cp.sum(
                x + 2 * alpha - cp.multiply(x, alpha**2) - cp.inv_pos(x) - t
            )
            objective = cp.Maximize(f)
            problem = cp.Problem(objective, constr)

            # Solve the problem
            try:
                problem.solve(solver=cp.MOSEK, verbose=False)
                current_optimal_val = problem.value
            except Exception as e:
                print(e)
                break

            if x.value is None:
                break

            # If x is not None and no error occur, we assign optimal value of
            # x, z and W into x_, z_ and W_
            x_ = x.value
            z_ = z.value
            W_ = [w.value if w is not None else None for w in W]
            # We also assign a new alpha = 1/x_ for the new iteration
            alpha = 1 / x_

        # Assign new beamforming weights to the system
        for i in range(self.M):
            if W_[i] is not None:
                self.AP_list[i].w = W_[i]

        return x_, z_

    def solve_P_by_fixing_W(
        self, x: np.array, z: np.array, tol=10**-3, max_iter=10
    ) -> float:
        """
        Find the optimal value of P by fixing W. Note that this function will
        change the current setting of P_UE to optimal P, i.e., each UE will be
        assigned with a new power after performing this function.

        Args:
            x (np.array): The value of x achieved from running optimize W
            z (np.array): The value of z achieved from running optimize W
            tol (float, optional): Tolerance. Defaults to 10**-3.
            max_iter (int, optional): Maximum number of iteration.
                Defaults to 10.
        Returns:
            Optimal average harvested power
        """
        # The column vector of transmit power used by UEs
        p_ = np.array([UE.P for UE in self.UE_list]).reshape(-1, 1)
        # Parameters
        # W[m, i] refers to the element w_{m, i}
        W = self.get_beamforming_weight_W()
        D = (self.tau_c - self.tau_p) * E_max / (1 - c) / self.tau_c
        F = D * c + E_th
        # Initiate the value of v_{mi}
        v = np.zeros(shape=(self.M, self.K))
        for m in range(self.M):
            for i in range(self.K):
                vector_u_mi = np.array(
                    [
                        self.U[m, j, i] / noise / a / self.AP_list[m].P
                        for j in range(self.K)
                    ]
                )
                v[m, i] = 1 / (
                    np.matmul(vector_u_mi, p_) + 1 / a / self.AP_list[m].P
                )

        trial = 0
        current_optimal_val = 0
        pre_optimal_val = -1000

        # Variables
        p = cp.Variable((self.K, 1), pos=True)
        # Introduce an additional variable to handle the infeasibility
        t = cp.Variable((self.K, 1), nonneg=True)

        while (trial < max_iter) and not np.isclose(
            pre_optimal_val, current_optimal_val, rtol=tol
        ):
            trial += 1
            # Assign the current optimal value to previous one
            pre_optimal_val = current_optimal_val

            # Constraints
            constr = []
            constr += [p >= 10**-6]
            constr += [p <= 40]
            constr += [D * x - self.tau_p * p >= F]

            for k in range(self.K):
                sum = 0
                for m in range(self.M):
                    for i in self.AP_list[m].UE_index:
                        vector_u_mki = (
                            np.array(
                                [
                                    self.U[m, j, i] * self.beta[m, k]
                                    if j != k
                                    else self.N
                                    * self.U[m, k, i]
                                    * self.beta[m, k]
                                    for j in range(self.K)
                                ]
                            )
                            / noise
                        )
                        sum += (
                            W[m, i]
                            * v[m, i]
                            * (vector_u_mki @ p + self.beta[m, k] / self.N)
                        )

                constr += [sum + t[k] >= z[k]]

            for m in range(self.M):
                for i in self.AP_list[m].UE_index:
                    vector_u_mi = np.array(
                        [
                            self.U[m, j, i] / noise / a / self.AP_list[m].P
                            for j in range(self.K)
                        ]
                    )
                    constr += [
                        vector_u_mi @ p + 1 / a / self.AP_list[m].P
                        <= 1 / v[m, i]
                    ]

            # Objective function
            obj = -cp.sum(p + t)

            objective = cp.Maximize(obj)
            problem = cp.Problem(objective, constr)

            # Solve the problem
            try:
                problem.solve(solver=cp.MOSEK, verbose=False)
                current_optimal_val = problem.value
            except Exception as e:
                print(e)
                break

            if p.value is None:
                break

            # If CVX succeeds in solving the optimization problem
            p_ = p.value
            # Assign new v_{m,i} for the new iteration
            for m in range(self.M):
                for i in range(self.K):
                    vector_u_mi = np.array(
                        [
                            self.U[m, j, i] / noise / a / self.AP_list[m].P
                            for j in range(self.K)
                        ]
                    )
                    v[m, i] = 1 / (np.matmul(vector_u_mi, p_) + 1)
        for i in range(self.K):
            self.UE_list[i].P = p_[i, 0]

        return self.calculate_upper_bound_of_mean_accumulated_power()

    def solve_optimization_jointly_W_P(
        self, tol=10**-3, max_iter=10
    ) -> float:
        """
        Solve the optimization by iteratively running the previous two
            functions.
        I.e., Fix P Solve W => Then Fix W Solve P => ... until convergence.

        Args:
            tol (_type_, optional): Tolerance. Defaults to 10**-3.
            max_iter (int, optional): Maximum number of iteration.
                Defaults to 10.
        Returns:
            Optimal average harvested power
        """
        start = time()
        trial = 0
        current_optimal_val = 0
        pre_optimal_val = -1000

        while (trial < max_iter) and not np.isclose(
            pre_optimal_val, current_optimal_val, rtol=tol
        ):
            trial += 1
            # Assign the current optimal value to previous one
            pre_optimal_val = current_optimal_val
            # Iteratively solve
            x_, z_ = self.solve_W_by_fixing_P()
            print(
                self.calculate_upper_bound_of_mean_accumulated_power() * 1000
            )
            current_optimal_val = self.solve_P_by_fixing_W(x=x_, z=z_)
            print(
                self.calculate_upper_bound_of_mean_accumulated_power() * 1000
            )

        end = time()
        print(
            "Finish after {} iterations, take {:.2f} seconds.".format(
                trial, end - start
            )
        )
