from __future__ import annotations

import numpy as np


class ConstantAccelerationKalmanFilter:
    """Constant Acceleration Kalman Filter (CA-KF) for 2D position tracking.

    State vector:
        x = [x, y, vx, vy, ax, ay]^T

    Dynamics:
        x(t + dt)  = x(t) + vx(t)*dt + 0.5*ax(t)*dt^2
        y(t + dt)  = y(t) + vy(t)*dt + 0.5*ay(t)*dt^2
        vx(t + dt) = vx(t) + ax(t)*dt
        vy(t + dt) = vy(t) + ay(t)*dt
        ax(t + dt) = ax(t)
        ay(t + dt) = ay(t)

    Continuous White Noise Acceleration (CWPA) model is used to form Q(dt).
    The Joseph stabilized form is used for covariance updates.
    """

    def __init__(
        self,
        initial_position: tuple[float, float],
        process_noise_std: float = 1.0,
        measurement_noise_std: float = 2.0,
        initial_velocity: tuple[float, float] = (0.0, 0.0),
        initial_acceleration: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """Initialize CA-KF with initial state and noise covariances."""
        self._q_std = max(1e-4, float(process_noise_std))
        self._r_std = max(1e-4, float(measurement_noise_std))

        # State: [x, y, vx, vy, ax, ay]^T
        self.x = np.array(
            [
                initial_position[0],
                initial_position[1],
                initial_velocity[0],
                initial_velocity[1],
                initial_acceleration[0],
                initial_acceleration[1],
            ],
            dtype=np.float64,
        )

        # Initial covariance P
        r_var = self._r_std**2
        self.P = np.diag([r_var, r_var, 100.0, 100.0, 50.0, 50.0]).astype(np.float64)

        # Measurement matrix H: maps state to [x, y]
        self.H = np.zeros((2, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0

        # Measurement noise R
        self.R = np.eye(2, dtype=np.float64) * r_var
        self.I6 = np.eye(6, dtype=np.float64)

    # ------------------------------------------------------------------ #
    #  State Transition and Process Noise Matrices
    # ------------------------------------------------------------------ #

    @staticmethod
    def state_transition_matrix(dt: float) -> np.ndarray:
        """Construct 6x6 state transition matrix F(dt)."""
        dt2 = 0.5 * dt * dt
        F = np.eye(6, dtype=np.float64)
        F[0, 2] = dt
        F[0, 4] = dt2
        F[1, 3] = dt
        F[1, 5] = dt2
        F[2, 4] = dt
        F[3, 5] = dt
        return F

    def process_noise_matrix(self, dt: float) -> np.ndarray:
        """Construct 6x6 continuous white noise process covariance Q(dt)."""
        dt = max(dt, 1e-4)
        dt5 = (dt**5) / 20.0
        dt4 = (dt**4) / 8.0
        dt3 = (dt**3) / 6.0
        dt2 = (dt**2) / 2.0

        q_var = self._q_std**2

        # 3x3 block for single dimension [pos, vel, acc]
        q_1d = q_var * np.array(
            [
                [dt5, dt4, dt3],
                [dt4, (dt**3) / 3.0, dt2],
                [dt3, dt2, dt],
            ],
            dtype=np.float64,
        )

        Q = np.zeros((6, 6), dtype=np.float64)
        # x-axis components: indices (0, 2, 4)
        x_idx = [0, 2, 4]
        for r_i, r in enumerate(x_idx):
            for c_i, c in enumerate(x_idx):
                Q[r, c] = q_1d[r_i, c_i]

        # y-axis components: indices (1, 3, 5)
        y_idx = [1, 3, 5]
        for r_i, r in enumerate(y_idx):
            for c_i, c in enumerate(y_idx):
                Q[r, c] = q_1d[r_i, c_i]

        return Q

    # ------------------------------------------------------------------ #
    #  Predict and Update Cycles
    # ------------------------------------------------------------------ #

    def predict(self, dt: float) -> None:
        """Propagate state and covariance forward by elapsed time dt seconds."""
        if dt <= 0:
            return

        F = self.state_transition_matrix(dt)
        Q = self.process_noise_matrix(dt)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, measurement: tuple[float, float] | np.ndarray) -> None:
        """Incorporate new position measurement [x, y] using Joseph form."""
        z = np.asarray(measurement, dtype=np.float64).reshape(2)

        # Innovation (residual)
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Joseph form update for numerical stability: P = (I - KH) P (I - KH)^T + K R K^T
        I_KH = self.I6 - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

    # ------------------------------------------------------------------ #
    #  Future Position Extrapolation
    # ------------------------------------------------------------------ #

    def predict_future(
        self, dt_offsets: list[float]
    ) -> list[tuple[float, float, tuple[float, float]]]:
        """Extrapolate future positions without altering the current filter state.

        Args:
            dt_offsets: Forward time offsets in seconds from current time.

        Returns:
            List of (pred_x, pred_y, (var_x, var_y)) for each offset.
        """
        results: list[tuple[float, float, tuple[float, float]]] = []

        for offset in dt_offsets:
            if offset <= 0:
                results.append((float(self.x[0]), float(self.x[1]), (float(self.P[0, 0]), float(self.P[1, 1]))))
                continue

            F = self.state_transition_matrix(offset)
            Q = self.process_noise_matrix(offset)

            future_x = F @ self.x
            future_P = F @ self.P @ F.T + Q

            pred_pos = (float(future_x[0]), float(future_x[1]))
            pred_var = (float(max(0.0, future_P[0, 0])), float(max(0.0, future_P[1, 1])))
            results.append((pred_pos[0], pred_pos[1], pred_var))

        return results

    # ------------------------------------------------------------------ #
    #  State Accessors
    # ------------------------------------------------------------------ #

    @property
    def position(self) -> tuple[float, float]:
        """Estimated current (x, y) coordinates."""
        return (float(self.x[0]), float(self.x[1]))

    @property
    def velocity(self) -> tuple[float, float]:
        """Estimated velocity (vx, vy) in px/s."""
        return (float(self.x[2]), float(self.x[3]))

    @property
    def acceleration(self) -> tuple[float, float]:
        """Estimated acceleration (ax, ay) in px/s^2."""
        return (float(self.x[4]), float(self.x[5]))

    @property
    def speed(self) -> float:
        """Magnitude of velocity vector in px/s."""
        return float(np.hypot(self.x[2], self.x[3]))

    @property
    def position_variance(self) -> tuple[float, float]:
        """Positional variances (var_x, var_y)."""
        return (float(max(0.0, self.P[0, 0])), float(max(0.0, self.P[1, 1])))


class ConstantVelocityKalmanFilter:
    """Constant Velocity Kalman Filter (CV-KF) for 2D position tracking baseline.

    State vector:
        x = [x, y, vx, vy]^T

    Dynamics:
        x(t + dt)  = x(t) + vx(t)*dt
        y(t + dt)  = y(t) + vy(t)*dt
        vx(t + dt) = vx(t)
        vy(t + dt) = vy(t)

    Acceleration is implicitly zero.
    Continuous White Noise Velocity (CWPV) model is used to form Q(dt).
    """

    def __init__(
        self,
        initial_position: tuple[float, float],
        process_noise_std: float = 1.0,
        measurement_noise_std: float = 2.0,
        initial_velocity: tuple[float, float] = (0.0, 0.0),
        **kwargs: Any,
    ) -> None:
        self._q_std = max(1e-4, float(process_noise_std))
        self._r_std = max(1e-4, float(measurement_noise_std))

        # State: [x, y, vx, vy]^T
        self.x = np.array(
            [
                initial_position[0],
                initial_position[1],
                initial_velocity[0],
                initial_velocity[1],
            ],
            dtype=np.float64,
        )

        r_var = self._r_std**2
        self.P = np.diag([r_var, r_var, 100.0, 100.0]).astype(np.float64)

        # Measurement matrix H: maps state to [x, y]
        self.H = np.zeros((2, 4), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0

        self.R = np.eye(2, dtype=np.float64) * r_var
        self.I4 = np.eye(4, dtype=np.float64)

    @staticmethod
    def state_transition_matrix(dt: float) -> np.ndarray:
        """Construct 4x4 state transition matrix F(dt)."""
        F = np.eye(4, dtype=np.float64)
        F[0, 2] = dt
        F[1, 3] = dt
        return F

    def process_noise_matrix(self, dt: float) -> np.ndarray:
        """Construct 4x4 process covariance Q(dt)."""
        dt = max(dt, 1e-4)
        dt3 = (dt**3) / 3.0
        dt2 = (dt**2) / 2.0

        q_var = self._q_std**2

        q_1d = q_var * np.array(
            [
                [dt3, dt2],
                [dt2, dt],
            ],
            dtype=np.float64,
        )

        Q = np.zeros((4, 4), dtype=np.float64)
        # x-axis components: indices (0, 2)
        x_idx = [0, 2]
        for r_i, r in enumerate(x_idx):
            for c_i, c in enumerate(x_idx):
                Q[r, c] = q_1d[r_i, c_i]

        # y-axis components: indices (1, 3)
        y_idx = [1, 3]
        for r_i, r in enumerate(y_idx):
            for c_i, c in enumerate(y_idx):
                Q[r, c] = q_1d[r_i, c_i]

        return Q

    def predict(self, dt: float) -> None:
        """Propagate state and covariance forward by elapsed time dt seconds."""
        if dt <= 0:
            return

        F = self.state_transition_matrix(dt)
        Q = self.process_noise_matrix(dt)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, measurement: tuple[float, float] | np.ndarray) -> None:
        """Incorporate new position measurement [x, y] using Joseph form."""
        z = np.asarray(measurement, dtype=np.float64).reshape(2)

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y

        I_KH = self.I4 - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

    def predict_future(
        self, dt_offsets: list[float]
    ) -> list[tuple[float, float, tuple[float, float]]]:
        """Extrapolate future positions along constant velocity trajectory."""
        results: list[tuple[float, float, tuple[float, float]]] = []

        for offset in dt_offsets:
            if offset <= 0:
                results.append((float(self.x[0]), float(self.x[1]), (float(self.P[0, 0]), float(self.P[1, 1]))))
                continue

            F = self.state_transition_matrix(offset)
            Q = self.process_noise_matrix(offset)

            future_x = F @ self.x
            future_P = F @ self.P @ F.T + Q

            pred_pos = (float(future_x[0]), float(future_x[1]))
            pred_var = (float(max(0.0, future_P[0, 0])), float(max(0.0, future_P[1, 1])))
            results.append((pred_pos[0], pred_pos[1], pred_var))

        return results

    @property
    def position(self) -> tuple[float, float]:
        """Estimated current (x, y) coordinates."""
        return (float(self.x[0]), float(self.x[1]))

    @property
    def velocity(self) -> tuple[float, float]:
        """Estimated velocity (vx, vy) in px/s."""
        return (float(self.x[2]), float(self.x[3]))

    @property
    def acceleration(self) -> tuple[float, float]:
        """Constant velocity baseline has zero acceleration."""
        return (0.0, 0.0)

    @property
    def speed(self) -> float:
        """Magnitude of velocity vector in px/s."""
        return float(np.hypot(self.x[2], self.x[3]))

    @property
    def position_variance(self) -> tuple[float, float]:
        """Positional variances (var_x, var_y)."""
        return (float(max(0.0, self.P[0, 0])), float(max(0.0, self.P[1, 1])))
