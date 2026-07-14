"""
GPU-accelerated Kalman Filter for batched multi-stream tracking.

This module provides a PyTorch-based implementation of the Kalman filter
that can process multiple tracks across multiple streams in a single batch.
"""

import torch
from typing import Tuple


class GPUKalmanFilter:
    """
    Constant-velocity Kalman filter implemented in PyTorch for GPU acceleration.

    State space: [x, y, a, h, vx, vy, va, vh]
    - x, y: bounding box center
    - a: aspect ratio (width / height)
    - h: height
    - vx, vy, va, vh: velocities

    Measurement space: [x, y, a, h]
    """

    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device)

        # Motion model: constant velocity
        # State transition matrix (8x8)
        self._motion_mat = torch.eye(8, device=self.device, dtype=torch.float32)
        self._motion_mat[0, 4] = 1.0  # x += vx
        self._motion_mat[1, 5] = 1.0  # y += vy
        self._motion_mat[2, 6] = 1.0  # a += va
        self._motion_mat[3, 7] = 1.0  # h += vh

        # Measurement matrix (4x8) - we observe only position, not velocity
        self._update_mat = torch.zeros(4, 8, device=self.device, dtype=torch.float32)
        self._update_mat[0, 0] = 1.0  # x
        self._update_mat[1, 1] = 1.0  # y
        self._update_mat[2, 2] = 1.0  # a
        self._update_mat[3, 3] = 1.0  # h

        # Uncertainty weights (same as original)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

        # Chi-square thresholds for gating (95% confidence)
        self.chi2inv95 = {
            1: 3.8415,
            2: 5.9915,
            3: 7.8147,
            4: 9.4877,
            5: 11.070,
            6: 12.592,
            7: 14.067,
            8: 15.507,
            9: 16.919
        }

    def initiate(self, measurement: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize a single track from an unassociated measurement.

        Args:
            measurement: [4] tensor with [x, y, a, h]

        Returns:
            mean: [8] state vector
            covariance: [8, 8] covariance matrix
        """
        mean_pos = measurement
        mean_vel = torch.zeros(4, device=self.device, dtype=torch.float32)
        mean = torch.cat([mean_pos, mean_vel])

        # Initial covariance
        std = torch.tensor([
            2 * self._std_weight_position * measurement[3],   # x
            2 * self._std_weight_position * measurement[3],   # y
            1e-2,                                              # a
            2 * self._std_weight_position * measurement[3],   # h
            10 * self._std_weight_velocity * measurement[3],  # vx
            10 * self._std_weight_velocity * measurement[3],  # vy
            1e-5,                                              # va
            10 * self._std_weight_velocity * measurement[3],  # vh
        ], device=self.device, dtype=torch.float32)

        covariance = torch.diag(std ** 2)
        return mean, covariance

    def batch_initiate(self, measurements: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize multiple tracks from measurements in a batch.

        Args:
            measurements: [N, 4] tensor with [x, y, a, h] for each detection

        Returns:
            means: [N, 8] state vectors
            covariances: [N, 8, 8] covariance matrices
        """
        N = measurements.shape[0]
        if N == 0:
            return (
                torch.empty(0, 8, device=self.device, dtype=torch.float32),
                torch.empty(0, 8, 8, device=self.device, dtype=torch.float32)
            )

        # Position is the measurement, velocity is zero
        mean_vel = torch.zeros(N, 4, device=self.device, dtype=torch.float32)
        means = torch.cat([measurements, mean_vel], dim=1)  # [N, 8]

        # Height for std computation
        h = measurements[:, 3:4]  # [N, 1]

        # Standard deviations
        std = torch.cat([
            2 * self._std_weight_position * h,   # x
            2 * self._std_weight_position * h,   # y
            torch.full((N, 1), 1e-2, device=self.device),   # a
            2 * self._std_weight_position * h,   # h
            10 * self._std_weight_velocity * h,  # vx
            10 * self._std_weight_velocity * h,  # vy
            torch.full((N, 1), 1e-5, device=self.device),   # va
            10 * self._std_weight_velocity * h,  # vh
        ], dim=1)  # [N, 8]

        # Diagonal covariance matrices
        covariances = torch.diag_embed(std ** 2)  # [N, 8, 8]

        return means, covariances

    def predict(self, mean: torch.Tensor, covariance: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single track prediction step.

        Args:
            mean: [8] state vector
            covariance: [8, 8] covariance matrix

        Returns:
            predicted mean: [8]
            predicted covariance: [8, 8]
        """
        # Motion noise covariance
        h = mean[3]
        std = torch.tensor([
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2,
            self._std_weight_position * h,
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            1e-5,
            self._std_weight_velocity * h,
        ], device=self.device, dtype=torch.float32)
        motion_cov = torch.diag(std ** 2)

        # Predict
        new_mean = self._motion_mat @ mean
        new_covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov

        # Covariance regularization: prevents overflow on long-lost tracks
        MAX_COVARIANCE = 1e5
        MIN_EIGENVALUE = 1e-4

        # Clamp diagonal elements
        diag = torch.diagonal(new_covariance)
        diag.clamp_(min=MIN_EIGENVALUE, max=MAX_COVARIANCE)

        # Ensure symmetry
        new_covariance = (new_covariance + new_covariance.T) / 2

        # Also clamp mean to prevent aspect ratio explosion
        # mean[2] = aspect ratio, should be in reasonable range
        new_mean[2].clamp_(min=0.1, max=10.0)
        # mean[3] = height, should be positive and reasonable
        new_mean[3].clamp_(min=1.0, max=1e4)

        return new_mean, new_covariance

    def batch_predict(
        self,
        means: torch.Tensor,
        covariances: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched prediction for multiple tracks.

        Args:
            means: [N, 8] state vectors
            covariances: [N, 8, 8] covariance matrices

        Returns:
            predicted means: [N, 8]
            predicted covariances: [N, 8, 8]
        """
        N = means.shape[0]
        if N == 0:
            return means, covariances

        # Motion noise (depends on height of each track)
        h = means[:, 3:4]  # [N, 1]

        std = torch.cat([
            self._std_weight_position * h,
            self._std_weight_position * h,
            torch.full((N, 1), 1e-2, device=self.device),
            self._std_weight_position * h,
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            torch.full((N, 1), 1e-5, device=self.device),
            self._std_weight_velocity * h,
        ], dim=1)  # [N, 8]

        # Diagonal motion noise covariance
        motion_cov = torch.diag_embed(std ** 2)  # [N, 8, 8]

        # Batched state transition: new_mean = motion_mat @ mean
        # [N, 8] = [N, 8] @ [8, 8].T
        new_means = means @ self._motion_mat.T

        # Batched covariance update:
        # new_cov = F @ cov @ F.T + Q
        # [N, 8, 8] = [8, 8] @ [N, 8, 8] @ [8, 8].T + [N, 8, 8]
        F = self._motion_mat  # [8, 8]
        # F @ cov: [N, 8, 8]
        temp = torch.einsum('ij,njk->nik', F, covariances)  # [N, 8, 8]
        # (F @ cov) @ F.T: [N, 8, 8]
        new_covariances = torch.einsum('nij,kj->nik', temp, F) + motion_cov  # [N, 8, 8]

        # Covariance regularization: prevents exponential growth of diagonal
        # elements (overflow protection on long-lost tracks)
        MAX_COVARIANCE = 1e5
        MIN_EIGENVALUE = 1e-4  # Ensure positive definiteness

        # Clamp diagonal elements to prevent overflow
        diag_indices = torch.arange(8, device=self.device)
        new_covariances[:, diag_indices, diag_indices] = torch.clamp(
            new_covariances[:, diag_indices, diag_indices],
            min=MIN_EIGENVALUE,
            max=MAX_COVARIANCE
        )

        # Ensure symmetry (numerical errors can break it)
        new_covariances = (new_covariances + new_covariances.transpose(1, 2)) / 2

        # Also clamp mean to prevent aspect ratio/height explosion
        # mean[:, 2] = aspect ratio, should be in reasonable range
        new_means[:, 2].clamp_(min=0.1, max=10.0)
        # mean[:, 3] = height, should be positive and reasonable
        new_means[:, 3].clamp_(min=1.0, max=1e4)

        return new_means, new_covariances

    def project(
        self,
        mean: torch.Tensor,
        covariance: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project state to measurement space.

        Args:
            mean: [8] or [N, 8] state vector(s)
            covariance: [8, 8] or [N, 8, 8] covariance(s)

        Returns:
            projected mean: [4] or [N, 4]
            projected covariance: [4, 4] or [N, 4, 4]
        """
        is_batch = mean.dim() == 2

        if is_batch:
            N = mean.shape[0]
            h = mean[:, 3:4]
            std = torch.cat([
                self._std_weight_position * h,
                self._std_weight_position * h,
                torch.full((N, 1), 1e-1, device=self.device),
                self._std_weight_position * h,
            ], dim=1)  # [N, 4]
            innovation_cov = torch.diag_embed(std ** 2)  # [N, 4, 4]

            # Project mean: H @ mean
            proj_mean = mean @ self._update_mat.T  # [N, 4]

            # Project covariance: H @ cov @ H.T + R
            H = self._update_mat  # [4, 8]
            temp = torch.einsum('ij,njk->nik', H, covariance)  # [N, 4, 8]
            proj_cov = torch.einsum('nij,kj->nik', temp, H) + innovation_cov  # [N, 4, 4]
        else:
            h = mean[3]
            std = torch.tensor([
                self._std_weight_position * h,
                self._std_weight_position * h,
                1e-1,
                self._std_weight_position * h,
            ], device=self.device, dtype=torch.float32)
            innovation_cov = torch.diag(std ** 2)

            proj_mean = self._update_mat @ mean
            proj_cov = self._update_mat @ covariance @ self._update_mat.T + innovation_cov

        return proj_mean, proj_cov

    def update(
        self,
        mean: torch.Tensor,
        covariance: torch.Tensor,
        measurement: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single track update step (measurement correction).

        Args:
            mean: [8] predicted state
            covariance: [8, 8] predicted covariance
            measurement: [4] observation [x, y, a, h]

        Returns:
            updated mean: [8]
            updated covariance: [8, 8]
        """
        proj_mean, proj_cov = self.project(mean, covariance)

        # Kalman gain: K = P @ H.T @ inv(S)
        # Where P = covariance [8, 8], H = update_mat [4, 8], S = proj_cov [4, 4]
        # P @ H.T = [8, 4]
        cov_HT = covariance @ self._update_mat.T  # [8, 4]

        # Solve S @ K.T = (P @ H.T).T to get K.T, then transpose
        # K = (inv(S) @ (P @ H.T).T).T = P @ H.T @ inv(S)
        kalman_gain = torch.linalg.solve(proj_cov, cov_HT.T).T  # [8, 4]

        # Innovation (measurement residual)
        innovation = measurement - proj_mean  # [4]

        # Update
        new_mean = mean + kalman_gain @ innovation  # [8]
        new_covariance = covariance - kalman_gain @ proj_cov @ kalman_gain.T  # [8, 8]

        return new_mean, new_covariance

    def batch_update(
        self,
        means: torch.Tensor,
        covariances: torch.Tensor,
        measurements: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched update for multiple tracks.

        Args:
            means: [N, 8] predicted states
            covariances: [N, 8, 8] predicted covariances
            measurements: [N, 4] observations

        Returns:
            updated means: [N, 8]
            updated covariances: [N, 8, 8]
        """
        N = means.shape[0]
        if N == 0:
            return means, covariances

        proj_means, proj_covs = self.project(means, covariances)  # [N, 4], [N, 4, 4]

        # Kalman gain: K = cov @ H.T @ inv(S)
        # cov @ H.T: [N, 8, 4]
        H = self._update_mat  # [4, 8]
        cov_HT = torch.einsum('nij,kj->nik', covariances, H)  # [N, 8, 4]

        # Solve for Kalman gain: S @ K.T = (cov @ H.T).T
        # K = (inv(S) @ (cov @ H.T).T).T
        kalman_gains = torch.linalg.solve(proj_covs, cov_HT.transpose(1, 2)).transpose(1, 2)  # [N, 8, 4]

        # Innovation
        innovations = measurements - proj_means  # [N, 4]

        # Update mean: mean + K @ innovation
        new_means = means + torch.einsum('nij,nj->ni', kalman_gains, innovations)  # [N, 8]

        # Update covariance: cov - K @ S @ K.T
        # K @ S: [N, 8, 4]
        KS = torch.einsum('nij,njk->nik', kalman_gains, proj_covs)  # [N, 8, 4]
        # K @ S @ K.T: [N, 8, 8]
        KSKT = torch.einsum('nij,nkj->nik', KS, kalman_gains)  # [N, 8, 8]
        new_covariances = covariances - KSKT  # [N, 8, 8]

        return new_means, new_covariances

    def gating_distance(
        self,
        mean: torch.Tensor,
        covariance: torch.Tensor,
        measurements: torch.Tensor,
        only_position: bool = False
    ) -> torch.Tensor:
        """
        Compute Mahalanobis distance between state and measurements.

        Args:
            mean: [8] or [N, 8] state(s)
            covariance: [8, 8] or [N, 8, 8] covariance(s)
            measurements: [M, 4] observations
            only_position: if True, only use x, y for distance

        Returns:
            distances: [M] or [N, M] Mahalanobis distances
        """
        proj_mean, proj_cov = self.project(mean, covariance)

        if only_position:
            proj_mean = proj_mean[..., :2]
            proj_cov = proj_cov[..., :2, :2]
            measurements = measurements[:, :2]

        is_batch = mean.dim() == 2

        if is_batch:
            # [N, M, d]
            diff = measurements.unsqueeze(0) - proj_mean.unsqueeze(1)
            # Solve: proj_cov @ x = diff.T for each track
            # [N, d, M] = solve([N, d, d], [N, d, M])
            chol = torch.linalg.cholesky(proj_cov)
            z = torch.linalg.solve_triangular(
                chol, diff.transpose(1, 2), upper=False
            )  # [N, d, M]
            squared_maha = (z ** 2).sum(dim=1)  # [N, M]
        else:
            # [M, d]
            diff = measurements - proj_mean
            chol = torch.linalg.cholesky(proj_cov)
            z = torch.linalg.solve_triangular(
                chol, diff.T, upper=False
            )  # [d, M]
            squared_maha = (z ** 2).sum(dim=0)  # [M]

        return squared_maha

    def batch_gating_distance(
        self,
        means: torch.Tensor,
        covariances: torch.Tensor,
        measurements: torch.Tensor,
        only_position: bool = False
    ) -> torch.Tensor:
        """
        Compute pairwise Mahalanobis distances between tracks and detections.

        Args:
            means: [N, 8] track states
            covariances: [N, 8, 8] track covariances
            measurements: [M, 4] detections
            only_position: if True, only use x, y

        Returns:
            distances: [N, M] Mahalanobis distances
        """
        return self.gating_distance(means, covariances, measurements, only_position)

    def to(self, device: str):
        """Move filter matrices to specified device."""
        self.device = torch.device(device)
        self._motion_mat = self._motion_mat.to(self.device)
        self._update_mat = self._update_mat.to(self.device)
        return self
