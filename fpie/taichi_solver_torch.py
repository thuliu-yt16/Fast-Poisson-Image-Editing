from typing import Tuple
import torch

import numpy as np
import taichi as ti


@ti.data_oriented
class EquSolver(object):
  """Taichi-based Jacobi method equation solver implementation."""

  def __init__(self, backend: str, n_cpu: int, block_size: int) -> None:
    super().__init__()
    self.parallelize = n_cpu
    self.block_dim = block_size
    ti.init(arch=getattr(ti, backend.split("-")[-1]))
    self.N = 0
    self.fb: ti.FieldsBuilder
    self.fbst: ti._snode.snode_tree.SNodeTree
    self.terr = ti.field(ti.f32, (3,))
    self.tA = ti.field(ti.i32)
    self.tB = ti.field(ti.f32)
    self.tX = ti.field(ti.f32)
    self.tmp = ti.field(ti.f32)
    self.pyt_device = 'cuda'

  def partition(self, mask: torch.Tensor) -> torch.Tensor:
    return torch.cumsum((mask > 0).reshape(-1), dim=0).reshape(mask.shape)
  
  def reset(self, N:int, A: torch.Tensor, X: torch.Tensor, B: torch.Tensor) -> None:
    """(4 - A)X = B"""
    self.N = N
    self.A = A
    self.B = B
    self.X = X
    if hasattr(self, "fbst"):
      self.fbst.destroy()
      self.tA = ti.field(ti.i32)
      self.tB = ti.field(ti.f32)
      self.tX = ti.field(ti.f32)
      self.tmp = ti.field(ti.f32)

    self.fb = ti.FieldsBuilder()
    self.fb.dense(ti.ij, (N, 4)).place(self.tA)
    self.fb.dense(ti.ij, (N, 3)).place(self.tB)
    self.fb.dense(ti.ij, (N, 3)).place(self.tX)
    self.fb.dense(ti.ij, (N, 3)).place(self.tmp)
    self.fbst = self.fb.finalize()
    self.tA.from_torch(A)
    self.tB.from_torch(B)
    self.tX.from_torch(X)
    self.tmp.from_torch(X)
    self.pyt_device = X.device

  def sync(self) -> None:
    pass

  @ti.kernel
  def iter_kernel(self) -> int:
    ti.loop_config(parallelize=self.parallelize, block_dim=self.block_dim)
    for i in range(1, self.N):
      i0, i1 = self.tA[i, 0], self.tA[i, 1]
      i2, i3 = self.tA[i, 2], self.tA[i, 3]
      # X = (B + AX) / 4
      self.tX[i, 0] = (
        self.tB[i, 0] + self.tX[i0, 0] + self.tX[i1, 0] + self.tX[i2, 0] +
        self.tX[i3, 0]
      ) / 4.0
      self.tX[i, 1] = (
        self.tB[i, 1] + self.tX[i0, 1] + self.tX[i1, 1] + self.tX[i2, 1] +
        self.tX[i3, 1]
      ) / 4.0
      self.tX[i, 2] = (
        self.tB[i, 2] + self.tX[i0, 2] + self.tX[i1, 2] + self.tX[i2, 2] +
        self.tX[i3, 2]
      ) / 4.0
    return 0

  @ti.kernel
  def error_kernel(self) -> int:
    ti.loop_config(parallelize=self.parallelize, block_dim=self.block_dim)
    for i in range(1, self.N):
      i0, i1 = self.tA[i, 0], self.tA[i, 1]
      i2, i3 = self.tA[i, 2], self.tA[i, 3]
      self.tmp[i, 0] = ti.abs(
        self.tB[i, 0] + self.tX[i0, 0] + self.tX[i1, 0] + self.tX[i2, 0] +
        self.tX[i3, 0] - 4.0 * self.tX[i, 0]
      )
      self.tmp[i, 1] = ti.abs(
        self.tB[i, 1] + self.tX[i0, 1] + self.tX[i1, 1] + self.tX[i2, 1] +
        self.tX[i3, 1] - 4.0 * self.tX[i, 1]
      )
      self.tmp[i, 2] = ti.abs(
        self.tB[i, 2] + self.tX[i0, 2] + self.tX[i1, 2] + self.tX[i2, 2] +
        self.tX[i3, 2] - 4.0 * self.tX[i, 2]
      )

    self.terr[0] = self.terr[1] = self.terr[2] = 0
    ti.loop_config(parallelize=self.parallelize, block_dim=self.block_dim)
    for i, j in self.tmp:
      self.terr[j] += self.tmp[i, j]

    return 0
  
  def step(self, iteration: int) -> Tuple[torch.Tensor, torch.Tensor]:
    for _ in range(iteration):
      self.iter_kernel()
    self.error_kernel()
    x = self.tX.to_torch(device=self.pyt_device)
    x[x < 0] = 0
    x[x > 255] = 255
    return x, self.terr.to_torch(device=self.pyt_device)


@ti.data_oriented
class GridSolver(object):
  """Taichi-based Jacobi method grid solver implementation."""

  def __init__(
    self, grid_x: int, grid_y: int, backend: str, n_cpu: int, block_size: int
  ) -> None:
    super().__init__()
    self.N = 0
    self.grid_x = grid_x
    self.grid_y = grid_y
    self.parallelize = n_cpu
    self.block_dim = block_size
    ti.init(arch=getattr(ti, backend.split("-")[-1]))
    self.fb: ti.FieldsBuilder
    self.fbst: ti._snode.snode_tree.SNodeTree
    self.terr = ti.field(ti.f32, (3,))
    self.tmask = ti.field(ti.i32)
    self.ttgt = ti.field(ti.f32)
    self.tgrad = ti.field(ti.f32)
    self.tmp = ti.field(ti.f32)
    self.pyt_device = 'cuda'

  def reset(
    self, N: int, mask: torch.Tensor, tgt: torch.Tensor, grad: torch.Tensor
  ) -> None:
    gx, gy = self.grid_x, self.grid_y
    self.orig_N, self.orig_M = N, M = mask.shape
    pad_x = 0 if N % gx == 0 else gx - (N % gx)
    pad_y = 0 if M % gy == 0 else gy - (M % gy)
    if pad_x or pad_y:
      mask = np.pad(mask, [(0, pad_x), (0, pad_y)])
      tgt = np.pad(tgt, [(0, pad_x), (0, pad_y), (0, 0)])
      grad = np.pad(grad, [(0, pad_x), (0, pad_y), (0, 0)])

    self.N, self.M = N, M = mask.shape
    bx, by = N // gx, M // gy
    self.mask = mask
    self.tgt = tgt
    self.grad = grad

    if hasattr(self, "fbst"):
      self.fbst.destroy()
      self.tmask = ti.field(ti.i32)
      self.ttgt = ti.field(ti.f32)
      self.tgrad = ti.field(ti.f32)
      self.tmp = ti.field(ti.f32)
    self.fb = ti.FieldsBuilder()
    self.fb.dense(ti.ij, (bx, by)).dense(ti.ij, (gx, gy)).place(self.tmask)
    self.fb.dense(ti.ij, (bx, by)).dense(ti.ij, (gx, gy)) \
      .dense(ti.k, 3).place(self.ttgt)
    self.fb.dense(ti.ij, (bx, by)).dense(ti.ij, (gx, gy)) \
      .dense(ti.k, 3).place(self.tgrad)
    self.fb.dense(ti.ij, (bx, by)).dense(ti.ij, (gx, gy)) \
      .dense(ti.k, 3).place(self.tmp)
    self.fbst = self.fb.finalize()
    self.tmask.from_torch(mask)
    self.ttgt.from_torch(tgt)
    self.tgrad.from_torch(grad)
    self.tmp.from_torch(grad)
    self.pyt_device = tgt.device

  def sync(self) -> None:
    pass

  @ti.kernel
  def iter_kernel(self) -> int:
    ti.loop_config(parallelize=self.parallelize, block_dim=self.block_dim)
    for i, j in self.tmask:
      if self.tmask[i, j] > 0:
        # tgt = (grad + Atgt) / 4
        self.ttgt[i, j, 0] = (
          self.tgrad[i, j, 0] + self.ttgt[i - 1, j, 0] + self.ttgt[i, j - 1, 0]
          + self.ttgt[i, j + 1, 0] + self.ttgt[i + 1, j, 0]
        ) / 4.0
        self.ttgt[i, j, 1] = (
          self.tgrad[i, j, 1] + self.ttgt[i - 1, j, 1] + self.ttgt[i, j - 1, 1]
          + self.ttgt[i, j + 1, 1] + self.ttgt[i + 1, j, 1]
        ) / 4.0
        self.ttgt[i, j, 2] = (
          self.tgrad[i, j, 2] + self.ttgt[i - 1, j, 2] + self.ttgt[i, j - 1, 2]
          + self.ttgt[i, j + 1, 2] + self.ttgt[i + 1, j, 2]
        ) / 4.0
    return 0

  @ti.kernel
  def error_kernel(self) -> int:
    ti.loop_config(parallelize=self.parallelize, block_dim=self.block_dim)
    for i, j in self.tmask:
      if self.tmask[i, j] > 0:
        self.tmp[i, j, 0] = ti.abs(
          self.tgrad[i, j, 0] + self.ttgt[i - 1, j, 0] +
          self.ttgt[i, j - 1, 0] + self.ttgt[i, j + 1, 0] +
          self.ttgt[i + 1, j, 0] - 4.0 * self.ttgt[i, j, 0]
        )
        self.tmp[i, j, 1] = ti.abs(
          self.tgrad[i, j, 1] + self.ttgt[i - 1, j, 1] +
          self.ttgt[i, j - 1, 1] + self.ttgt[i, j + 1, 1] +
          self.ttgt[i + 1, j, 1] - 4.0 * self.ttgt[i, j, 1]
        )
        self.tmp[i, j, 2] = ti.abs(
          self.tgrad[i, j, 2] + self.ttgt[i - 1, j, 2] +
          self.ttgt[i, j - 1, 2] + self.ttgt[i, j + 1, 2] +
          self.ttgt[i + 1, j, 2] - 4.0 * self.ttgt[i, j, 2]
        )
      else:
        self.tmp[i, j, 0] = self.tmp[i, j, 1] = self.tmp[i, j, 2] = 0.0

    self.terr[0] = self.terr[1] = self.terr[2] = 0
    ti.loop_config(parallelize=self.parallelize, block_dim=self.block_dim)
    for i, j, k in self.tmp:
      self.terr[k] += self.tmp[i, j, k]

    return 0

  def step(self, iteration: int) -> Tuple[np.ndarray, np.ndarray]:
    for _ in range(iteration):
      self.iter_kernel()
    self.error_kernel()

    tgt = self.ttgt.to_torch(device=self.pyt_device)[:self.orig_N, :self.orig_M]
    tgt[tgt < 0] = 0
    tgt[tgt > 255] = 255
    return tgt, self.terr.to_torch(device=self.pyt_device)
