"""Pick the GPU that headless rendering (MuJoCo over EGL) runs on.

EGL numbers devices in its own order, which differs from nvidia-smi's (on some multi-GPU nodes EGL device i is
GPU (i + 4) % 8), and MuJoCo takes EGL device 0 unless MUJOCO_EGL_DEVICE_ID says otherwise. So without this, a
renderer can land on a GPU that is busy serving models; on a full card it returns garbage frames.
"""
from __future__ import annotations

import ctypes
import os

EGL_CUDA_DEVICE_NV = 0x323A


def egl_index_of_gpu(gpu: int) -> int:
    """The EGL device index of the GPU with CUDA ordinal (= nvidia-smi index) gpu."""
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from mujoco.egl import egl_ext as E
    from OpenGL import EGL
    query = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_long))(
        EGL.eglGetProcAddress("eglQueryDeviceAttribEXT"))
    for i, dev in enumerate(E.eglQueryDevicesEXT()):
        ptr = dev if isinstance(dev, int) else ctypes.cast(dev, ctypes.c_void_p).value
        value = ctypes.c_long(-1)
        if query(ptr, EGL_CUDA_DEVICE_NV, ctypes.byref(value)) and value.value == gpu:
            return i
    raise RuntimeError(f"no EGL device for GPU {gpu}")


def use_gpu(gpu: int | None) -> None:
    """Render on GPU `gpu` (nvidia-smi index). Call before the first renderer is created."""
    if gpu is None:
        return
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_index_of_gpu(int(gpu)))
