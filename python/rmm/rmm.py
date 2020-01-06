# Copyright (c) 2019, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import ctypes
import logging
import sys
from collections import deque
from enum import IntEnum

import numpy as np
from numba import cuda
from numba.cuda.cudadrv.memory import HostOnlyCUDAMemoryManager, MemoryPointer
from numba.utils import UniqueDict, logger_hasHandlers
import rmm._lib as librmm


# Utility Functions
class RMMError(Exception):
    def __init__(self, errcode, msg):
        self.errcode = errcode
        super(RMMError, self).__init__(msg)


class rmm_allocation_mode(IntEnum):
    CudaDefaultAllocation = (0,)
    PoolAllocation = (1,)
    CudaManagedMemory = (2,)


# API Functions
def _initialize(
    pool_allocator=False,
    managed_memory=False,
    initial_pool_size=None,
    devices=0,
    logging=False,
):
    """
    Initializes RMM library using the options passed
    """
    allocation_mode = 0

    if pool_allocator:
        allocation_mode |= rmm_allocation_mode.PoolAllocation
    if managed_memory:
        allocation_mode |= rmm_allocation_mode.CudaManagedMemory

    if not pool_allocator:
        initial_pool_size = 0
    elif pool_allocator and initial_pool_size is None:
        initial_pool_size = 0
    elif pool_allocator and initial_pool_size == 0:
        initial_pool_size = 1

    if devices is None:
        devices = [0]
    elif isinstance(devices, int):
        devices = [devices]

    return librmm.rmm_initialize(
        allocation_mode, initial_pool_size, devices, logging
    )


def _finalize():
    """
    Finalizes the RMM library, freeing all allocated memory
    """
    return librmm.rmm_finalize()


def reinitialize(
    pool_allocator=False,
    managed_memory=False,
    initial_pool_size=None,
    devices=0,
    logging=False,
):
    """
    Finalizes and then initializes RMM using the options passed. Using memory
    from a previous initialization of RMM is undefined behavior and should be
    avoided.

    Parameters
    ----------
    pool_allocator : bool, default False
        If True, use a pool allocation strategy which can greatly improve
        performance.
    managed_memory : bool, default False
        If True, use managed memory for device memory allocation
    initial_pool_size : int, default None
        When `pool_allocator` is True, this indicates the initial pool size in
        bytes. None is used to indicate the default size of the underlying
        memorypool implementation, which currently is 1/2 total GPU memory.
    devices : int or List[int], default 0
        GPU device  IDs to register. By default registers only GPU 0.
    logging : bool, default False
        If True, enable run-time logging of all memory events
        (alloc, free, realloc).
        This has significant performance impact.
    """
    _finalize()
    return _initialize(
        pool_allocator=pool_allocator,
        managed_memory=managed_memory,
        initial_pool_size=initial_pool_size,
        devices=devices,
        logging=logging,
    )


def is_initialized():
    """
    Returns true if RMM has been initialized, false otherwise
    """
    return librmm.rmm_is_initialized()


def csv_log():
    """
    Returns a CSV log of all events logged by RMM, if logging is enabled
    """
    return librmm.rmm_csv_log()


def get_ipc_handle(ary, stream=0):
    """
    Get an IPC handle from the DeviceArray ary with offset modified by
    the RMM memory pool.
    """
    ipch = cuda.devices.get_context().get_ipc_handle(ary.gpu_data)
    ptr = ary.device_ctypes_pointer.value
    offset = librmm.rmm_getallocationoffset(ptr, stream)
    # replace offset with RMM's offset
    ipch.offset = offset
    desc = dict(shape=ary.shape, strides=ary.strides, dtype=ary.dtype)
    return cuda.cudadrv.devicearray.IpcArrayHandle(
        ipc_handle=ipch, array_desc=desc
    )


def get_info(stream=0):
    """
    Get the free and total bytes of memory managed by a manager associated with
    the stream as a namedtuple with members `free` and `total`.
    """
    return librmm.rmm_getinfo(stream)


class RMMNumbaManager(HostOnlyCUDAMemoryManager):
    def __init__(self):
        self.allocations = UniqueDict()
        # Deallocations lazily initialised when context activated
        self.deallocations = None

    def memalloc(self, bytesize):
        # XXX: Need to support stream being passed in
        stream = 0
        addr = librmm.rmm_alloc(bytesize, stream)
        ctx = cuda.current_context()
        ptr = ctypes.c_uint64(int(addr))
        finalizer = _make_finalizer(addr, stream)
        mem = MemoryPointer(ctx, ptr, bytesize, finalizer=finalizer)
        #print(csv_log())
        return mem

    def prepare_for_use(self, memory_info):
        if self.deallocations is None:
            reinitialize(logging=True)
            free, total = get_info()
            self.deallocations = _PendingDeallocs(total)


def _make_logger():
    logger = logging.getLogger(__name__)
    # is logging configured?
    if not logger_hasHandlers(logger):
        # read user config
        # lvl = '' # str(config.CUDA_LOG_LEVEL).upper()
        # lvl = getattr(logging, lvl, None)
        lvl = logging.CRITICAL
        if not isinstance(lvl, int):
            # default to critical level
            lvl = logging.CRITICAL
        logger.setLevel(lvl)
        # did user specify a level?
        if True:
            # create a simple handler that prints to stderr
            handler = logging.StreamHandler(sys.stderr)
            fmt = '== CUDA [%(relativeCreated)d] %(levelname)5s -- %(message)s'
            handler.setFormatter(logging.Formatter(fmt=fmt))
            logger.addHandler(handler)
        else:
            # otherwise, put a null handler
            logger.addHandler(logging.NullHandler())
    return logger


_logger = _make_logger()


class _SizeNotSet(object):
    """
    Dummy object for _PendingDeallocs when *size* is not set.
    """
    def __str__(self):
        return '?'

    def __int__(self):
        return 0


_SizeNotSet = _SizeNotSet()


class _PendingDeallocs(object):
    """
    Pending deallocations of a context (or device since we are using the
    primary context).
    """
    def __init__(self, capacity):
        self._cons = deque()
        self._disable_count = 0
        self._size = 0
        self._memory_capacity = capacity

    @property
    def _max_pending_bytes(self):
        return int(self._memory_capacity * 0.5)
        # return int(self._memory_capacity * config.CUDA_DEALLOCS_RATIO)

    def add_item(self, dtor, handle, size=_SizeNotSet):
        """
        Add a pending deallocation.

        The *dtor* arg is the destructor function that takes an argument,
        *handle*.  It is used as ``dtor(handle)``.  The *size* arg is the byte
        size of the resource added.  It is an optional argument.  Some
        resources (e.g. CUModule) has an unknown memory footprint on the
        device.
        """
        _logger.info('add pending dealloc: %s %s bytes', dtor.__name__, size)
        self._cons.append((dtor, handle, size))
        self._size += int(size)
        if self._size > self._max_pending_bytes:
            self.clear()

    def clear(self):
        """
        Flush any pending deallocations unless it is disabled.
        Do nothing if disabled.
        """
        if not self.is_disabled:
            while self._cons:
                [dtor, handle, size] = self._cons.popleft()
                _logger.info('dealloc: %s %s bytes', dtor.__name__, size)
                dtor(handle)
            self._size = 0

    @contextlib.contextmanager
    def disable(self):
        """
        Context manager to temporarily disable flushing pending deallocation.
        This can be nested.
        """
        self._disable_count += 1
        try:
            yield
        finally:
            self._disable_count -= 1
            assert self._disable_count >= 0

    @property
    def is_disabled(self):
        return self._disable_count > 0

    def __len__(self):
        """
        Returns number of pending deallocations.
        """
        return len(self._cons)


def use_rmm_for_numba():
    cuda.cudadrv.driver.set_memory_manager(RMMNumbaManager)


try:
    import cupy
except Exception:
    cupy = None


def rmm_cupy_allocator(nbytes):
    """
    A CuPy allocator that make use of RMM.

    Examples
    --------
    >>> import rmm
    >>> import cupy
    >>> cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)
    """
    if cupy is None:
        raise ModuleNotFoundError("No module named 'cupy'")

    buf = librmm.device_buffer.DeviceBuffer(size=nbytes)
    dev_id = -1 if buf.ptr else cupy.cuda.device.get_device_id()
    mem = cupy.cuda.UnownedMemory(
        ptr=buf.ptr, size=buf.size, owner=buf, device_id=dev_id
    )
    ptr = cupy.cuda.memory.MemoryPointer(mem, 0)

    return ptr


def _make_finalizer(handle, stream):
    """
    Factory to make the finalizer function.
    We need to bind *handle* and *stream* into the actual finalizer, which
    takes no args.
    """

    def finalizer():
        """
        Invoked when the MemoryPointer is freed
        """
        librmm.rmm_free(handle, stream)
        #print(csv_log())

    return finalizer


def _register_atexit_finalize():
    """
    Registers rmmFinalize() with ``std::atexit``.
    """
    librmm.register_atexit_finalize()
