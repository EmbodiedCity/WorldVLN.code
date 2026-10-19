"""Offload saved activations without copying FSDP parameter views.

FSDP owns the lifetime of its unsharded parameter storage and restores it in
pre-backward. Copying every saved tensor with ``save_on_cpu`` also copies a
Linear's weight on every autoregressive invocation. Keep those views under
FSDP's normal lifetime management and copy only activation values.
"""

from contextlib import AbstractContextManager
import weakref

import torch
from torch import nn


class _PackedTensor:
    def __init__(self, tensor, original_device, offloaded):
        self.tensor = tensor
        self.original_device = original_device
        self.offloaded = offloaded


class selective_save_on_cpu(AbstractContextManager):
    """Exact saved-tensor offload for a model containing FSDP units.

    The context only needs to enclose the forward: PyTorch retains each saved
    value's unpack hook until backward. ``stats`` counts copies, so
    ``offloaded_bytes`` is an upper bound on the context's CPU saved values.
    ``offload_device_types`` is configurable for CPU regression tests; normal
    training should use its CUDA default.

    Storage identity uses StorageImpl identity (``_cdata``), never data_ptr:
    CUDA's allocator may recycle an allocation's pointer for an activation.
    No historical storage IDs are cached, avoiding identity reuse as well.
    """

    _flat_storage_attributes = (
        "_full_param_padded", "_full_prec_full_param_padded", "_mp_shard",
        "_local_shard",
    )

    def __init__(self, model, pin_memory=False, offload_device_types=("cuda",)):
        self.pin_memory = bool(pin_memory)
        self.offload_device_types = frozenset(offload_device_types)
        self._handles = tuple(
            handle for module in model.modules()
            if (handle := getattr(module, "_handle", None)) is not None
            and getattr(handle, "flat_param", None) is not None
        )
        self._dedup = {}
        self._hooks = None
        self.stats = dict(
            parameter_view_tensors=0, parameter_view_bytes=0,
            offloaded_tensors=0, offloaded_bytes=0,
            deduplicated_tensors=0, deduplicated_bytes=0,
            passthrough_tensors=0, unpacked_tensors=0, unpacked_bytes=0,
        )

    @staticmethod
    def _storage_identity(tensor):
        if tensor.layout != torch.strided:
            return None
        return tensor.untyped_storage()._cdata

    def _is_parameter_view(self, tensor):
        # Ordinary Parameters and FSDP's usual split/view/transpose chains.
        base = tensor
        while base is not None:
            if isinstance(base, nn.Parameter):
                return True
            base = getattr(base, "_base", None)

        storage_id = self._storage_identity(tensor)
        if storage_id is None:
            return False
        # FSDP can install tensors without a Parameter _base. Inspect *current*
        # handle storages at pack time; full buffers are lazily initialized.
        for handle in self._handles:
            flat = handle.flat_param
            candidates = (flat,) + tuple(
                getattr(flat, name, None) for name in self._flat_storage_attributes
            )
            for candidate in candidates:
                if isinstance(candidate, torch.Tensor) and self._storage_identity(candidate) == storage_id:
                    return True
        return False

    def _pack(self, tensor):
        size = tensor.numel() * tensor.element_size()
        if self._is_parameter_view(tensor):
            self.stats["parameter_view_tensors"] += 1
            self.stats["parameter_view_bytes"] += size
            # detach avoids a reference cycle with the original graph. It
            # shares the original storage; autograd still differentiates the
            # original operation, using this tensor only as a saved value.
            return _PackedTensor(tensor.detach(), tensor.device, False)
        if tensor.device.type not in self.offload_device_types:
            self.stats["passthrough_tensors"] += 1
            return _PackedTensor(tensor.detach(), tensor.device, False)
        if tensor.layout != torch.strided:
            raise RuntimeError("selective saved-tensor offload requires strided activations")

        # Repeated saving of the same live Tensor (e.g. a nonlinear output and
        # the following Linear input) needs just one CPU copy. Only weak
        # references to the source are kept, so this does not retain GPU
        # activations or confuse a recycled Python id with the old Tensor.
        cache_key = id(tensor)
        version = tensor._version
        cached = self._dedup.get(cache_key)
        if cached is not None:
            source_ref, saved_version, packed_ref = cached
            packed = packed_ref()
            if source_ref() is tensor and version == saved_version and packed is not None:
                self.stats["deduplicated_tensors"] += 1
                self.stats["deduplicated_bytes"] += size
                return packed
        pin = self.pin_memory and tensor.device.type != "cpu"
        cpu = torch.empty(tensor.size(), dtype=tensor.dtype, layout=tensor.layout,
                          device="cpu", pin_memory=pin)
        # Blocking D2H, as in torch.save_on_cpu: the copy is complete before
        # CUDA may reuse the source activation's allocation.
        cpu.copy_(tensor.detach())
        packed = _PackedTensor(cpu, tensor.device, True)
        self._dedup[cache_key] = (weakref.ref(tensor), version, weakref.ref(packed))
        self.stats["offloaded_tensors"] += 1
        self.stats["offloaded_bytes"] += size
        return packed

    def _unpack(self, packed):
        if not packed.offloaded:
            return packed.tensor
        self.stats["unpacked_tensors"] += 1
        self.stats["unpacked_bytes"] += packed.tensor.numel() * packed.tensor.element_size()
        return packed.tensor.to(packed.original_device, non_blocking=self.pin_memory)

    def __enter__(self):
        if self._hooks is not None:
            raise RuntimeError("selective_save_on_cpu contexts cannot be reentered")
        self._hooks = torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack)
        self._hooks.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return self._hooks.__exit__(exc_type, exc_value, traceback)
        finally:
            self._dedup.clear()
