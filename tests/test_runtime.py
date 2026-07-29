import unittest
import warnings
from unittest import mock

import torch

from llm_controllability.models.runtime import (
    resolve_device,
    resolve_runtime,
)


class RuntimeResolutionTests(unittest.TestCase):
    @unittest.skipUnless(
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available(),
        "requires an Apple MPS device",
    )
    def test_live_mps_forward_and_backward(self):
        device = resolve_device("mps")
        values = torch.arange(
            12,
            dtype=torch.float32,
            device=device,
        ).reshape(3, 4)
        values.requires_grad_(True)

        loss = (values @ values.transpose(0, 1)).mean()
        loss.backward()

        self.assertEqual(values.device.type, "mps")
        self.assertIsNotNone(values.grad)
        self.assertTrue(torch.isfinite(values.grad).all().item())

    @mock.patch(
        "llm_controllability.models.runtime.torch.cuda.is_available",
        return_value=True,
    )
    def test_auto_preserves_accelerate_device_map_on_cuda(self, _cuda):
        runtime = resolve_runtime(
            device_map="auto",
            dtype="bfloat16",
            attention_implementation="flash_attention_2",
        )

        self.assertEqual(runtime.device, torch.device("cuda"))
        self.assertEqual(runtime.device_map, "auto")
        self.assertEqual(runtime.dtype, torch.bfloat16)
        self.assertEqual(runtime.attention_implementation, "flash_attention_2")

    @mock.patch(
        "llm_controllability.models.runtime.platform.mac_ver",
        return_value=("14.6", ("", "", ""), ""),
    )
    @mock.patch(
        "llm_controllability.models.runtime.mps_is_available",
        return_value=True,
    )
    @mock.patch(
        "llm_controllability.models.runtime.torch.cuda.is_available",
        return_value=False,
    )
    def test_auto_selects_mps_and_replaces_flash_attention(
        self,
        _cuda,
        _mps,
        _mac,
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            runtime = resolve_runtime(
                device_map="auto",
                dtype="bfloat16",
                attention_implementation="flash_attention_2",
            )

        self.assertTrue(any("flash_attention_2" in str(item.message) for item in caught))
        self.assertEqual(runtime.device, torch.device("mps"))
        self.assertEqual(runtime.device_map, "mps")
        self.assertEqual(runtime.dtype, torch.bfloat16)
        self.assertEqual(runtime.attention_implementation, "eager")

    @mock.patch(
        "llm_controllability.models.runtime.platform.mac_ver",
        return_value=("13.7", ("", "", ""), ""),
    )
    @mock.patch(
        "llm_controllability.models.runtime.mps_is_available",
        return_value=True,
    )
    @mock.patch(
        "llm_controllability.models.runtime.torch.cuda.is_available",
        return_value=False,
    )
    def test_mps_bfloat16_falls_back_on_older_macos(
        self,
        _cuda,
        _mps,
        _mac,
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            runtime = resolve_runtime(device_map="auto", dtype="bfloat16")

        self.assertTrue(any("macOS 14" in str(item.message) for item in caught))
        self.assertEqual(runtime.dtype, torch.float16)
        self.assertEqual(runtime.dtype_name, "float16")

    @mock.patch(
        "llm_controllability.models.runtime.mps_is_available",
        return_value=False,
    )
    @mock.patch(
        "llm_controllability.models.runtime.torch.cuda.is_available",
        return_value=False,
    )
    def test_cpu_float16_falls_back_to_float32(self, _cuda, _mps):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            runtime = resolve_runtime(device_map="auto", dtype="float16")

        self.assertTrue(any("CPU float16" in str(item.message) for item in caught))
        self.assertEqual(runtime.device, torch.device("cpu"))
        self.assertEqual(runtime.dtype, torch.float32)
        self.assertEqual(runtime.dtype_name, "float32")

    @mock.patch(
        "llm_controllability.models.runtime.mps_is_available",
        return_value=False,
    )
    @mock.patch(
        "llm_controllability.models.runtime.mps_is_built",
        return_value=True,
    )
    def test_explicit_unavailable_mps_is_rejected(self, _built, _available):
        with self.assertRaisesRegex(RuntimeError, "no compatible device"):
            resolve_device("mps")

if __name__ == "__main__":
    unittest.main()
