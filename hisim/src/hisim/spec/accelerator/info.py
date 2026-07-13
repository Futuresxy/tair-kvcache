from hisim.spec.accelerator.base import AcceleratorInfo


class NVIDIA:
    NVIDIA_A100_SXM_80GB = AcceleratorInfo.from_dict(
        config={
            "name": "NVIDIA A100 SXM 80GB",
            "device_alias": ["A100", "A100_SXM"],
            "tflops": {
                "INT8_TENSOR": 624,
                "FP16_TENSOR": 312,
                "BF16_TENSOR": 312,
                "FP32": 19.5,
            },
            "hbm_capacity_gb": 80,
            "hbm_bandwidth_gb": 2039,
            "inter_node_bandwidth_gb": 64,
            "intra_node_bandwidth_gb": 600,
            "vendor": "NVIDIA",
            "ref": "https://www.nvidia.com/en-us/data-center/a100/",
        },
        save_to_registry=True,
    )

    NVIDIA_H100_SXM_80GB = AcceleratorInfo.from_dict(
        config={
            "name": "NVIDIA H100 SXM 80GB",
            "device_alias": ["H100", "H100_SXM"],
            "tflops": {
                "FP8_TENSOR": 1978,
                "INT8_TENSOR": 1978,
                "FP16_TENSOR": 989,
                "BF16_TENSOR": 989,
                "FP32": 67,
            },
            "hbm_capacity_gb": 80,
            "hbm_bandwidth_gb": 3350,
            "inter_node_bandwidth_gb": 64,
            "intra_node_bandwidth_gb": 900,
            "vendor": "NVIDIA",
            "ref": "https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet",
        },
        save_to_registry=True,
    )

    NVIDIA_H200_SXM_141GB = AcceleratorInfo.from_dict(
        config={
            "name": "NVIDIA H200 SXM 141GB",
            "device_alias": ["H200", "H200_SXM"],
            "tflops": {
                "FP8_TENSOR": 1978,
                "INT8_TENSOR": 1978,
                "FP16_TENSOR": 989,
                "BF16_TENSOR": 989,
                "FP32": 67,
            },
            "hbm_capacity_gb": 141,
            "hbm_bandwidth_gb": 4800,
            "inter_node_bandwidth_gb": 64,
            "intra_node_bandwidth_gb": 900,
            "vendor": "NVIDIA",
            "ref": "https://www.nvidia.com/en-us/data-center/h200/",
        },
        save_to_registry=True,
    )

    NVIDIA_H20 = AcceleratorInfo.from_dict(
        config={
            "name": "NVIDIA H20",
            "device_alias": ["H20", "h20_sxm"],
            "tflops": {
                "FP8_TENSOR": 296,
                "INT8_TENSOR": 296,
                "FP16_TENSOR": 148,
                "BF16_TENSOR": 148,
                "FP32": 74,
            },
            "hbm_capacity_gb": 96,
            "hbm_bandwidth_gb": 4022,
            "inter_node_bandwidth_gb": 64,
            "intra_node_bandwidth_gb": 450,
            "vendor": "NVIDIA",
            "ref": "https://viperatech.com/product/nvidia-hgx-h20",
        },
        save_to_registry=True,
    )

    NVIDIA_RTX_4090 = AcceleratorInfo.from_dict(
        config={
            "name": "NVIDIA GeForce RTX 4090",
            "device_alias": ["RTX4090", "RTX_4090", "GeForce RTX 4090"],
            "tflops": {
                "INT8_TENSOR": 661,
                "FP16_TENSOR": 165,
                "BF16_TENSOR": 165,
                "FP32": 82.6,
            },
            "hbm_capacity_gb": 24,
            "hbm_bandwidth_gb": 1008,
            "inter_node_bandwidth_gb": 32,
            "intra_node_bandwidth_gb": 64,
            "vendor": "NVIDIA",
            "ref": "https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/",
        },
        save_to_registry=True,
    )


class Huawei:
    ASCEND_910B = AcceleratorInfo.from_dict(
        config={
            "name": "Huawei Ascend 910B",
            "device_alias": ["Ascend910B", "Ascend 910B", "910B"],
            "tflops": {
                "INT8_TENSOR": 640,
                "FP16_TENSOR": 320,
                "BF16_TENSOR": 320,
                "FP32": 80,
            },
            "hbm_capacity_gb": 64,
            "hbm_bandwidth_gb": 1200,
            "inter_node_bandwidth_gb": 25,
            "intra_node_bandwidth_gb": 392,
            "vendor": "Huawei",
            "ref": "https://arthurchiao.art/blog/gpu-data-sheets/",
        },
        save_to_registry=True,
    )

    ASCEND_950 = AcceleratorInfo.from_dict(
        config={
            "name": "Huawei Ascend 950",
            "device_alias": ["Ascend950", "Ascend 950", "950"],
            "tflops": {
                "FP4_TENSOR": 1560,
                "INT8_TENSOR": 1560,
                "FP16_TENSOR": 780,
                "BF16_TENSOR": 780,
                "FP32": 195,
            },
            "hbm_capacity_gb": 144,
            "hbm_bandwidth_gb": 4000,
            "inter_node_bandwidth_gb": 25,
            "intra_node_bandwidth_gb": 2000,
            "vendor": "Huawei",
            "ref": "https://www.huawei.com/en/news/2025/9/hc-xu-keynote-speech",
        },
        save_to_registry=True,
    )
