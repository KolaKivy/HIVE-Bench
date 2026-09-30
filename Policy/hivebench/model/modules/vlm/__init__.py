# hivebench/model/modules/vlm/__init__.py
def get_vlm_model(config):
    vlm_name = config.framework.qwenvl.base_vlm

    if "xiaomi-robotics" in vlm_name.lower() or "xiaomirobotics" in vlm_name.lower():
        from .XiaomiRobotics1 import _XiaomiRobotics1_Interface
        return _XiaomiRobotics1_Interface(config)

    elif "depthvlm" in vlm_name.lower():
        from .DepthVLM import _DepthVLM_Interface
        return _DepthVLM_Interface(config)

    elif "Qwen2.5-VL" in vlm_name or "nora" in vlm_name.lower():
        from .QWen2_5 import _QWen_VL_Interface
        return _QWen_VL_Interface(config)

    elif "Qwen3.5" in vlm_name:
        from .QWen3_5 import _QWen3_5_VL_Interface
        return _QWen3_5_VL_Interface(config)

    elif "Qwen3-VL" in vlm_name:
        from .QWen3 import _QWen3_VL_Interface
        return _QWen3_VL_Interface(config)

    elif "gemma-3" in vlm_name.lower() or "gemma3" in vlm_name.lower():
        from .Gemma3 import _Gemma3_VL_Interface
        return _Gemma3_VL_Interface(config)

    elif "smolvlm" in vlm_name.lower() or "smol" in vlm_name.lower():
        from .SmolVLM import _SmolVLM_Interface
        return _SmolVLM_Interface(config)

    elif "openvla" in vlm_name.lower():
        from .OpenVLA import _OpenVLA_Interface
        return _OpenVLA_Interface(config)

    elif "florence" in vlm_name.lower():
        from .Florence2 import _Florence_Interface
        return _Florence_Interface(config)

    elif "cosmos-reason2" in vlm_name.lower():
        from .CosmosReason2 import _CosmosReason2_Interface
        return _CosmosReason2_Interface(config)

    else:
        raise NotImplementedError(f"VLM model {vlm_name} not implemented")
