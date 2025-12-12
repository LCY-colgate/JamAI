from dataclasses import dataclass
from typing import Any, Dict, Sequence
import prompt_types

@dataclass
class StaticConfig:
    params: Dict[str, Any]

    def get_values(self) -> Dict[str, Any]:
        return dict(self.params)

    def set(self, key: str, value: Any):
        self.params[key] = value


def make_default_config(
    initial_prompts: Sequence[str],
    beats_per_loop: int,
    use_prerecorded_input: bool,
    num_audio_prompts: int = 1,
) -> "StaticConfig":
    
    """
    - build_sampling_option_ui()
    - build_hidden_option_ui()
    - build_prompt_ui()
    """
    params: Dict[str, Any] = {}

    # === build_sampling_option_ui() ===
    params["temperature"] = 1.2
    params["topk"] = 30
    params["guidance_weight"] = 1.5 if use_prerecorded_input else 0.8

    if use_prerecorded_input:
        params["model_volume"] = 0.0
        params["input_volume"] = 1.0
        params["metronome"] = False
    else:
        params["metronome"] = True
        params["model_volume"] = 0.0
        params["input_volume"] = 0.0

    # === build_hidden_option_ui() ===
    params["input_gap"] = 0               
    params["beats_per_loop"] = beats_per_loop
    params["model_feedback"] = 0.0    
        
    # === build_prompt_ui() ===
    idx = 0

    # text prompts
    for j, text in enumerate(initial_prompts):
        params[f"prompt_value_{idx}"] = prompt_types.TextPrompt(text)
        params[f"prompt_weight_{idx}"] = 1.0 if j == 0 else 0.0
        idx += 1
    # audio prompts
    for _ in range(num_audio_prompts):
        params[f"prompt_value_{idx}"] = None 
        params[f"prompt_weight_{idx}"] = 0.0
        idx += 1
    
    return StaticConfig(params=params)
        
