import os
import asyncio

from core import run_infer_script
from utils import DATA_DIR, MODELS_DIR, send_infer_file


def infer(user_id, model_name, infer_path, f0up):
    input_dir = os.path.join(DATA_DIR, "infer", str(user_id))
    run_infer_script(
        f0up_key=f0up,
        filter_radius=3,
        index_rate=0.75,
        rms_mix_rate=1,
        protect=0.5,
        hop_length=128,
        f0method="rmvpe",
        input_path=os.path.join(input_dir, infer_path),
        output_path=os.path.join(input_dir, infer_path.replace(".ogg", "_out.ogg")),
        pth_path=os.path.join(MODELS_DIR, model_name, "model.pth"),
        index_path=os.path.join(MODELS_DIR, model_name, "index.index"),
        split_audio=False,
        f0autotune=False,
        clean_audio=False,
        clean_strength=0.5,
        export_format="OGG"
    )
    asyncio.run(send_infer_file(user_id, os.path.join(input_dir, infer_path.replace(".ogg", "_out.ogg"))))
    os.remove(os.path.join(input_dir, infer_path))
    os.remove(os.path.join(input_dir, infer_path.replace(".ogg", "_out.ogg")))
