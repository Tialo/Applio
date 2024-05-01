import os
import asyncio

from pydub import AudioSegment

from core import run_infer_script
from utils import DATA_DIR, MODELS_DIR, send_infer_file


def infer(user_id, model_name, infer_path):
    input_dir = os.path.join(DATA_DIR, "infer", str(user_id))
    audio = AudioSegment.from_ogg(os.path.join(input_dir, infer_path))
    audio.export(os.path.join(input_dir, infer_path.replace(".ogg", ".wav")), format="wav")
    run_infer_script(
        f0up_key=0,
        filter_radius=3,
        index_rate=0.75,
        rms_mix_rate=1,
        protect=0.5,
        hop_length=128,
        f0method="rmvpe",
        input_path=os.path.join(input_dir, infer_path),
        output_path=os.path.join(input_dir, infer_path.replace(".ogg", "out.wav")),
        pth_path=os.path.join(MODELS_DIR, f"{user_id}_{model_name}", "model.pth"),
        index_path=os.path.join(MODELS_DIR, f"{user_id}_{model_name}", "index.index"),
        split_audio=False,
        f0autotune=False,
        clean_audio=False,
        clean_strength=0.5,
        export_format="WAV"
    )
    asyncio.run(send_infer_file(user_id, os.path.join(input_dir, infer_path.replace(".ogg", ".wav"))))
    os.remove(os.path.join(input_dir, infer_path))
    os.remove(os.path.join(input_dir, infer_path.replace(".ogg", ".wav")))
