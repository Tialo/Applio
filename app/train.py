import os

from pydub import AudioSegment

from tabs.train.train import save_drop_dataset_audio
from core import run_preprocess_script, run_extract_script

from utils import db, DATA_DIR


def merge_wav(user_id, model_name):
    merged = AudioSegment.empty()
    wav_folder = os.path.join(DATA_DIR, str(user_id), model_name)
    for file in os.listdir(os.path.join(DATA_DIR, str(user_id), model_name)):
        if file == "merged.wav":
            continue
        current_wav = AudioSegment.from_file(os.path.join(wav_folder, file))
        merged += current_wav
    filename = os.path.join(wav_folder, "merged.wav")
    merged.export(filename)
    return filename


def create_dataset(user_id, model_name):
    filename = merge_wav(user_id, model_name)
    save_drop_dataset_audio(filename, f"{user_id}_{model_name}")


def preprocess_dataset(user_id, model_name):
    # TODO: заменить 40000 на select
    run_preprocess_script(f"{user_id}_{model_name}", f"assets/datasets/{user_id}_{model_name}", "40000")


def extract_features(user_id, model_name):
    # TODO: заменить 40000 на select
    run_extract_script(f"{user_id}_{model_name}", "v2", "rmvpe", 128, "40000")


def train(user_id, model_name):
    create_dataset(user_id, model_name)
    preprocess_dataset(user_id, model_name)
    extract_features(user_id, model_name)
