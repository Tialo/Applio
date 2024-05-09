import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import shutil

from pydub import AudioSegment

from tabs.train.train import save_drop_dataset_audio
from core import run_preprocess_script, run_extract_script, run_train_script

from utils import db, DATA_DIR, MODELS_DIR, ROOT_DIR, update_status


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


def preprocess_dataset(user_id, model_name, sr):
    run_preprocess_script(f"{user_id}_{model_name}", f"assets/datasets/{user_id}_{model_name}", str(sr))


def extract_features(user_id, model_name, sr):
    run_extract_script(f"{user_id}_{model_name}", "v2", "rmvpe", 128, str(sr))


def train_model(user_id, model_name, epochs, batch_size, sr, g_path, d_path):
    run_train_script(
        model_name=f"{user_id}_{model_name}",
        rvc_version="v2",
        save_every_epoch=100,
        save_only_latest=True,
        save_every_weights=False,
        total_epoch=epochs,
        sampling_rate=sr,
        batch_size=batch_size,
        gpu="0",
        pitch_guidance=True,
        overtraining_detector=True,
        overtraining_threshold=75,
        pretrained=True,
        custom_pretrained=True,
        g_pretrained_path=g_path,
        d_pretrained_path=d_path
    )


def save_model(user_id, model_name):
    index_dir = os.path.join(ROOT_DIR, "logs", f"{user_id}_{model_name}")
    index_file = [
        file for file in
        os.listdir(index_dir)
        if file.endswith(".index") and file.startswith("added_")
    ][0]
    model_dir = os.path.join(ROOT_DIR, "logs")
    model_file = [
        file for file in
        os.listdir(model_dir)
        if file.endswith(".pth") and file.startswith(f"{user_id}_{model_name}")
    ][0]
    os.makedirs(os.path.join(MODELS_DIR, f"{user_id}_{model_name}"), exist_ok=True)
    os.rename(os.path.join(index_dir, index_file), os.path.join(MODELS_DIR, f"{user_id}_{model_name}", "index.index"))
    os.rename(os.path.join(model_dir, model_file), os.path.join(MODELS_DIR, f"{user_id}_{model_name}", "model.pth"))
    shutil.rmtree(DATA_DIR / str(user_id) / model_name)
    shutil.rmtree(index_dir)
    shutil.rmtree(ROOT_DIR / "assets" / "datasets" / f"{user_id}_{model_name}")


def train(task_id, user_id, model_name):
    with db.connect() as con:
        curs = con.cursor()
        curs.execute(
            "select epochs, batch_size, pretrain_name from models where user_id = ? and model_name = ?",
            (user_id, model_name)
        )
        [epochs, batch_size, pretrain_name] = curs.fetchone()
        curs.execute("select sr, g_path, d_path from pretrains where pretrain_name = ?", (pretrain_name, ))
        [sr, g_path, d_path] = curs.fetchone()
    update_status(task_id, "Создание датасета")
    create_dataset(user_id, model_name)
    update_status(task_id, "Предобработка датасета")
    preprocess_dataset(user_id, model_name, sr)
    update_status(task_id, "Вычисление признаков")
    extract_features(user_id, model_name, sr)
    update_status(task_id, "Обучение модели")
    train_model(user_id, model_name, epochs, batch_size, sr, g_path, d_path)
    update_status(task_id, "Сохранение модели")
    save_model(user_id, model_name)
