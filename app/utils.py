from io import BytesIO
import os
import shutil
from string import ascii_letters
import sqlite3
from pathlib import Path

from telegram import Bot
from pydub import AudioSegment

token = os.environ["TOKEN"]

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
os.makedirs(DATA_DIR, exist_ok=True)


class DataBase:
    def __init__(self, db_name=None):
        if db_name is None:
            db_name = os.path.join(Path(__file__).parent.parent, "main.db")
        self.db_name = db_name
        self.create_tables()

    def connect(self):
        return sqlite3.connect(self.db_name)

    def create_tables(self):
        with self.connect() as con:
            cursor = con.cursor()
            cursor.execute("""
                create table if not exists train_params (
                    id integer primary key,
                    user_id integer not null,
                    model_name text,
                    pretrain_name text,
                    epochs integer,
                    batch_size integer
                )
            """)
            con.commit()
            cursor.execute("""
                create table if not exists pretrains (
                    id integer primary key,
                    pretrain_name text not null,
                    sr integer not null,
                    d_path text not null,
                    g_path text not null
                )
            """)
            con.commit()
            cursor.execute("""
                create table if not exists models (
                    id integer primary key,
                    model_name text not null,
                    user_id integer not null,
                    public integer not null,
                    pretrain_name text not null,
                    epochs integer not null,
                    batch_size integer not null
                )
            """)
            con.commit()
            cursor.execute("""
                create table if not exists train_data (
                    id integer primary key,
                    user_id integer not null,
                    model_name integer not null,
                    filename text not null
                )
            """)
            con.commit()
            cursor.execute("""
                create table if not exists queue (
                    id integer primary key,
                    user_id integer not null,
                    model_name integer not null,
                    status text not null,
                    add_time integer timestamp not null,
                    task_type text not null,
                    infer_path text,
                    f0up integer
                )
            """)
            con.commit()
            cursor.execute("""
                create table if not exists infers (
                    user_id integer not null,
                    model_name text,
                    infer_path text not null
                )
            """)


db = DataBase()


def upload_pretrain(pretrain_name, sr, g_path, d_path):
    with db.connect() as con:
        curs = con.cursor()
        curs.execute("select pretrain_name from pretrains")
        pretrains = set(curs.fetchall())

    pretraineds_custom_dir = Path(__file__).parent.parent / "rvc" / "pretraineds" / "pretraineds_custom"
    if pretrain_name in pretrains:
        raise ValueError("Модель с таким названием уже есть")

    if sr not in [32000, 40000, 48000]:
        raise ValueError("Допустимые значения sample rate: [32000, 40000, 48000]")

    if not os.path.isfile(g_path):
        raise ValueError("Такого файла нет. Введите правильный путь до генератора")

    if not os.path.isfile(d_path):
        raise ValueError("Такого файла нет. Введите правильный путь до дискриминатора")

    new_g_path = os.path.join(pretraineds_custom_dir, os.path.basename(g_path))
    new_d_path = os.path.join(pretraineds_custom_dir, os.path.basename(d_path))
    shutil.copy(g_path, new_g_path)
    shutil.copy(d_path, new_d_path)

    with db.connect() as con:
        curs = con.cursor()
        curs.execute(
            "insert into pretrains (pretrain_name, sr, g_path, d_path) values (?, ?, ?, ?)",
            (pretrain_name, sr, new_g_path, new_d_path)
        )
        con.commit()


def upload_model(model_path, index_path, model_name, user_id, pretrain_name, epochs, batch_size, public=False):
    if not model_path.endswith(".pth"):
        raise ValueError("Файл модели должен иметь расширение .pth")
    if not index_path.endswith(".index"):
        raise ValueError("Файл индекса модели должен иметь расширение .index")

    with db.connect() as con:
        curs = con.cursor()
        curs.execute("select count(*) from models where user_id = ? and model_name = ?", (user_id, model_name))
        if curs.fetchone()[0]:
            raise ValueError("Модель с таким названием уже есть у пользователя")
        curs.execute("select count(*) from pretrains where pretrain_name = ?", (pretrain_name, ))
        if not curs.fetchone()[0]:
            raise ValueError(f"Pretrain модели с именем {pretrain_name} не существует")

    if not valid_model_name(model_name):
        raise ValueError("В названии модели могут быть только латинские буквы и цифры")
    with db.connect() as con:
        curs = con.cursor()
        curs.execute(
            "insert into models (user_id, model_name, pretrain_name, public, epochs, batch_size) values "
            "(?, ?, ?, ?, ?, ?)", (user_id, model_name, pretrain_name, public, epochs, batch_size)
        )

    os.makedirs(os.path.join(MODELS_DIR, f"{user_id}_{model_name}"), exist_ok=True)
    shutil.copy(model_path, os.path.join(MODELS_DIR, f"{user_id}_{model_name}", "model.pth"))
    shutil.copy(index_path, os.path.join(MODELS_DIR, f"{user_id}_{model_name}", "index.index"))


def valid_model_name(name):
    for c in name:
        if c not in ascii_letters and c not in "0123456789":
            return False
    return True


async def send_infer_file(user_id, file_path):
    async with Bot(token) as bot:
        with BytesIO() as bio:
            audio = AudioSegment.from_ogg(file_path)
            duration = audio.duration_seconds
            audio.export(bio)
            bio.seek(0)
            await bot.send_voice(chat_id=user_id, voice=bio, caption="Преобразованное сообщение", duration=duration)
