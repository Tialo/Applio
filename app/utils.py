import os
import sqlite3
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
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
                    sample rate integer not null
                )
            """)
            con.commit()
            cursor.execute("""
                create table if not exists models (
                    id integer primary key,
                    model_name text not null,
                    user_id integer not null,
                    public integer not null
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
                    task_type text not null
                )
            """)
            con.commit()


db = DataBase()
