import time

import train
import infer
from utils import db


def main_loop():
    while True:
        with db.connect() as con:
            curs = con.cursor()
            curs.execute("select count(*) from queue where status = 'queue'")
            if not curs.fetchone()[0]:
                time.sleep(5)
                continue
            curs.execute(
                "select id, task_type, user_id, model_name, infer_path, f0up from queue "
                "where status = 'queue' order by add_time asc limit 1"
            )
            [task_id, task_type, user_id, model_name, infer_path, f0up] = curs.fetchone()
            curs.execute("update queue set status = ? where id = ?", ("running", task_id))
            con.commit()
        if task_type == "train":
            print(f"train {user_id=} {model_name=}")
            try:
                train.train(user_id, model_name)
            except Exception as e:
                print(e)
                print(str(e))
                with db.connect() as con:
                    curs = con.cursor()
                    # curs.execute("delete from models where user_id = ? and model_name = ?", (user_id, model_name))
                    # con.commit()
                    curs.execute("update queue set status = ? where id = ?", ("error", task_id))
                    con.commit()
                raise e  # TODO: Заменить на логи
                continue
        elif task_type == "infer":
            print(f"infer {user_id=} {model_name=} {infer_path=}")
            try:
                infer.infer(user_id, model_name, infer_path, f0up)
            except Exception as e:
                print(e)
                print(str(e))
                with db.connect() as con:
                    curs = con.cursor()
                    # curs.execute(
                    #     "delete from infers where user_id = ? and model_name = ? and infer_path = ?",
                    #     (user_id, model_name, infer_path)
                    # )
                    # con.commit()
                    curs.execute("update queue set status = ? where id = ?", ("error", task_id))
                    con.commit()
                raise e  # TODO: Заменить на логи
                continue
        else:
            raise NotImplementedError

        with db.connect() as con:
            curs = con.cursor()
            curs.execute("update queue set status = ? where id = ?", ("done", task_id))
            con.commit()


if __name__ == '__main__':
    main_loop()
