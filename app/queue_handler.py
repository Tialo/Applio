import time

import train
from utils import db


def main_loop():
    while True:
        with db.connect() as con:
            curs = con.cursor()
            curs.execute("select count(*) from queue where status not in  ('done', 'error')")
            tasks = curs.fetchone()[0]
            if not tasks:
                print("no tasks")
                time.sleep(10)
                continue
            curs.execute("select count(*) from queue where status = 'queue'")
            queue_tasks = curs.fetchone()[0]
            if queue_tasks != tasks or not tasks:
                print("something is running")
                time.sleep(10)
                continue
            curs.execute(
                "select id, task_type, user_id, model_name from queue "
                "where status = 'queue' order by add_time asc limit 1"
            )
            [task_id, task_type, user_id, model_name] = curs.fetchone()
            # curs.execute("update queue set status = ? where id = ?", ("running", task_id))
            con.commit()
            if task_type == "train":
                print(f"train {user_id=} {model_name=}")
                train.train(user_id, model_name)
                curs.execute("update queue set status = ? where id = ?", ("done", task_id))
                con.commit()
            else:
                raise NotImplementedError


if __name__ == '__main__':
    main_loop()
