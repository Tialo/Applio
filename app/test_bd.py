from utils import db

with db.connect() as con:
    curs = con.cursor()
    curs.execute("select * from queue")
    res = curs.fetchall()
    print(res)
    curs.execute("delete from queue")
