from utils import db

with db.connect() as con:
    curs = con.cursor()
    curs.execute("select * from queue")
    print(curs.fetchall())
    curs.execute("select * from models")
    print(curs.fetchall())
    curs.execute("select * from pretrains")
    print(curs.fetchall())
    # curs.execute("delete from queue")
    # con.commit()
    # curs.execute("delete from models")
    # con.commit()
    # curs.execute("delete from pretrains")
    # con.commit()
    # curs.execute("update queue set status = 'done'")
    # con.commit()
