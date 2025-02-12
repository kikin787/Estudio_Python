import sqlite3

with sqlite3.connect("CursoPython/sqlite/app.db") as con:
    cursor = con.cursor()
    usuarios = [
        (2, 'Kikin'),
        (3, 'Perro'),
    ]
    cursor.executemany(
        " insert into usuarios values(?, ?)", 
        usuarios,
        )