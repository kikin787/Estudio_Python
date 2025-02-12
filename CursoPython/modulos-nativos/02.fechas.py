# import time

# print(time.time())

from datetime import datetime

fecha = datetime(2025, 1, 1)
fecha2 = datetime(2025, 2, 1)

ahora = datetime.now()
fechaStr = datetime.strptime('2023-1-01', '%Y-%m-%d')

print(fecha.strftime("%Y.%m.%d"))
print(fecha2, ahora, fechaStr)
print(fecha > fecha2)

print(
    fecha.year,
    fecha.month,
    fecha.day,
    fecha.hour,
    fecha.minute,
    fecha.second,
)