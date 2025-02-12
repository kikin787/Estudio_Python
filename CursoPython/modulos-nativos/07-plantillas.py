from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from pathlib import Path
import smtplib
from string import Template

# path = Path("modulos-nativos/holamundo.jpg")
# mime_image = MIMEImage(path.read_bytes())
plantilla = Path("CursoPython/modulos-nativos/plantilla.html").read_text("utf-8")
template = Template(plantilla)
cuerpo = template.substitute({"usuario": "Kikin" })

mensaje = MIMEMultipart()
mensaje["from"] = "hola mundo"
mensaje["to"] = "ultimatepython@holamundo.io"
mensaje["subject"] = "esta es una prueba"
cuerpo = MIMEText(cuerpo,"html")
mensaje.attach(cuerpo)
# mensaje.attach(mime_image)

with smtplib.SMTP(host="smtp.gmail.com", port=587)  as smtp:
    smtp.ehlo()
    smtp.starttls()

    smtp.login("velezduran_25@hotmail.com", "chidO787")
    smtp.send_message(mensaje)
    print("mensaje enviado")