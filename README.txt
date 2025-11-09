
📦 Despliegue de backend CAPTCHA en Render.com

1. Crea una cuenta en https://render.com
2. Crea un nuevo servicio Web (Web Service)
3. Conecta este repositorio (o sube los archivos como ZIP)
4. Elige:
   - Runtime: Python 3
   - Build command: pip install -r requirements.txt
   - Start command: python backend_captcha_recoleccion.py
5. El servicio se desplegará en una URL pública, por ejemplo:
   https://captcha-backend.onrender.com

6. En tu HTML usa esa URL en fetch:
   fetch("https://captcha-backend.onrender.com/captura", {...})

📝 Los datos se guardan en la carpeta /datos_captcha como archivos CSV por día.
