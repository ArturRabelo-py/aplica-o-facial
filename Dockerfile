# Usa uma imagem base do Python 3.9
FROM python:3.9-slim

# Define o diretório de trabalho dentro do "computador"
WORKDIR /app

# Copia a sua "lista de compras" para dentro do computador
COPY requirements.txt .

# Instala todas as bibliotecas necessárias
RUN pip install --no-cache-dir -r requirements.txt

# Copia o resto do seu projeto (o seu ficheiro .py, a pasta cadastros, etc.)
COPY . .

# Expõe a porta 8050 para que o mundo exterior possa aceder
EXPOSE 8050

# A instrução final para ligar o servidor
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "--timeout", "300", "Bobina:server"]
