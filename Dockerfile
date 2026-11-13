FROM python:3.13-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY router router
COPY sim sim
COPY bench bench
COPY models models
ENV ROUTER_MODEL_FILE=models/passnet.pt ROUTER_TAU_FILE=models/tau.json
EXPOSE 8000
CMD ["uvicorn", "router.service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
