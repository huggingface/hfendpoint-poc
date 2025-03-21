FROM vllm/vllm-openai:v0.8.1

# Copy current hfendpoint folder for now
COPY . /opt/hfendpoint

WORKDIR /opt
RUN cd /opt/hfendpoint &&  \
    python3 -m pip install .[openai,audio] && \
    python3 -m pip install -r examples/whisper/requirements.txt


WORKDIR /opt/hfendpoint/examples/whisper

ENV GRANIAN_TASK_IMPL=asyncio
ENV GRANIAN_WORKERS=1
ENV GRANIAN_RUNTIME_BLOCKING_THREADS=4
ENV GRANIAN_RUNTIME_THREADS=4

ENV MODEL_ID=/repository
ENV VLLM_USE_V1=0

EXPOSE 5000
ENTRYPOINT ["granian", "--interface", "asgi", "--host", "0.0.0.0", "--port", "5000", "--runtime-mode", "mt"]
CMD ["transcribe:app"]