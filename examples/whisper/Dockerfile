FROM vllm/vllm-openai

# Copy current infinity folder for now
COPY . /opt/infinity

WORKDIR /opt
RUN cd /opt/infinity &&  \
    python3 -m pip install .[openai,vllm,audio] && \
    python3 -m pip install -r examples/whisper/requirements.txt


WORKDIR /opt/infinity/examples/whisper

ENV GRANIAN_TASK_IMPL=asyncio
ENV GRANIAN_WORKERS=1
ENV GRANIAN_RUNTIME_BLOCKING_THREADS=4
ENV GRANIAN_RUNTIME_THREADS=4

EXPOSE 5000
ENTRYPOINT ["granian", "--interface", "asgi", "--host", "0.0.0.0", "--port", "5000", "--runtime-mode", "mt"]
CMD ["endpoint:app"]