FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --gid 1000 jupyter \
    && useradd --uid 1000 --gid 1000 --create-home jupyter

WORKDIR /home/jupyter/project
COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY solution/ ./solution/
RUN chown -R jupyter:jupyter /home/jupyter/project

USER jupyter
CMD ["python", "solution/run.py", "--help"]
