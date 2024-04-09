# Start from a core stack version
FROM jupyter/datascience-notebook:latest

COPY requirements1.txt /tmp/requirements1.txt
RUN python3 -m pip install -r /tmp/requirements1.txt

COPY requirements2.txt /tmp/requirements2.txt
RUN python3 -m pip install -r /tmp/requirements2.txt --no-deps

COPY requirements3.txt /tmp/requirements3.txt
RUN python3 -m pip install -r /tmp/requirements3.txt