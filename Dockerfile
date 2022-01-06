FROM python:3.8.5

ADD cellregmap /app/cellregmap/cellregmap
ADD setup.py setup.cfg version.py proof.md LICENSE MANIFEST.in /app/cellregmap/
RUN pip install /app/cellregmap
