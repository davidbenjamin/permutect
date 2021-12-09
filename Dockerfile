FROM python:3.8-slim-buster

COPY requirements.txt /
COPY setup.py /
RUN pip install -r /requirements.txt

ADD mutect3/ /mutect3

RUN python3 -m build --sdist
RUN pip install dist/*.tar.gz

CMD ["/bin/sh"]