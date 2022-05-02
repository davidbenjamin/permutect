FROM python:3.8-buster

COPY requirements.txt /
COPY setup.py /
RUN pip install --no-cache-dir -r /requirements.txt

ADD mutect3/ /mutect3

RUN pip install build
RUN python3 -m build --sdist
RUN pip install dist/*.tar.gz

CMD ["/bin/sh"]
