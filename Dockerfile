FROM ubuntu:20.04

WORKDIR /app

ADD . /app

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.8 python3.8-dev python3-pip python3-setuptools python3-wheel \
    git make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev \
    libxmlsec1-dev libffi-dev liblzma-dev

RUN curl https://pyenv.run | bash && \
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc && \
    echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

RUN pyenv install && pyenv local

RUN pyenv exec pip install --no-cache-dir poetry

RUN poetry install --with gpu

CMD ["poetry", "run", "dvc", "repro"]
