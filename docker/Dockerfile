FROM ubuntu:22.04

# Upgrade OS
RUN apt-get update -q && \
    DEBIAN_FRONTEND=noninteractive apt-get upgrade -y && \
    apt-get autoclean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

# Timezone, Launguage設定
RUN apt update &&\
    apt install -y --no-install-recommends \
    locales \
    software-properties-common tzdata &&\
    locale-gen ja_JP ja_JP.UTF-8 &&\
    update-locale LC_ALL=ja_JP.UTF-8 LANG=ja_JP.UTF-8 &&\
    add-apt-repository universe

ENV LANG=ja_JP.UTF-8
ENV TZ=Asia/Tokyo

# gui tools
RUN apt-get update && \
    apt-get install -y libqt5gui5 && \
    rm -rf /var/lib/apt/lists/*
ENV QT_DEBUG_PLUGINS=1

# install basic tools
RUN apt update &&\
    apt-get install -y nano \
    iputils-ping \
    net-tools \
    git \
    sudo &&\
    rm -rf /var/lib/apt/lists/*

# Add user and group, then allow sudo
ARG UID
ARG GID
ARG PASSWORD
ARG USER_NAME
ARG GROUP_NAME
ARG GIT_USER_NAME
ARG GIT_USER_EMAIL

RUN groupadd -g ${GID} ${GROUP_NAME}
RUN useradd -u ${UID} -g ${GID} -G sudo -s /bin/bash -m ${USER_NAME}
RUN echo "${USER_NAME}:${PASSWORD}" | chpasswd
RUN echo "Defaults visiblepw" >> /etc/sudoers
RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER ${USER_NAME}

# install pip
COPY requirements.txt /home/${USER_NAME}/requirements.txt

RUN sudo apt-get update
RUN sudo apt-get install -y python3-pip
RUN sudo pip3 install --upgrade pip
RUN sudo pip3 install --upgrade setuptools

RUN sudo pip3 install -r /home/${USER_NAME}/requirements.txt --ignore-installed

# git config
RUN git config --global user.name "${GIT_USER_NAME}" && \
    git config --global user.email "${GIT_USER_EMAIL}"

# setup
COPY setup.sh /home/${USER_NAME}/setup.sh
RUN bash ~/setup.sh

CMD ["/bin/bash"]
