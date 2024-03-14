FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
USER root

# 设置环境变量
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# Anaconda的环境变量
ENV PATH /opt/conda/bin:$PATH 
ENV ANACONDA_VERSION="Anaconda3-2020.02-Linux-x86_64"

# 下载依赖的软件包
# wget下载Anaconda用
RUN buildDeps='wget sudo vim' \ 
&& apt-get update \
&& apt-get install -y $buildDeps \
# 清除apt缓存
&& rm -rf /var/lib/apt/lists/*

# 下载 安装anaconda并配置环境变量
RUN wget --quiet https://repo.anaconda.com/archive/$ANACONDA_VERSION.sh -O ~/anaconda.sh \
# 安装anaconda
&& /bin/bash ~/anaconda.sh -b -p /opt/conda \
# 删除安装包
&& rm ~/anaconda.sh \
&& ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh  \
&& echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc  

# 设置工作目录
RUN mkdir /workdir \
&& chmod 777 /workdir

RUN mkdir /.dgl \
&& chmod 777 /.dgl
WORKDIR /workdir

# 安装必要的Python包
COPY requirements.txt ./
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ \
&& conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ \
&& conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ \
&& conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ \
&& conda config --set show_channel_urls yes \
&& conda config --set remote_connect_timeout_secs 40 \
&& conda config --set remote_read_timeout_secs 100 \
&& conda update -n base -c defaults conda \
&& conda install -y pytorch=1.6.0 torchvision cudatoolkit=10.1

EXPOSE 22 8888 6006 8080