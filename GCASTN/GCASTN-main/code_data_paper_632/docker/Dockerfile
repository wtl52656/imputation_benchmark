FROM a4-1:5000/huangyiheng/pristi-pytorch:1.10-cuda11.3
USER root

# 设置环境变量
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN mkdir /.dgl \
&& chmod 777 /.dgl
WORKDIR /workdir

# 安装必要的Python包
COPY requirements.txt ./
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple