1. conda创建Python虚拟环境

2. 安装依赖 canda--》pip--》编译安装

3. 安装conda  官网下载安装 conda --version

4. 常用命令
  - 列出所有环境  conda env list
  - 创建环境 conda create --name 环境名称
  - 进入环境 conda activate 环境名称
  - 退出环境 conda deactivatve --name 环境名称 --all
  - 创建指定Pythe
  - 删除环境 conda remove版本环境 conda env remove --name py3.7

5. conda镜像源
conda config --add channels https://mirrors.aliyun.com/anaconda/pkgs/main/
conda config --add channels https://mirrors.aliyun.com/anaconda/cloud/conda-forge/

 conda config --add channels https://mirrors.aliyun.com/anaconda/cloud/bioconda/

 conda config --show-sources
6.  安装依赖包
conda install opencv
pip install mediapipe


1. 切conda环境，然后安装

  conda install jupyterlab

2. 启动
  jupyter-lab 
