FROM ubuntu:16.04
ENV http_proxy $HTTP_PROXY
ENV https_proxy $HTTPS_PROXY
ARG DOWNLOAD_LINK=http://registrationcenter-download.intel.com/akdlm/irc_nas/16057/l_openvino_toolkit_p_2019.3.376.tgz
ARG INSTALL_DIR=/opt/intel/openvino
ARG TEMP_DIR=/tmp/openvino_installer
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    cpio \
    sudo \
    lsb-release && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p $TEMP_DIR && cd $TEMP_DIR && \
    wget -c $DOWNLOAD_LINK && \
    tar xf l_openvino_toolkit*.tgz && \
    cd l_openvino_toolkit* && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    rm -rf $TEMP_DIR
RUN $INSTALL_DIR/install_dependencies/install_openvino_dependencies.sh
# build Inference Engine samples
RUN mkdir $INSTALL_DIR/deployment_tools/inference_engine/samples/build && cd $INSTALL_DIR/deployment_tools/inference_engine/samples/build && \
    /bin/bash -c "source $INSTALL_DIR/bin/setupvars.sh && cmake .. && make -j1"

#Movidius
RUN apt-get install unzip autoconf automake libtool --yes
RUN cd /tmp/ && \
   wget https://github.com/libusb/libusb/archive/v1.0.22.zip && \
   unzip v1.0.22.zip && cd libusb-1.0.22 && \
   ./bootstrap.sh && \
   ./configure --disable-udev --enable-shared && \
   make -j4 && make install && \
   rm -rf /tmp/*

#python
RUN apt install software-properties-common --yes
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && sudo apt install python3.7 python3-pip python3-dev --yes
RUN apt-get install libpython3.7-dev --yes
RUN python3.7 -m  pip install --upgrade pip

#jupyter
RUN python3.7 -m pip install jupyter

#openCL

RUN cd /opt/intel/openvino/install_dependencies && bash install_NEO_OCL_driver.sh

RUN useradd -G video -ms /bin/bash user

#user
#RUN useradd -m -g sudo user

RUN python3.7 -m pip install -r /opt/intel/openvino/inference_engine/demos/python_demos/requirements.txt
RUN python3.7 -m pip install scipy pyyaml
RUN echo "source /opt/intel/openvino/bin/setupvars.sh" >> /home/user/.bashrc
RUN echo "alias python='python3.7'" >> /home/user/.bashrc

USER user
WORKDIR /home/user
RUN mkdir predictor

#run
EXPOSE 8899
ENTRYPOINT ["bash", "predictor/run.sh"]
#ENTRYPOINT ["jupyter-notebook", "--ip=0.0.0.0", "--port=8899"]
#CMD [""]
