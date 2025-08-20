# Setup instructions

We will setup a server that will "plaster" data captured using BRICS.

  
- Install relevant packages
```
sudo apt update
sudo apt install -y build-essential cmake libboost-all-dev gdb iperf sshfs chrony iputils-ping
```

- Setup key-pair authentication for Github
```
ssh-keygen -t rsa -C "srinath+brics@brown.edu" # Default pub file, empty password
cat ~/.ssh/id_rsa.pub # Copy the public key
```

Paste the copied public key to [https://github.com/settings/keys](https://github.com/settings/keys).  You can test if it worked by `ssh -T git@github.com`.

- Clone repo: `mkdir -p ~/code && cd ~/code && git clone git@github.com:brown-ivl/mortar.git`

- Setup shell functions and bashrc, and install services:
```
sudo cp ~/code/mortar/server/sink/sink.bashrc ~/.bashrc
```

- Compile special version of GCC for C++20 features. This will take some time (>1 hour).
```
mkdir ~/sources
cd ~/sources
wget https://ftp.mpi-inf.mpg.de/mirrors/gnu/mirror/gcc.gnu.org/pub/gcc/releases/gcc-13.3.0/gcc-13.3.0.tar.gz
tar xvf gcc-13.3.0.tar.gz
cd gcc-13.3.0
./contrib/download_prerequisites
mkdir objdir
cd objdir
# Don't install system-wide. It will mess things up
$PWD/../configure --prefix=/opt/mortar --enable-languages=c,c++ --disable-multilib
nohup make -j $(nproc) 2>&1 | tee progress.log &
sudo make install
```

- Install OpenCV for file efficient file writing

```
sudo apt install ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev zip
mkdir -p ~/sources
cd ~/sources
wget https://github.com/opencv/opencv/archive/3.4.16.zip
unzip 3.4.16.zip
cd opencv-3.4.16
mkdir build && cd build
cmake \
-D CMAKE_BUILD_TYPE=RELEASE \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D WITH_FFMPEG=ON \
-D WITH_QT=OFF \
-D WITH_TIFF=OFF \
-D WITH_GTK=ON \
-D BUILD_opencv_python2=OFF \
-D BUILD_opencv_python3=OFF \
-D WITH_GSTREAMER=OFF \
-D WITH_JPEG=ON \
-D CMAKE_INSTALL_PREFIX=/opt/mortar \
..
make -j$(nproc)
sudo make install
```

- Build the sink server
```
mkdir -p ~/code/mortar/build
cd ~/code/mortar/build
cmake -DCMAKE_C_COMPILER=/opt/mortar/bin/gcc -DCMAKE_CXX_COMPILER=/opt/mortar/bin/g++ -DCMAKE_INSTALL_PREFIX=/opt/mortar -DBUILD_SINK_SERVER=ON -DCMAKE_BUILD_TYPE=Release .. 
sudo make install
```

- Set chrony configuration

```
sudo mv /etc/chrony/chrony.conf /etc/chrony/chrony.conf.old
sudo cp ~/code/mortar/server/sink/chrony.conf /etc/chrony/chrony.conf
# Optional
sudo timedatectl set-timezone America/New_York
```

- Install services. To mount Brown disk via SSHFS, you will need to [follow instructions here](https://cs.brown.edu/about/system/connecting/ssh/osx/).

```
sudo systemctl enable /opt/mortar/mortar-mount-brics-universe.service
#sudo systemctl enable /opt/mortar/mortar-start-sink.service
sudo reboot
```

