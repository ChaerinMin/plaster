# Setup instructions

We will setup a server that will "plaster" data captured using BRICS.

  
- Install relevant packages
```
sudo apt update
sudo apt install -y build-essential cmake libboost-all-dev gdb iperf sshfs chrony iputils-ping git libgl1
```

- Setup key-pair authentication for Github
```
ssh-keygen -t rsa -C "srinath+brics@brown.edu" # Default pub file, empty password
cat ~/.ssh/id_rsa.pub # Copy the public key
```

Paste the copied public key to [https://github.com/settings/keys](https://github.com/settings/keys).  You can test if it worked by `ssh -T git@github.com`.

- Clone repo: `mkdir -p ~/code && cd ~/code && git clone git@github.com:brown-ivl/plaster.git`. Follow instructions in its repo for installation.

- Setup shell functions and bashrc, and install services:
```
cp ~/code/plaster/server/plaster.bashrc ~/.bashrc
sudo cp ~/code/plaster/server/mount-data.* /opt/
```

- Set chrony configuration

```
sudo mv /etc/chrony/chrony.conf /etc/chrony/chrony.conf.old
sudo cp ~/code/plaster/server/chrony.conf /etc/chrony/chrony.conf
# Optional
sudo timedatectl set-timezone America/New_York
```

- Install services. To mount Brown disk via SSHFS, you will need to [follow instructions here](https://cs.brown.edu/about/system/connecting/ssh/osx/).

```
sudo mkdir /mnt/brics-universe /mnt/brics-studio /mnt/project-bolt
sudo chown $USER /mnt/brics-universe /mnt/brics-studio /mnt/project-bolt
sudo chmod 770 /mnt/brics-universe /mnt/brics-studio /mnt/project-bolt
sudo systemctl enable /opt/mount-data.service
sudo reboot
```

