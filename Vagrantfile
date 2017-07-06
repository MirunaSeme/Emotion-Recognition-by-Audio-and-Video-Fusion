# -*- mode: ruby -*-
# vi: set ft=ruby :
Vagrant.configure('2') do |config|
  config.vm.box      = 'ubuntu/xenial64'
  config.vm.hostname = 'deep-learning'
  # password is password
  # config.ssh.insert_key = false
  # config.ssh.username = 'ubuntu'
  # config.ssh.password = 'password'
  # config.ssh.private_key_path = './.vagrant/machines/default/virtualbox/private_key'
  # config.vm.network :forwarded_port, guest: 3000, host: 3000
  # config.vm.network :forwarded_port, guest: 8888, host: 8888

  config.vm.provider "virtualbox" do |v|
    v.memory = 1024
    v.cpus = 4
  end

  config.vm.provision :shell, path: 'provision.sh', keep_color: true
  config.vm.synced_folder ".", "/home/vagrant/"
end
