# -*- mode: ruby -*-
# vi: set ft=ruby :
Vagrant.configure('2') do |config|
  config.vm.box      = 'ubuntu/xenial64'
  config.vm.hostname = 'deep-learning'
  # config.vm.network :forwarded_port, guest: 3000, host: 3000
  # config.vm.network :forwarded_port, guest: 8888, host: 8888

  config.vm.provider "virtualbox" do |v|
    v.memory = 8192
    v.cpus = 8
    v.customize ["modifyvm", :id, "--cpuexecutioncap", "50"]
  end

  config.vm.provision :shell, path: 'provision.sh', keep_color: true
  config.vm.synced_folder ".", "/home/ubuntu/"
end
