# FSLeyes Plugin for Shimming Toolbox

This plugin allows users to integrate `NeuroPoly`'s `shimming-toolbox` application with the `FSLeyes` GUI. You can read the [Shimming Toolbox Documentation](https://shimming-toolbox.org/en/latest/) for more information, or view the [Shimming Toolbox GitHub Repo](https://github.com/shimming-toolbox/shimming-toolbox) to see the source code.

## Installation

In the `fsleyes-plugin-shimming-toolbox` folder, run:

```
make install
```

## Running

Now, you can open the `FSLeyes` GUI from any environment by running:

```
shimming-toolbox
```

The `FSLeyes` GUI should open. Once it is open, select from the toolbar:

```
Settings --> OrthoView --> Shimming Toolbox
```

The `ShimmingToolbox` plugin should open as a panel.

## Developer Section

### Testing with Docker

We can use `Docker` to spin up a Linux instance and test our install procedure in a clean
environment. You will need to install `Docker` on your computer first: https://www.docker.com/products/docker-desktop

To create our testing container, we will first build an image called `fpst:latest`:
```
docker build --tag fpst:latest .
```

Once our image is built, we want to remove any running instances of the container:
```
docker rm --force fpst
```

Then, we can create a container from our `fpst:latest` image:
```
docker run --name fpst -dit fpst:latest
```

To test our package, we can use the `bash` function of the container:
```
docker exec -it fpst bash
```

Once inside the container terminal, we can find our plugin package and test it:
```
cd src/fsleyes-plugin-shimming-toolbox
make install
```

Altogether:

```
docker rm --force fpst
docker build --tag fpst:latest .
docker run --name fpst -dit fpst:latest
docker exec -it fpst bash
```

### Testing with VirtualBox

To test on different operating systems, you will need to use a virtual machine. You will need to
install `VirtualBox`: https://www.virtualbox.org/wiki/Downloads. You will also need to install
the Oracle VM VirtualBox Extension Pack in order to test MacOSX.

`Vagrant` is a tool that interfaces with `VirtualBox` and streamlines the process:
https://learn.hashicorp.com/tutorials/vagrant/getting-started-index?in=vagrant/getting-started

We have 3 different folders with `Vagrantfile`s for testing each OS:

```
| testing
| -- vagrant_linux/
| -- vagrant_mac/
| -- vagrant_windows/
```

To create the virtual box, run:
```
cd testing/vagrant_{OS}
vagrant up
```

Next, ssh into the shell and run the `fsleyes-plugin-shimming-toolbox` installer:
```
cd src/fsleyes-plugin/shimming-toolbox/
sudo make install
source /Users/vagrant/.bashrc
```

To convert the install to an editable developer install:
```
source /Users/vagrant/shimming_toolbox/python/etc/profile.d/conda.sh
conda activate pst_venv_1267b18e73341ad94da34474
sudo pip3 install -e .
```

#### GUI in VirtualBox

Currently this is only functional for the Mac VirtualBox. When you run `vagrant up`, a GUI
should open. When you see the login screen, enter:

```
user: vagrant
password: vagrant
```

You can open a terminal from this GUI, and run `FSLeyes` by:

```
cd src/fsleyes-plugin/shimming-toolbox/
make run
```

#### Vagrant Tips

To stop the box from running (but not remove it):
```
vagrant suspend
```

To resume the box:
```
vagrant resume
```

To remove the box completely:
```
vagrant destroy
```

If you update your `Vagrantfile` and you want to reload the box:
```
vagrant reload
```
