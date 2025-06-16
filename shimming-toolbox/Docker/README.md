# Steps to run ST and SCT on the MRI console

## Step 1 (on your personal computer) - Build and export the environment
> [!NOTE]
> Since most of the commands executed in this step require a long time to be executed, it is best to do them prior to the experiment on the MRI console.

### 1.1 - Go to the ST directory
In a terminal window, navigate to the directory where Shimming Toolbox (ST) is located:
```
cd path/to/ST
```

### 1.2 - Build the Docker image
```
docker build --platform linux/amd64 -t st_image ./Docker 
```
This step should last around 8 to 10 minutes.

### 1.3 - Create a container
This container will serve as the base filesystem for transfer:
```
docker run -d --name st-builder st_image
```

### 1.4 - Export the container filesystem
Plug the USB key into your computer. Find the USB key subfolder, then copy the full path (e.g., `path/to/the/usb/key`) and run :
```
docker export st-builder -o st-rootfs.tar
cp st-rootfs.tar path/to/the/usb/key/
cp Docker/docker_tar_to_chroot.sh path/to/the/usb/key/
rm st-rootfs.tar
```
This step should last around 30 minutes. The `st-rootfs.tar`file should weight around 8 GB.

## Step 2 - Prepare the MRI console

## 2.1 – Locate the USB key
Plug the USB key into the MRI console. Find the USB key subfolder, then copy the full path (e.g., `path/to/the/usb/key`). To verify if it contains the filesystem, run:
```
cd path/to/the/usb/key
ls
```
You should see the following files : `st-rootfs.tar`, `docker_tar_to_chroot.sh`.

### 2.2 – Prepare the chroot image
```
mkdir -p st_rootfs
path/to/the/usb/key/docker_tar_to_chroot.sh path/to/the/usb/key/st-rootfs.tar st_rootfs/st-chroot.img
```
Don't forget to replace `path/to/the/usb/key` with the path you copied in step 2.1.

## Step 3 - Use ST and SCT on the MRI console

### 3.1 - Enter the chroot image
```
sudo chroot st_rootfs/st-chroot.img /bin/bash
```

### 3.2 - Run commands
Once you’re inside the chroot, you can run any ST or SCT command as usual, for example:
```
st_dicom_to_nifti -i /path/to/dicoms --subject <subject_name> -o /path/to/nifti
```
> [!WARNING]
> Make sure that the paths you provide (e.g., /path/to/dicoms) exist inside the chroot environment (they may differ from the host system).

### 3.3 - Exit
Once your work is done, to exit the environment, just type:
```
exit
```
