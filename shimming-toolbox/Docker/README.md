# Steps to run ST on the MARS system hardware

## Step 1 (on your personal computer) - Build and export the chroot image
> [!NOTE]
> Since most of the commands in this step are time-consuming, it is recommended to perform them before the experiment.

### 1.1 - Navigate to the ST Docker directory
In a terminal window, go to the Docker directory where Shimming Toolbox (ST) is located:
```
cd <path/to/ST>
cd Docker
```
This folder contains the Dockerfile and scripts used to generate the chroot image.

### 1.2 - Create the chroot image
The following command builds the ST Docker image, exports the container’s filesystem to a tarball (.tar file) and creates a chroot image (.img file) representing a bootable Linux root filesystem.
```
chmod +x ./docker_to_chroot.sh
chmod +x ./tar_to_chroot.sh
./docker_to_chroot.sh st_image st_chroot.img
```
> [!NOTE]
> This process typically takes between 12 and 14 minutes.

### 1.3 - Export the chroot image to a USB key
Connect your USB key to the computer. Navigate to its folder and run:
```
zip -j st_chroot.zip st_chroot.img
cp st_chroot.zip <path/to/the/usb/key/>
rm st_chroot.zip
rm st_chroot.img
```
This compresses and transfers the image file to the USB key. The original image is deleted to free up disk space.
> [!NOTE]
> This process typically takes around 10 minutes.

## Step 2 (on the MARS system hardware) - Prepare the environment

## 2.1 – Locate the USB key
Insert the USB key into the MARS system hardware and navigate to its mount point:
```
cd <path/to/the/usb/key>
ls
```
Ensure that st_chroot.zip is present before continuing

### 2.2 – Mount the chroot image on the MARS system
While still in the USB directory, run:
```
mkdir -p /mnt/fire_chroot
unzip st_chroot.zip -d /mnt/fire_chroot
mount --bind /dev /mnt/fire_chroot/st_chroot.img/dev
mount --bind /proc /mnt/fire_chroot/st_chroot.img/proc
mount --bind /sys /mnt/fire_chroot/st_chroot.img/sys
```
These commands mount the image and bind essential system folders (/dev, /proc, and /sys) to the corresponding paths inside the chroot. This provides the chrooted environment with access to hardware devices and system information, making it fully functional.

## Step 3 (on the MARS system hardware) - Use ST

### 3.1 - Enter the chroot environment
```
chroot /mnt/fire_chroot/st_chroot.img /bin/bash
```
This command launches a shell inside the chroot environment, where ST is installed and ready to use.

### 3.2 - Run commands
Once you have started, you can run any ST command as usual, for example:
```
st_dicom_to_nifti -i </path/to/dicoms> --subject <subject_name> -o </path/to/nifti>
```
> [!WARNING]
> Make sure that the paths you provide (e.g., `/path/to/dicoms`) exist inside the chroot. They may differ from the file structure of the host system.

### 3.3 - Exit the chroot environment
To leave the chroot environment, simply type:
```
exit
```
This returns control to the host system’s shell.

## Step 4 (on the MARS system hardware) - Unmount the chroot image
Before replacing or removing the chroot image, unmount it to safely detach it from the system:
```
umount /mnt/fire_chroot/st_chroot.img
```
