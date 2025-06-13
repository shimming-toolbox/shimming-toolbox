# Steps to run ST and SCT on the MRI console

## Step 1 (on your personal computer) - Build and export the environment
> [!NOTE]
> Since some of these steps require a long time to be executed, it is best to do them prior to the experiment on the MRI console.

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
docker run -d --name st-builder st_image sleep infinity
```

### 1.4 - Export the container filesystem
Plug the USB key into your computer. Find the USB key subfolder, then copy the full path (e.g., `path/to/the/usb/key`) and run :
```
docker export st-builder -o st-rootfs.tar
cp st-rootfs.tar path/to/the/usb/key/
rm st-rootfs.tar
```
This step should last around 30 minutes.

## Step 2 - Prepare the MRI console

## 2.1 – Locate the USB key
Plug the USB key into the MRI console. Find the USB key subfolder, then copy the full path (e.g., `path/to/the/usb/key`). To verify if it contains the filesystem, run:
```
cd path/to/the/usb/key
ls
```
You should see the file `st-rootfs.tar`.

### 2.2 - Copy the container filesystem from the USB key
Copy the st-rootfs.tar file from the USB key to the appropriate directory on the MRI console:
```
cd
sudo mkdir -p /opt/st/
sudo cp path/to/the/usb/key/st-rootfs.tar /opt/st/
```
Don't forget to replace `path/to/the/usb/key` with the path you copied in step 2.1.

### 2.3 – Extract and prepare the chroot environment
```
sudo mkdir -p /opt/st/chroot
sudo tar xf /opt/st/st-rootfs.tar -C /opt/st/chroot
```

### 2.4 - Mount essential directories
These are needed to allow chroot to interact with the system:
```
sudo mount --bind /dev /opt/st/chroot/dev
sudo mount -t proc proc /opt/st/chroot/proc
sudo mount -t sysfs sys /opt/st/chroot/sys
```

## Step 3 - Use ST and SCT on the MRI console

### 3.1 - Enter the chroot environment
```
sudo chroot /opt/st/chroot /bin/bash
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
