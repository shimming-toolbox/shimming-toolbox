# Creating a chroot disk image for Shimming Toolbox
This guide walks you through creating a disk image (.img) containing a complete Linux filesystem intended to be used as the root directory for a changed root (chroot) environment. The resulting image can be deployed on MARS system hardware to run Shimming Toolbox locally on the MRI scanner.

## Prerequisites

Before starting, ensure you have:
- **Docker**: Installed and running on your system
- **Disk space**: At least 10GB free space for the build process
- **USB drive**: 8GB+ capacity (recommended) for image transfer
- **Time**: Allow 30-40 minutes for the complete process
- **Permissions**: Root/sudo access may be required for some operations

## Overview

The process consists of three main steps:
1. Build a Docker image containing Shimming Toolbox
2. Export the container filesystem and create a chroot-compatible disk image
3. Compress and transfer the image to a USB drive for deployment

---

## Step 1 - Navigate to the ST Docker directory
First, locate your Shimming Toolbox installation and navigate to the Docker directory:
```bash
cd <path/to/ST>
cd Docker
```
Verify you're in the correct directory by checking for the required files:
```bash
ls -la
```
You should see `Dockerfile` and the build scripts (`docker_to_chroot.sh`, `tar_to_chroot.sh`).

---

## Step 2 - Create the chroot image
Make the build scripts executable and run the image creation process:
```bash
chmod +x ./docker_to_chroot.sh
chmod +x ./tar_to_chroot.sh
./docker_to_chroot.sh st_image st_chroot.img
```

### What happens during this step:
1. **Docker build**: Creates a Docker image containing Shimming Toolbox and dependencies
2. **Container export**: Exports the container's filesystem to a tarball
3. **Image creation**: Converts the tarball to a Linux root filesystem (.img)

> [!NOTE]
> **Expected duration:** 12-14 minutes
> **Output file:** `st_chroot.img` (approximately 9.5GB)

---

## Step 3 - Export the chroot image to your USB key

### 3.1 Prepare and verify
Connect your USB drive and identify its mount point:
```bash
# On macOS
df -h | grep /Volumes/

# On Linux
df -h | grep /media/
```

### 3.2 Create compressed archive

Compress the image to reduce transfer time and storage requirements:
```bash
zip st_chroot.zip st_chroot.img
```
> [!NOTE]
> **Output file:** `st_chroot.zip` (approximately 3.5GB)

### 3.3 Transfer to your USB drive

Replace `/Volumes/YOUR_USB_NAME/` with your actual USB drive path:
```bash
cp st_chroot.zip </Volumes/YOUR_USB_NAME/>
```
> [!NOTE]
> **Expected transfer time:** 2-10 minutes depending on USB drive speed (USB 3.0+ recommended)

### 3.4 Verify and cleanup

Verify the transfer completed successfully:
```bash
ls -la /Volumes/YOUR_USB_NAME/st_chroot.zip
```

Compare file sizes to ensure integrity:
```bash
ls -l st_chroot.zip /Volumes/YOUR_USB_NAME/st_chroot.zip
```

Only after successful verification, clean up local files:
```bash
rm st_chroot.zip
rm st_chroot.img
```

> [!WARNING]
> Don't delete the local files until you've verified the USB transfer completed successfully.

---

## File Size Reference

| File | Approximate Size |
|------|------------------|
| st_chroot.img | 9-10 GB |
| st_chroot.zip | 3-4 GB |
| Required USB space | 4+ GB (recommended 8GB+) |

---

## Next Steps

After completing this process:
1. Safely eject the USB drive
2. Transport to the MRI scanner system
3. Follow the deployment instructions for mounting the chroot environment on the MARS hardware
