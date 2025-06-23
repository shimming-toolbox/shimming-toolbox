#!/bin/bash
# Adapted from Kelvin Chow's python-ismrmrd-server repository
# Source: https://github.com/kspaceKelvin/python-ismrmrd-server

# This script takes a Docker container export (.tar) and creates a chroot image (.img)
# Note that root privileges are required to mount the loopback images

EXPORT_FILE=${1}
CHROOT_FILE=${2}
BUFFER_MB=${3:-50}

# Files have a minimum storage size of 4k due to block size
exportSize=$(tar -tvf "${EXPORT_FILE}" | awk '{s+=int($3/4096+0.99999)*4096} END{printf "%.0f\n", s}')

# Add a minimum buffer of free space to account for filesystem overhead
chrootMinSize=$(( exportSize/(1024*1024) * 115/100 + ${BUFFER_MB}))

# Round up to the nearest 100 MB
chrootSize=$(( ((${chrootMinSize%.*})/100+1)*100 ))

echo ----------------------------------------------------------------------
echo Total size of files from Docker image is $(( exportSize/(1024*1024) )) MB with ${BUFFER_MB} MB of buffer
echo Creating chroot file ${CHROOT_FILE} of size ${chrootSize} MB
echo ----------------------------------------------------------------------

if test -f "${CHROOT_FILE}"; then
    echo "Warning -- ${CHROOT_FILE} exists and will be overwritten!"
    rm ${CHROOT_FILE}
fi

# Create blank ext3 chroot image
dd if=/dev/zero of=${CHROOT_FILE} bs=1M count=${chrootSize}
mke2fs -F -t ext3 ${CHROOT_FILE}

# Mount image and copy contents from tar export
echo Copying files to chroot image -- please wait...
mkdir /mnt/chroot
mount -o loop ${CHROOT_FILE} /mnt/chroot
tar -xf ${EXPORT_FILE} --directory=/mnt/chroot --totals

# Show the amount of free space left on the chroot
df -h

umount /mnt/chroot

echo ----------------------------------------------------------------------
echo Finished!  Verify that no errors have occured and that available space
echo on the last row of the above df output is greater than 100 MB
echo ----------------------------------------------------------------------
