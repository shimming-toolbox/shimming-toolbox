---
orphan: true
---

# All about constraint files

## What is in the scanner constraint file?
The scanner constraint file is a JSON file that lists the different parameters used by Shimming Toolbox
to generate a scanner-specific shim solution that respects the constraints of the scanner.

The file contains the following fields:
- `name`: The name of the scanner. This will be used when creating the name of the output shim file.
- `coef_channel_minmax`: A dictionary that lists the minimum and maximum values for each shim channel.
The keys of the dictionary are the order of the shim coefficients (`0` for spherical harmonic order 0 (frequency),
`1` for order 1 (X, Y, Z), `2` for order 2 (Z2, ZX, ZY, X2 - Y2, XY) and `3` for order 3 (Z3, Z2X, Z2Y, Z(X2 - Y2), XYZ, X(X2 - Y2), Y(X2 - Y2)).
When using Siemens 3rd order shim (which only have 4 channels), use a list of 4 channels in the order described above.
- `coef_sum_max`: Number specifying the maximum allowed sum of the absolute values of the shim coefficients.
This is typically not a concern for scanner constraints, you can write `null` to disregard the constraint.
- `coefs_used`: A dictionary to specify the shim value used when acquiring the B0 map. These values are specified in
the same order as `coef_channel_minmax`. Use `null` instead of a list of values to let Shimming Toolbox read the
metadata of the B0 map to figure out these values.

Here is an example of a scanner constraint file for a Siemens MAGNETOM Prisma Fit.:
```
{
    "name": "MAGNETOM Prisma Fit",
    "coef_channel_minmax":
    {
        "0": [[123100100, 123265000]],
        "1": [[-2300, 2300],
              [-2300, 2300],
              [-2300, 2300]],
        "2": [[-4959.01, 4959.01],
              [-3551.29, 3551.29],
              [-3503.299, 3503.299],
              [-3551.29, 3551.29],
              [-3487.302, 3487.302]]
    },
    "coef_sum_max": null,
    "coefs_used":
    {
        "0": [0],
        "1": [0, 0, 0],
        "2": [0, 0, 0, 0, 0]
    }
}
```

## Do I need to use a scanner constraint file?
To generate a scanner-specific shim solution that respects the constraints of a scanner, Shimming Toolbox needs to know
different parameters of the scanner (i.e.: the minimum and maximum shim coefficient for each channel, the current shim coefficients).
There are 2 ways for Shimming Toolbox to know the scanner constraints:
1. We store the shim constraints of specific scanners internally (no need for a scanner constraint file).
2. Provide these parameters in a file using the `--scanner-coil-constraints` argument when running `st_b0shim dynamic`. This file will overwrite the internal scanner constraints if provided.

## Option 1: How can Shimming Toolbox know the current shim values and scanners constraints automatically (recommended)
The benefit of this option is that you don't need to fill the current shim coefficients (`coefs_used`) yourself everytime you shim. You also don't need to add the `--scanner-coil-constraints` option to the command.
This reduces the risk of human error and saves you time at the scanner.
### Siemens
Since Siemens writes the shim coefficient of orders 0, 1 and 2 in the DICOM metadata, they are also available in the BIDS
sidecar which Shimming Toolbox can access. The values of orders 1 and 2 are stored using DAC units instead of uT/m and uT/m^2 which needs
scanner specific conversion factors. These conversion factors can be calculated from running both of these commands in the terminal on the scanner:

```
AdjValidate -shim -info -mp
AdjValidate -shim -info
```

The order 0 minimum and maximum value can be found in the manual frequency adjustment tab when a protocol is loaded on the scanner.

To add your scanner to the internal list of supported scanners, you can send us, by opening an [issue](https://github.com/shimming-toolbox/shimming-toolbox/issues),
the output of both `AdjValidate` commands, the order 0 (frequency) minimum and maximum values, and a BIDS JSON
sidecar of an acquisition on your scanner (to identify your scanner's name and unique number).

### Philips
Philips does not store the shim coefficients (orders 1, 2 and 3) in the DICOM metadata, it is therefore not possible to automatically read the current shim values automatically from a DICOM to NIfTI conversion.
We have developped a patch to improve Shimming Toolbox's compatibility with Philips scanners. The patch automatically generates a constraint file for each acquisition.
You can contact us to get this patch and instructions on how to use it.

### GE
GE only stores the orders 0 and 1 shim coefficients in the DICOM metadata. Contact us by opening an [issue](https://github.com/shimming-toolbox/shimming-toolbox/issues)
if you want to use Shimming Toolbox with a GE scanner so we can figure out the best way to read the current shim values and store the scanner's constraints.

### Other manufacturers
For other manufacturers, please reach out to us so we can find how to read the current shim values and store the scanner's constraints.

## Option 2: How do I find the values to fill in the scanner constraint file?
### Siemens
The order 0 minimum and maximum value can be found in the manual frequency adjustment tab when a protocol is loaded on the scanner.

Orders 1 and 2 can be found by using the following command in the terminal on the scanner:

```
AdjValidate -shim -info -mp
```

The output will be in the same order as `coef_channel_minmax`.

Orders 0 will automatically be read from the B0 map metadata, `coef_channel_minmax[0]` should therefore be `null`.
Orders 1, 2 and 3 can be found in the manual shim adjustment tab. The coefficent values to fill in are in the same units as that tab (uT/m, uT/m^2, uT/m^3).

### Philips
We recommend contacting us when using Shimming Toolbox with Philips scanners as we have made a patch to improve
Shimming Toolbox's compatibility with Philips scanners. The patch automatically generates a constraint file for each acquisition.

### Other manufacturers
For other scanners, please reach out to us so we can help figure out the correct values to fill in.

## The custom coil constraint file
The coil constraint file is a JSON file that lists the different parameters used by Shimming Toolbox to respect the custom coils contraints.
Here are the different fields in the coil constraint file:
- `name`: The name of the coil. This will be used when creating the name of the output shim file.
- `coef_channel_minmax`: A dictionary that lists the minimum and maximum values for each shim channel.
We don't enforce units but your coil profiles need to be consistent (if your coil profiles are in Hz/A, then these
values are in A, if your coil profiles are in Hz/mA, then these values should be in mA).
Use `null` to set the limit to infinity.
- `coef_sum_max`: Number specifying the maximum allowed sum of the absolute values of the shim coefficients. Use `null` to set the limit to infinity.
- `coefs_used`: A dictionary to specify the shim value used when acquiring the B0 map. These values are specified in
the same order as `coef_channel_minmax`. These will typically be 0.
- `Units`: Used for display purposes only and does not affect any shim current output.

```
{
    "name": "custom",
    "coef_channel_minmax":
    {
        "coil": [[-2.5, 2.5],
                 [-2.5, 2.5],
                 [-2.5, 2.5],
                 [-2.5, 2.5],
                 [-2.5, 2.5],
                 [-2.5, 2.5],
                 [-2.5, 2.5],
                 [-2.5, 2.5],
                 [-2.5, 2.5]]
    },
    "coef_sum_max": 20,
    "coefs_used":
    {
        "coil": [0, 0, 0, 0, 0, 0, 0, 0, 0]
    },
    "Units": "A"
}
```

This file gets populated with the correct values when running `st_create_coil_profiles`. For more information, see the {ref}`create_b0_coil_profiles` tutorial.

## Which shim algorithms support constraints?
The following algorithms support scanner constraints (--optimizer-method):
- `least_squares`: Minimum and maximum bounds (coef_channel_minmax), and maximum sum of absolute values (coef_sum_max)
- `quad_prog`: Minimum and maximum bounds (coef_channel_minmax), and maximum sum of absolute values (coef_sum_max)
- `bfgs`: Minimum and maximum bounds (coef_channel_minmax)

The following algorithms *do not* support scanner constraints:
- `pseudo_inverse`
