## Instrucions
1) To use a specific file copy the file to the default settings location:

        * For Windows it's in 'Documents\Airsim\'
        * For Linux it's in '~Documents/Airsim/'

2) Next rename the file to 'settings.json' (make sure to back up your original file)

## Caution
* The settings files are provided because the code in 'car_env<#>.py' has been modified to not require additional image reshaping for performance optimization.
* Ensure the image shapes match the ones in the settings file otherwise you will get errors.
* Finally, depending on the image size selected a different about of ram will be required.
  * Example: 'image_size = 320 x 180' with 500000(5e5) timesteps
    * Will require ~28GB of ram
  * For lower spec PC stick to 'image_size = 128 x 128' or lower with 500000(5e5) timesteps
    * This will require ~8GB of ram
