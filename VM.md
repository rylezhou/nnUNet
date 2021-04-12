# Actual steps to create VM:

0. install anaconda  
`
apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
`
download the Anaconda installer for Linux
`bash Anaconda*.sh`
1. create conda env
2. install nnunet
`pip install nnunet`
3. install cudnn
`conda install -c conda-forge cudnn`

# VM tests
## preprocess
nnUNet_plan_and_preprocess  > 128GB RAM  

## train
### file handling(np.save, np.load)
**npy**: the standard binary file format in NumPy for persisting a single arbitrary NumPy array on disk. ... The format is designed to be as simple as possible while achieving its limited goals.  
**npz**: simple way to combine multiple arrays into a single file, one can use ZipFile to contain multiple “.npy” files. We recommend using the file extension “.npz” for these archives.
The main advantage is that the arrays are lazy loaded. That is, if you have an npz file with 100 arrays you can load the file without actually loading any of the data. If you request a single array, only the data for that array is loaded.
A downside to npz files is they can't be memory mapped (using load(<file>, mmap_mode='r')), so for large arrays they may not be the best choice.
