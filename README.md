# AlignCell

A deep learning model designed to integrate sequencing data across platforms, omics, and species by learning the biological features across various omics platforms, enabling the discovery of key factors. AlignCell uses the Performer encoder and a triplet loss function within a Triplet network to reduce non-biological factors from multi-source omics data and accurately capture biological features, facilitating data integration.

<img width="1560" height="780" alt="image" src="https://github.com/user-attachments/assets/a8c0a5bc-0d27-4e14-a794-574b99b4430f" />

Developed based on ```torch 2.6.0```, the test folder contains both the pre-trained and trained models, as well as the running script. After downloading the test data and placing it in the ```test/example``` folder, it can be executed.

After downloading, navigate to the root directory of AlignCell. You can first create a new environment using  ```conda create -n AlignCell_env -f environment.yaml``` and then activate it with ```conda activate AlignCell_env```. These steps can also be performed with `mamba` as a faster alternative if it is installed. Next, run  ```sh pip install -r requirements.txt``` to install the dependencies, and then run ```sh pip install -e .``` to complete the installation.

The data for ```test/example``` can be obtained from ``` https://doi.org/10.5281/zenodo.15285426```.


## Citation

If you use this resource, please cite:

Du Z, Zhang X, Zhang Q, Li J, Cao Z, Gao G, Lin T, Wang D, Gao S.  
**Multisource omic alignment and biological feature discovery with Performer encoder and triplet networks.**  
*Genome Research*. 2026 Jul 22.  
doi: [10.1101/gr.281629.125](https://doi.org/10.1101/gr.281629.125)
