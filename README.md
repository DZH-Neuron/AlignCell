# AlignCell

A deep learning model designed to integrate sequencing data across platforms, omics, and species by learning the biological features across various omics platforms, enabling the discovery of key factors. AlignCell uses the Performer encoder and a triplet loss function within a Triplet network to reduce non-biological factors from multi-source omics data and accurately capture biological features, facilitating data integration.

![image](https://github.com/user-attachments/assets/27ec253c-eee0-407a-b197-b25c1737def4)

Developed based on ```torch 2.6.0```, the test folder contains both the pre-trained and trained models, as well as the running script. After downloading the test data and placing it in the ```test/example``` folder, it can be executed.

After downloading, switch to the first AlignCell directory, first run  ```sh pip install -r requirements``` to install the dependencies, and then run ```sh pip install -e .``` to complete the installation.

The data for ```test/example``` can be obtained from ```https://zenodo.org/uploads/15285426```.
