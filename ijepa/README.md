## Embedding Predictive Pre-Training

This module sets up the embedding predictive pre-training pipeline based on [i-JEPA](https://github.com/facebookresearch/ijepa).

### Setup Instructions

1. **Clone the i-JEPA Repository**  
   First, clone the official i-JEPA repository and follow its instructions to set up the Python environment:

   ```bash
   git clone https://github.com/facebookresearch/ijepa.git
   cd ijepa
   # Follow environment setup instructions from their README
   ```

2. **Integrate Modified Components**  
   Copy the contents of this directory (i.e., the current folder you're in) into the corresponding directories within the cloned i-JEPA repository.
   If prompted to overwrite files, replace them with the versions from this directory.

3. **Configure the Training Script**  
   Open `configs/dream_vitb16_ep200.yaml` and update the following:
   - Provide the path to the **Dream** dataset.
   - Set the path to your **i-JEPA** directory as prompted in the config file.

4. **Run Training**  
   From the i-JEPA root directory, start the training using:
   ```bash
   python main.py \
     --fname configs/dream_vitb16_ep200.yaml \
     --devices cuda:0 cuda:1 cuda:2 cuda:3
    ```

5. **Use for Downstream Tasks**  
    Once pre-training is complete, return to the RoboPEPP project and use the trained model for downstream tasks, as outlined in the main RoboPEPP README.

The pre-trained weights can also be downloaded from [here]().