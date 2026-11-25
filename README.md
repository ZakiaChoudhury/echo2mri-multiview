# Multi-view Echocardiography to Cardiac MRI Synthesis

Deep learning code for generating short-axis (PSAX) cardiac MRI from four echocardiography views (A2CH, A3CH, A4CH, PSAX).

This is part of my PhD research at the University of Otago, New Zealand.

## Method

1. A convolutional autoencoder is first trained on MRI (MRI → MRI).
2. Its decoder is reused as the MRI generator.
3. A CNN encoder maps the 4-view echo input (stacked as 4 channels, 128×128) to the decoder's latent space.
4. The full echo → MRI model is trained with a combined loss: 0.9 × MAE + 0.1 × image-gradient loss.

Test MAE is logged every epoch. Predicted images (`.png`) and arrays (`.npy`) are saved at regular intervals.

## Data layout

Patient data are **not** included in this repository.

```
data/
├── Echo_A2ch/        Echo_A2ch_Test/
├── Echo_A3ch/        Echo_A3ch_Test/
├── Echo_A4ch/        Echo_A4ch_Test/
├── Echo_PSAX/        Echo_PSAX_Test/
├── MRI_Train/        MRI_Test/
```

Each folder holds one `.npy` file per image. Files must be paired by sorted file name across all views and the MRI folder.

## Usage

```bash
pip install -r requirements.txt

python train.py \
    --data_dir data \
    --pretrained_ae path/to/pretrained_mri_ae.h5 \
    --out_dir results \
    --epochs 1000 --batch_size 64
```

## Related publications

- Choudhury, Z. Z., Dey, S., Wang, H., Coffey, S., Chakraborti, T., & McCane, B. (2026). Cross-modal image generation with uncertainty quantification from echocardiogram to MRI. *Methods*, 247, 1–11. https://doi.org/10.1016/j.ymeth.2025.12.012
- Choudhury, Z. Z., McCane, B., & Coffey, S. (2025). Echocardiogram to CMR image synthesis using generative models. *40th IVCNZ*. https://doi.org/10.1109/IVCNZ67716.2025.11281646
- Choudhury, Z. Z., McCane, B., & Coffey, S. (2024). Medical image synthesis using autoencoder with vision transformer. *39th IVCNZ*. https://doi.org/10.1109/IVCNZ64857.2024.10794465

## Contact

Zakia Zinat Choudhury — zakiaru01@gmail.com
