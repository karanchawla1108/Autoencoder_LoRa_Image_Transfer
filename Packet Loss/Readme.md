## Overview
 
This project implements a VAE-based image compression and transmission system over LoRa networks using two Raspberry Pi 4 nodes.
 
- **Pi A (Sender)** — encodes image using VAE, transmits over LoRa
- **Pi B (Receiver)** — receives packets, decodes image, calculates SSIM
 
---
 
## PATHS YOU MUST CHANGE
 
These paths are different on every setup. Change them before running any script.
 
### Pi A (Sender) — sender.py and sender_loss_test.py
 
```python
# LINE 1 — Model path (change to where your .pth file is)
model.load_state_dict(torch.load(
    '/home/karan/Desktop/AutoEncoders/MNIST/vae_model (3).pth',
    #  CHANGE THIS to your actual model path on Pi A
    map_location='cpu'
))
 
# LINE 2 — Test image path
image_path = '/home/karan/test_image.png'
#             CHANGE THIS to your actual image path on Pi A
```
 
### Pi B (Receiver) — receiver.py and receiver_loss_test.py
 
```python
# LINE 1 — Model path on Pi B
model.load_state_dict(torch.load(
    '/home/ysj/Desktop/AutoEncoders/MNIST/vae_model (3).pth',
    #  CHANGE THIS to your actual model path on Pi B
    map_location='cpu'
))
 
# LINE 2 — Original image path for SSIM comparison
Image.open('/home/ysj/test_image.png')
#          CHANGE THIS to where you copied the test image on Pi B
#          Note: may be .jpg or .png depending on how you copied it
 
# LINE 3 — Output folder for saved images
base_folder = '/home/ysj/Packet loss Image'
#              CHANGE THIS if you want images saved somewhere else
```
 
---
 
## File Structure
 
```
dissertation-vae/
    sender.py                    <- Basic sender (single transmission)
    sender_loss_test.py          <- Sender with 3 packet loss tests
    receiver.py                  <- Basic receiver (single transmission)
    receiver_loss_test.py        <- Receiver with 3 tests + comparison images
    vae_model (3).pth            <- Trained VAE model weights
    Training_model.ipynb         <- VAE training code (Google Colab)
    Training_model_extend.ipynb  <- Full experiment pipeline
    kaggle_Auto_encoder.ipynb    <- Kaggle baseline model code
    README.md                    <- This file
```
 

 
## Software Installation — Both Pis
 
```bash
pip3 install torch==2.6.0 torchvision==0.21.0 --break-system-packages
pip3 install adafruit-circuitpython-rfm9x --break-system-packages
pip3 install adafruit-circuitpython-ina219 --break-system-packages
pip3 install scikit-image pillow numpy --break-system-packages
```
 
Use torch==2.6.0 exactly — newer versions cause Illegal Instruction error on Pi 4.
 
### Enable Interfaces — Both Pis
 
```bash
sudo raspi-config
# Interface Options -> SPI -> Enable
# Interface Options -> I2C -> Enable
# Reboot
```
 
---
 
## Copy Files to Both Pis
 
```bash
# Copy model to Pi A
scp 'vae_model (3).pth' karan@[PI-A-IP]:/home/karan/Desktop/AutoEncoders/MNIST/
 
# Copy model to Pi B
scp 'vae_model (3).pth' ysj@[PI-B-IP]:/home/ysj/Desktop/AutoEncoders/MNIST/
 
# Copy test image to Pi B
scp /home/karan/test_image.png ysj@[PI-B-IP]:/home/ysj/test_image.png
 
# Copy receiver scripts to Pi B
scp receiver.py ysj@[PI-B-IP]:/home/ysj/
scp receiver_loss_test.py ysj@[PI-B-IP]:/home/ysj/
```
 
---
 
## Generate Test Image (Pi A only)
 
```bash
python3 -c "
import torchvision
import torchvision.transforms as transforms
from PIL import Image
dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True,
    transform=transforms.ToTensor()
)
img, label = dataset[0]
img_pil = Image.fromarray(
    (img.squeeze().numpy() * 255).astype('uint8')
)
img_pil.save('/home/karan/test_image.png')
print('Saved MNIST digit: ' + str(label))
"
```
 
---
 
## Running — Basic Single Test
 
```bash
# STEP 1 — Start receiver on Pi B FIRST
python3 /home/ysj/receiver.py
 
# STEP 2 — Run sender on Pi A within 30 seconds
python3 /home/karan/sender.py
```
 
---
 
## Running — Packet Loss Tests
 
```bash
# STEP 1 — Start receiver on Pi B FIRST
python3 /home/ysj/receiver_loss_test.py
 
# STEP 2 — Run sender on Pi A
python3 /home/karan/sender_loss_test.py
```
 
Three automatic tests run:
- Test 1 — 0% packet loss (all 3 packets sent)
- Test 2 — 33% packet loss (~1 packet dropped)
- Test 3 — 66% packet loss (~2 packets dropped)
 
---
 
## Output Images
 
Images are saved automatically in:
 
```
/home/ysj/Packet loss Image/
    run1/
        test1_0pct_loss_reconstructed.png
        test1_0pct_loss_comparison.png
        test2_33pct_loss_reconstructed.png
        test2_33pct_loss_comparison.png
        test3_66pct_loss_reconstructed.png
        test3_66pct_loss_comparison.png
    run2/
        ...
```
 
Each run creates a new numbered folder automatically.
 
The comparison image shows:
- Left side — Original MNIST image
- Right side — Reconstructed image after LoRa transmission
- Top banner — Test number, loss rate, packets received, SSIM score
- Red text — Which packets were lost (if any)
 
 
## LoRa Settings Used
 
```python
rfm.tx_power = 23
rfm.signal_bandwidth = 125000
rfm.coding_rate = 5
rfm.spreading_factor = 7
rfm.enable_crc = True
frequency = 433.0
```
 
---
 
## References
 
- Kingma, D.P. and Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv:1312.6114
- LeCun, Y. et al. (1998). Gradient-Based Learning Applied to Document Recognition
- Wang, Z. et al. (2004). Image Quality Assessment: SSIM. IEEE Transactions on Image Processing
- Jebril, A. et al. (2018). Overcoming Limitations of LoRa Physical Layer in Image Transmission
