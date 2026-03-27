
## Overview


 
This project implements an improved VAE-based image compression and transmission system over LoRa networks using two Raspberry Pi 4 nodes. The system also includes packet loss testing and power measurement using INA219.
 
- Pi A (Sender) — encodes image using Improved VAE, splits into 6 packets, transmits over LoRa with optional packet loss
- Pi B (Receiver) — receives packets, reconstructs missing data, decodes image, calculates SSIM and saves results
 

 
## PATHS YOU MUST CHANGE
 
These paths are different on every setup. Change them before running any script.
 
### Pi A (Sender) — New_LoRa_Sender_Loss.py
 
```python
# LINE 1 — Model path (Improved VAE model)
model.load_state_dict(torch.load(
    '/home/karan/Desktop/New Improved MNIST VAE model/vae_model_improved.pth',
    # CHANGE THIS to your actual model path on Pi A
    map_location='cpu'
))
 
# LINE 2 — Test image path
IMAGE_PATH = '/home/karan/Desktop/Image_dissertation/test_image.png'
# CHANGE THIS to your actual image path on Pi A



```
 
### Pi B (Receiver) — receiver.py and receiver_loss_test.py
 
```python
# LINE 1 — Model path on Pi B
model.load_state_dict(torch.load(
    '/home/ysj/Desktop/New Improved MNIST VAE model/vae_model_improved.pth',
    # CHANGE THIS to your actual model path on Pi B
    map_location='cpu'
))
 
# LINE 2 — Original image path for SSIM comparison
ORIGINAL_IMAGE_PATH = '/home/ysj/Image_dissertation/test_image.png'
# CHANGE THIS to where your image is stored on Pi B
 
# LINE 3 — Output folder
BASE_FOLDER = '/home/ysj/Packet loss Image Improved'
# CHANGE THIS if you want images saved somewhere else
```
 
---
 
## File Structure
 
```
Dissertation_LoRa_Image_Transfer/
    Packet Loss/
        New_LoRa_Sender_Loss.py      <- Updated sender (6 packets + test ID)
        New_LoRa_Receiver_Loss.py    <- Updated receiver (smart waiting + SSIM)
        Packet_loss_Sender.py        <- Old sender (3 packets)
        Packet_loss_Receiver.py      <- Old receiver
        Readme.md                    <- This file
 
    New Improved MNIST VAE model/
        improved_vae_model.ipynb
        vae_model_improved.pth
 
    MNIST Autoencoder/
    Kaggle Autoencoder/
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
# Copy model to both Pis
scp vae_model_improved.pth karan@$PI_A:/home/karan/
scp vae_model_improved.pth ysj@$PI_B:/home/ysj/

# Copy test image to receiver (Pi B)
scp test_image.png ysj@$PI_B:/home/ysj/

# Copy receiver scripts to Pi B
scp New_LoRa_Receiver_Loss.py ysj@$PI_B:/home/ysj/

# Copy sender script to Pi A
scp New_LoRa_Sender_Loss.py karan@$PI_A:/home/karan/
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
python3 /home/ysj/New_LoRa_Receiver_Loss.py
 
# STEP 2 — Run sender on Pi A
python3 /home/karan/New_LoRa_Sender_Loss.py
```
 
Four automatic tests:

Test 1 — 0% packet loss (all 6 packets sent)
Test 2 — 1 packet loss
Test 3 — 2 packet loss
Test 4 — 3 packet loss

---
 
## Output Images
 
Images are saved automatically in:
 
```
/home/ysj/Packet loss Image Improved/
    run1/
        test1_0pct_loss_reconstructed.png
        test1_0pct_loss_comparison.png
        test2_1packet_loss_reconstructed.png
        test2_1packet_loss_comparison.png
        ...
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
