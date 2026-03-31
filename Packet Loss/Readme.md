# Dissertation_LoRa_Image_Transfer

## Enhancing Image Transmission over LoRa Networks Using Adaptive AI Compression — SF7 Packet Loss Module



This repository contains the SF7 packet loss transmission code for the VAE-based LoRa image transmission system. This module tests reconstruction quality under simulated packet loss conditions using the Improved VAE model with 6-packet transmission.

---

## Overview

This project implements an improved VAE-based image compression and transmission system over LoRa networks using two Raspberry Pi 4 nodes. The system also includes packet loss testing and power measurement using INA219.

- **Pi A (Sender)** — encodes image using Improved VAE, splits into 6 packets, transmits over LoRa with optional packet loss simulation
- **Pi B (Receiver)** — receives packets, reconstructs missing data, decodes image, calculates SSIM and saves results

---

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

### Pi B (Receiver) — New_LoRa_Receiver_Loss.py

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
        New_LoRa_Sender_Loss.py       <- Updated sender (6 packets + test ID)
        New_LoRa_Receiver_Loss.py     <- Updated receiver (smart waiting + SSIM)
        Packet_loss_Sender.py         <- Old sender (3 packets)
        Packet_loss_Receiver.py       <- Old receiver
        README.md                     <- This file

    New Improved MNIST VAE model/
        improved_vae_model.ipynb
        vae_model_improved.pth

    MNIST Autoencoder/
    Kaggle Autoencoder/
```

---

## LoRa Settings (SF7)

| Parameter | Value | Reason |
|-----------|-------|--------|
| Spreading Factor | SF7 | Standard range |
| Bandwidth | 125 kHz | Standard bandwidth |
| Coding Rate | 5 | Standard error correction |
| TX Power | 23 dBm | Maximum power |
| Frequency | 433 MHz | Unlicensed band |
| CRC | Enabled | Error detection |

```python
rfm.tx_power = 23
rfm.signal_bandwidth = 125000
rfm.coding_rate = 5
rfm.spreading_factor = 7
rfm.enable_crc = True
frequency = 433.0
```


---

## Hardware Setup

| Component | Specification | Node |
|-----------|---------------|------|
| Raspberry Pi 4 Model B | 1.8GHz quad-core, 4GB RAM | Both |
| RFM9x LoRa Module | 433MHz, SX1276 chip | Both |
| INA219 Power Sensor | I2C, 12-bit ADC, address 0x40 | Both |

**Pi A (Sender):** username = karan, Python 3.11
**Pi B (Receiver):** username = ysj, Python 3.13

---

## Wiring — Both Nodes

### RFM9x LoRa

| RFM9x Pin | Pi Pin | GPIO | Function |
|-----------|--------|------|----------|
| VCC | Pin 1 | 3.3V | Power |
| GND | Pin 6 | GND | Ground |
| MOSI | Pin 19 | GPIO10 | SPI data out |
| MISO | Pin 21 | GPIO9 | SPI data in |
| SCK | Pin 23 | GPIO11 | SPI clock |
| CS | Pin 26 | CE1 (GPIO7) | Chip select — must be CE1 not CE0 |
| RST | Pin 22 | GPIO25 | Reset |
| DIO0 | Pin 18 | GPIO24 | Interrupt |

### INA219 Power Sensor

| INA219 Pin | Pi Pin | Function |
|------------|--------|----------|
| VCC | Pin 1 | Power |
| GND | Pin 9 | Ground |
| SDA | Pin 3 | I2C data |
| SCL | Pin 5 | I2C clock |
| Vin+ | RFM9x VCC | Measures LoRa module power |
| Vin- | Pin 17 | Reference voltage |

---

## Software Installation — Both Pis

```bash
pip3 install torch==2.6.0 torchvision==0.21.0 --break-system-packages
pip3 install adafruit-circuitpython-rfm9x --break-system-packages
pip3 install adafruit-circuitpython-ina219 --break-system-packages
pip3 install scikit-image pillow numpy --break-system-packages
```

> Use torch==2.6.0 exactly — newer versions cause Illegal Instruction error on Pi 4.

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
scp vae_model_improved.pth karan@$PI_A:/home/karan/Desktop/New\ Improved\ MNIST\ VAE\ model/
scp vae_model_improved.pth ysj@$PI_B:/home/ysj/Desktop/New\ Improved\ MNIST\ VAE\ model/

# Copy test image to receiver (Pi B)
scp test_image.png ysj@$PI_B:/home/ysj/Image_dissertation/

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
img_pil.save('/home/karan/Desktop/Image_dissertation/test_image.png')
print('Saved MNIST digit: ' + str(label))
"
```

Copy the same image to Pi B:
```bash
scp /home/karan/Desktop/Image_dissertation/test_image.png ysj@$PI_B:/home/ysj/Image_dissertation/test_image.png
```

---

## Running — Standard Single Test

```bash
# STEP 1 — Start receiver on Pi B FIRST
python3 /home/ysj/receiver.py
# Wait for: 'Waiting for transmission...'

# STEP 2 — Run sender on Pi A within 30 seconds
python3 /home/karan/sender.py
```

---

## Running — Packet Loss Tests

```bash
# STEP 1 — Start receiver on Pi B FIRST
python3 /home/ysj/New_LoRa_Receiver_Loss.py
# Wait for: 'Waiting for TEST 1...'

# STEP 2 — Run sender on Pi A
python3 /home/karan/New_LoRa_Sender_Loss.py
```

Three automatic tests run sequentially:

| Test | Loss Rate | Packets Dropped |
|------|-----------|-----------------|
| Test 1 | 0% loss | None — all 6 sent |
| Test 2 | 33% loss | Packet 2 dropped |
| Test 3 | 66% loss | Packets 1 and 2 dropped |

---

## Output Images

```
/home/ysj/Packet loss Image Improved/
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
- **Left side** — Original MNIST image
- **Right side** — Reconstructed image after LoRa transmission
- **Top banner** — Test number, loss rate, packets received, SSIM score
- **Red text** — Which packets were lost (if any)

---



## Related Repositories

- **SF12 Long Range System:** [Dissertation_LoRa_Image_Transfer_Long_Distance](https://github.com/karanchawla1108/Dissertation_LoRa_Image_Transfer_long_Distance)
- **VAE Model Training:** [Autoencoder-IoT-LoRa-Dissertation](https://github.com/karanchawla1108/Autoencoder-IoT-LoRa-Dissertation)

---

## References

- Kingma, D.P. and Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv:1312.6114
- LeCun, Y. et al. (1998). Gradient-Based Learning Applied to Document Recognition
- Wang, Z. et al. (2004). Image Quality Assessment: SSIM. IEEE Transactions on Image Processing
- Jebril, A. et al. (2018). Overcoming Limitations of LoRa Physical Layer in Image Transmission

