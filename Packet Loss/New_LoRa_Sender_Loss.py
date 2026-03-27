# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import time
import busio
import board
import adafruit_rfm9x
import digitalio
import adafruit_ina219
from PIL import Image
import os

# -----------------------------
# MODEL
# -----------------------------
LATENT_DIM = 64

class ImprovedVAE(nn.Module):
    def __init__(self):
        super(ImprovedVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(128, LATENT_DIM)
        self.var_layer  = nn.Linear(128, LATENT_DIM)
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        return self.mean_layer(x), self.var_layer(x)

    def reparameterise(self, mean, var):
        return mean + var * torch.randn_like(var)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, var = self.encode(x)
        z = self.reparameterise(mean, var)
        return self.decode(z), mean, var


# -----------------------------
# PATHS
# -----------------------------
MODEL_PATH = '/home/karan/Desktop/New Improved MNIST VAE model/vae_model_improved.pth'
IMAGE_PATH = '/home/karan/Desktop/Image_Disseratation/test_image.png'

print("Checking files...")
print("Model exists:", os.path.exists(MODEL_PATH))
print("Image exists:", os.path.exists(IMAGE_PATH))

print("Loading improved VAE model...")
model = ImprovedVAE()
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()
print("Model loaded OK")

# -----------------------------
# INA219
# -----------------------------
i2c = busio.I2C(board.SCL, board.SDA)
ina = adafruit_ina219.INA219(i2c)
print("INA219 ready!")
print("Idle voltage: " + str(round(ina.bus_voltage, 2)) + "V")
print("Idle current: " + str(round(ina.current, 2)) + "mA")
print("Idle power:   " + str(round(ina.power, 2)) + "mW")

# -----------------------------
# LORA
# -----------------------------
spi = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)
cs  = digitalio.DigitalInOut(board.CE1)
rst = digitalio.DigitalInOut(board.D25)
rfm = adafruit_rfm9x.RFM9x(spi, cs, rst, 433.0)

rfm.tx_power = 23
rfm.signal_bandwidth = 125000
rfm.coding_rate = 5
rfm.spreading_factor = 7
rfm.enable_crc = True

print("LoRa ready")




# -----------------------------
# FUNCTIONS
# -----------------------------
def prepare_image(path):
    img = Image.open(path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)
    return tensor

def encode_image(tensor):
    with torch.no_grad():
        t_start = time.time()
        p_before = ina.power
        mean, var = model.encode(tensor)
        z = model.reparameterise(mean, var)
        p_after = ina.power
        t_end = time.time()

    encode_time = (t_end - t_start) * 1000
    encode_power = (p_before + p_after) / 2

    print("Encoded in " + str(round(encode_time, 1)) + "ms")
    print("Encode power: " + str(round(encode_power, 2)) + "mW")

    return z.numpy().flatten(), encode_time, encode_power

def split_packets(latent_vector):
    payload = latent_vector.astype(np.float32).tobytes()
    packets = []

    for i in range(0, len(payload), 48):
        packets.append(payload[i:i+48])

    print("Payload size: " + str(len(payload)) + " bytes")
    print("Split into " + str(len(packets)) + " packets")
    return packets

def send_packets(test_num, packets, drop_list=None):
    if drop_list is None:
        drop_list = []

    print("Sending TEST " + str(test_num) + " | " + str(len(packets)) + " packets | dropping: " + str(drop_list))

    t_start = time.time()
    p_before = ina.power

    sent_count = 0
    dropped_count = 0

    for i, packet in enumerate(packets):
        packet_num_human = i + 1

        if packet_num_human in drop_list:
            print("  Packet " + str(packet_num_human) + "/" + str(len(packets)) + " DROPPED (simulated loss)")
            dropped_count += 1
        else:
            # Header = [test_num, packet_index, total_packets]
            header = bytes([test_num, i, len(packets)])
            rfm.send(header + packet)
            print("  Sent packet " + str(packet_num_human) + "/" + str(len(packets)))
            sent_count += 1

        time.sleep(0.2)

    p_after = ina.power
    t_end = time.time()

    tx_time = (t_end - t_start) * 1000
    tx_power = (p_before + p_after) / 2

    print("Sent " + str(sent_count) + "/" + str(len(packets)) + " packets")
    print("Dropped " + str(dropped_count) + "/" + str(len(packets)) + " packets")
    print("TX time: " + str(round(tx_time, 1)) + "ms")
    print("TX power: " + str(round(tx_power, 2)) + "mW")

    return tx_time, tx_power, sent_count, dropped_count

def run_test(test_num, drop_list, image_path, label=""):
    print("")
    print("==========================================")
    print("TEST " + str(test_num) + " - " + label)
    print("==========================================")

    tensor = prepare_image(image_path)
    latent, enc_time, enc_power = encode_image(tensor)
    packets = split_packets(latent)
    tx_time, tx_power, sent, dropped = send_packets(test_num, packets, drop_list=drop_list)

    print("Test " + str(test_num) + " done. Waiting 8 seconds before next test...")
    time.sleep(8)

    return enc_time, enc_power, tx_time, tx_power, sent, dropped, len(packets)
    
    
    

# -----------------------------
# MAIN
# -----------------------------
print("")
print("Starting 6-packet loss tests...")
print("Make sure receiver is running on Pi B!")
print("Starting in 10 seconds...")
time.sleep(10)

results = []

tests = [
    (1, [],        "0% loss - send all 6 packets"),
    (2, [3],       "1 packet loss - drop packet 3"),
    (3, [2, 5],    "2 packet loss - drop packets 2 and 5"),
    (4, [1, 3, 6], "3 packet loss - drop packets 1, 3 and 6"),
]

for test_num, drop_list, label in tests:
    enc_time, enc_power, tx_time, tx_power, sent, dropped, total_packets = run_test(
        test_num, drop_list, IMAGE_PATH, label
    )

    results.append({
        'test': test_num,
        'label': label,
        'drop_list': drop_list,
        'sent': sent,
        'dropped': dropped,
        'total_packets': total_packets,
        'enc_time': enc_time,
        'enc_power': enc_power,
        'tx_time': tx_time,
        'tx_power': tx_power
    })

print("")
print("==========================================")
print("ALL TESTS COMPLETE - SUMMARY")
print("==========================================")

for r in results:
    print(
        "Test " + str(r['test']) +
        " | " + r['label'] +
        " | Sent: " + str(r['sent']) + "/" + str(r['total_packets']) +
        " | Dropped: " + str(r['dropped']) + "/" + str(r['total_packets']) +
        " | Encode: " + str(round(r['enc_time'], 1)) + "ms" +
        " | TX: " + str(round(r['tx_time'], 1)) + "ms"
    )

print("Check receiver folder for SSIM scores and reconstructed images.")
