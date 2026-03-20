# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import time
import random
import busio
import board
import adafruit_rfm9x
import digitalio
from PIL import Image
 
class SimpleVAE(nn.Module):
    def __init__(self):
        super(SimpleVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(64, 32)
        self.var_layer  = nn.Linear(64, 32)
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
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
 
print("Loading VAE model...")
model = SimpleVAE()
model.load_state_dict(torch.load(
    '/home/karan/Desktop/AutoEncoders/MNIST/vae_model (3).pth',
    map_location='cpu'
))
model.eval()
print("Model loaded OK")
 
class FakeINA:
    power = 0.0
ina = FakeINA()
 
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
 
def prepare_image(path):
    img = Image.open(path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)
    return tensor
 
def encode_image(tensor):
    with torch.no_grad():
        t_start = time.time()
        mean, var = model.encode(tensor)
        z = model.reparameterise(mean, var)
        t_end = time.time()
    encode_time = (t_end - t_start) * 1000
    print("Encoded in " + str(round(encode_time, 1)) + "ms")
    return z.numpy().flatten(), encode_time
 
def split_packets(latent_vector):
    payload = latent_vector.astype(np.float32).tobytes()
    packets = []
    for i in range(0, len(payload), 48):
        packets.append(payload[i:i+48])
    return packets
 
def send_packets(packets, loss_rate=0.0):
    print("Sending " + str(len(packets)) + " packets at " + str(int(loss_rate*100)) + "% loss rate...")
    t_start = time.time()
    sent_count = 0
    dropped_count = 0
    for i, packet in enumerate(packets):
        if random.random() < loss_rate:
            print("  Packet " + str(i+1) + "/" + str(len(packets)) + " DROPPED")
            dropped_count += 1
        else:
            header = bytes([i, len(packets)])
            rfm.send(header + packet)
            print("  Sent packet " + str(i+1) + "/" + str(len(packets)))
            sent_count += 1
        time.sleep(0.2)
    t_end = time.time()
    tx_time = (t_end - t_start) * 1000
    print("Sent " + str(sent_count) + "/" + str(len(packets)) + " packets")
    print("Dropped " + str(dropped_count) + "/" + str(len(packets)) + " packets")
    print("TX time: " + str(round(tx_time, 1)) + "ms")
    return tx_time, sent_count, dropped_count
 
def run_test(test_num, loss_rate, image_path):
    print("")
    print("==========================================")
    print("TEST " + str(test_num) + "Loss Rate: " + str(int(loss_rate*100)) + "%")
    print("==========================================")
    tensor = prepare_image(image_path)
    latent, enc_time = encode_image(tensor)
    packets = split_packets(latent)
    tx_time, sent, dropped = send_packets(packets, loss_rate=loss_rate)
    print("Test " + str(test_num) + " done. Waiting 8 seconds for receiver...")
    time.sleep(8)
    return enc_time, tx_time, sent, dropped
    
    
    
image_path = '/home/karan/test_image.png' / Image path
print("")
print("Starting 3 packet loss tests...")
print("Make sure receiver is running on Pi B!")
print("Starting in 5 seconds...")
time.sleep(5)
 
results = []
tests = [
    (1, 0.0),
    (2, 0.33),
    (3, 0.66),
]
 
for test_num, loss_rate in tests:
    enc_time, tx_time, sent, dropped = run_test(test_num, loss_rate, image_path)
    results.append({
        'test': test_num,
        'loss_rate': int(loss_rate * 100),
        'sent': sent,
        'dropped': dropped,
        'enc_time': enc_time,
        'tx_time': tx_time
    })
 
print("")
print("==========================================")
print("ALL TESTS COMPLETEs SUMMARY")
print("==========================================")
for r in results:
    print("Test " + str(r['test']) + " | Loss: " + str(r['loss_rate']) + "% | Sent: " + str(r['sent']) + "/3 | Dropped: " + str(r['dropped']) + "/3 | TX: " + str(round(r['tx_time'], 1)) + "ms")
print("Check Pi B for SSIM scores!")
 
