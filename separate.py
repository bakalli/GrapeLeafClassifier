import os 
import shutil as sh 
import sys

rot_filenames = []
measles_filenames = []
blight_filenames = []

with open("black_rot.txt",'r') as uht:
	rot_filenames = uht.readlines()
with open("leaf_blight.txt",'r') as uht:
	blight_filenames = uht.readlines()
with open("measles.txt",'r') as uht:
	measles_filenames = uht.readlines()


for file in rot_filenames:
	sh.move(os.getcwd()+file, os.getcwd()+"/black-rot/"+file)

for file in measles_filenames:
	sh.move(os.getcwd()+file, os.getcwd()+"/measles/"+file)

for file in blight_filenames:
	sh.move(os.getcwd()+file, os.getcwd()+"/leaf-blight/"+file)