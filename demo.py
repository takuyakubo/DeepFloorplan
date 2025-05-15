import os
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
import imageio.v2 as imageio
from matplotlib import pyplot as plt

# GPUの設定
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# input image path
parser = argparse.ArgumentParser()
parser.add_argument('--im_path', type=str, default='./demo/45765448.jpg',
                    help='input image paths.')

# color map
floorplan_map = {
	0: [255,255,255], # background
	1: [192,192,224], # closet
	2: [192,255,255], # batchroom/washroom
	3: [224,255,192], # livingroom/kitchen/dining room
	4: [255,224,128], # bedroom
	5: [255,160, 96], # hall
	6: [255,224,224], # balcony
	7: [255,255,255], # not used
	8: [255,255,255], # not used
	9: [255, 60,128], # door & window
	10:[  0,  0,  0]  # wall
}

def ind2rgb(ind_im, color_map=floorplan_map):
	rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3))
	for i, rgb in color_map.items():
		rgb_im[(ind_im==i)] = rgb
	return rgb_im

def main(args):
	# load input
	im = imageio.imread(args.im_path, mode='RGB')
	im = np.array(Image.fromarray(im).resize((512,512))) / 255.
	
	# モデルの読み込み
	model = tf.saved_model.load('./pretrained/pretrained_r3d_saved_model')
	infer = model.signatures['serving_default']
	
	# 推論
	input_tensor = tf.convert_to_tensor(im.reshape(1,512,512,3), dtype=tf.float32)
	outputs = infer(inputs=input_tensor)
	
	# 結果の取得
	room_type = outputs['Cast:0'].numpy()
	room_boundary = outputs['Cast_1:0'].numpy()
	room_type, room_boundary = np.squeeze(room_type), np.squeeze(room_boundary)


	# 結果のマージ
	floorplan = room_boundary.copy()
	floorplan[room_boundary==1] = 9
	floorplan[room_boundary==2] = 10
	floorplan_rgb = ind2rgb(floorplan)

	# 結果の表示
	plt.figure(figsize=(12, 6))
	plt.subplot(121)
	plt.title('imput image')
	plt.imshow(im)
	plt.axis('off')
	
	plt.subplot(122)
	plt.title('inference')
	plt.imshow(floorplan_rgb/255.)
	plt.axis('off')
	
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	FLAGS, unparsed = parser.parse_known_args()
	main(FLAGS)
