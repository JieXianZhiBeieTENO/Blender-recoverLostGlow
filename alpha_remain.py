import cv2
import numpy as np
import numba
import os, sys, time
import multiprocessing as mul

def decode_img(path):
	return cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_UNCHANGED)

@numba.jit(nopython=True)
def alpha_remain(img):
	for i in img:
		for o in i:
			if o[3]!=0:
				continue
			l = np.sort(o[:3])
			max_value = l[-1]
			if max_value == 0:
				continue
			rate = 255/max_value
			for c,u in enumerate(o):
				o[c] = int(u*rate)
			o[3] = max_value
	return img

def time_summary(func):
	def cache(*args, **kwargs):
		s=time.time()
		func(*args, **kwargs)
		print("用时",time.time()-s,"s")
	return cache

def output(img, path, name=""):
	splitname = os.path.splitext(os.path.basename(path))

	name_suf = "_保留Alpha"

	if name:
		output_path = path+name_suf
		if not os.path.exists(output_path):
			os.mkdir(output_path)
		output_path += "\\"+name
	else:
		output_path = os.path.dirname(path)+"\\"+splitname[0]+name_suf+splitname[1]
		name = os.path.basename(path)
	cv2.imencode(os.path.splitext(name)[1], img)[1].tofile(output_path)
	print(f"{name} 创建完成！", end = " ")

PROC_COUT = None
def more_process(func,var,channels):
    if channels==0:
        return None
    print("\n创建进程\n")
    pool = mul.Pool(channels)
    end = pool.map(func,var)
    return end

def use_op(path_n_name):
	@time_summary
	def cache():
		path = path_n_name[0]
		name = path_n_name[1]
		output(alpha_remain(decode_img(path+(("\\"+name) if name else ""))), path, name)
	cache()

def op_forFolder(path):
	image_list = os.listdir(path)
	image_number = len(image_list)
	if image_number == 0:
		return
	elif image_number == 1:
		use_op([path, image_list[0]])
	else:
		more_process(use_op, zip([path]*image_number, image_list), PROC_COUT)


def use_op_forimage(path):
	use_op([path,""])
def op_forImage(img_path_list):
	image_number = len(img_path_list)
	if image_number == 0:
		return
	elif image_number == 1:
		use_op_forimage(img_path_list[0])
	else:
		more_process(use_op_forimage, img_path_list, PROC_COUT)
	

def main():
	img_path_list = []
	folder_path_list = []
	for sys_path in sys.argv[1:]:
		if os.path.splitext(os.path.basename(sys_path))[1]=='':
			folder_path_list.append(sys_path)
		else:
			img_path_list.append(sys_path)

	PROC_COUT = int(mul.cpu_count()/2)
	if PROC_COUT == 0:
		PROC_COUT = 1

	for folder_path in folder_path_list:
		op_forFolder(folder_path)
	op_forImage(img_path_list)


#多进程一定要加！！
if __name__ == "__main__":
	if hasattr(sys, 'frozen'):
		mul.freeze_support()

	all_time_start = time.time()
	print("开始创建\n")
	main()
	print(f"\n处理完成! 总耗时 {time.time()-all_time_start} s")
	time.sleep(1)
