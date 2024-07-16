####################################################################
#
# tigervid.py
#
# Luke Sheneman
# sheneman@uidaho.edu
# July 2024
#
# Given a directory of videos, process each video to look for animals
# Extracts video clips which include animals into destination directory
# Writes summary log
#
####################################################################


import os, sys, time, pathlib
import argparse
from multiprocessing import Process, current_process, freeze_support, Lock, RLock, Manager
import cv2
import math
import nvidia_smi
import logging
import random
import torch
import glob
import numpy as np
import imageio
from ultralytics import YOLO
from tqdm import tqdm	

from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests
import copy



DEFAULT_INPUT_DIR	 = "inputs"
DEFAULT_OUTPUT_DIR	 = "outputs"
DEFAULT_LOGGING_DIR  	 = "logs"

MEGADETECTOR_MODEL       = 'md_v5a.0.0.pt'
TIGER_MODEL              = 'best_enlightengan_and_yolov8.pt'
FLORENCE_MODEL           = 'microsoft/Florence-2-large'

DEFAULT_INTERVAL         = 1.0   # number of seconds between samples
DEFAULT_PADDING		 = 5.0   # number of seconds of video to include before first detection and after last detection in a clip
DEFAULT_REPORT_FILENAME  = "report.csv"
DEFAULT_NPROCS           = 4
DEFAULT_NOBAR		 = False
DEFAULT_YOLO_VERSION	 = 8   # either 5 or 8



parser = argparse.ArgumentParser(prog='tigervid', description='Analyze videos and extract clips and metadata which contain animals.')

parser.add_argument('input',  metavar='INPUT_DIR',  default=DEFAULT_INPUT_DIR,  help='Path to input directory containing MP4 videos')
parser.add_argument('output', metavar='OUTPUT_DIR', default=DEFAULT_OUTPUT_DIR, help='Path to output directory for clips and metadatas')

parser.add_argument('-i', '--interval', type=float, default=DEFAULT_INTERVAL,        help='Number of seconds between AI sampling/detection (DEFAULT: '+str(DEFAULT_INTERVAL)+')')
parser.add_argument('-p', '--padding',  type=float, default=DEFAULT_PADDING,         help='Number of seconds of video to pad on front and end of a clip (DEFAULT: '+str(DEFAULT_PADDING)+')')
parser.add_argument('-r', '--report',   type=str,   default=DEFAULT_REPORT_FILENAME, help='Name of report metadata (DEFAULT: '+DEFAULT_REPORT_FILENAME+')')
parser.add_argument('-j', '--jobs',	type=int,   default=DEFAULT_NPROCS,          help='Number of concurrent (parallel) processes (DEFAULT: '+str(DEFAULT_NPROCS)+')')
parser.add_argument('-l', '--logging',  type=str,   default=DEFAULT_LOGGING_DIR,     help='The directory for log files (DEFAULT: '+str(DEFAULT_LOGGING_DIR)+')')

parser.add_argument('-n', '--nobar',    action='store_true',  default=DEFAULT_NOBAR,     help='Turns off the Progress Bar during processing.  (DEFAULT: Use Progress Bar)')

group = parser.add_mutually_exclusive_group()
group.add_argument('-g', '--gpu', action='store_true',  default=True, help='Use GPU if available (DEFAULT)')
group.add_argument('-c', '--cpu', action='store_true', default=False, help='Use CPU only')

args = parser.parse_args()



args.cpu


if(not os.path.exists(args.input)):
	print("Error:  Could not find input directory path '%s'" %args.input, flush=True)
	parser.print_usage()
	exit(-1)

if(not os.path.exists(args.output)):
	print("Could not find output directory path '%s'...Creating Directory!" %args.output, flush=True)
	os.makedirs(args.outputs)

if(not os.path.exists(args.logging)):
	print("Could not find logging directory path '%s'...Creating Directory!" %args.logging, flush=True)
	os.makedirs(args.logging)

if(args.cpu==True):
	device = "cpu"
	torch.device(device)
	if __name__ == '__main__':
	    print("Using CPU", flush=True)
	usegpu = False
else:
	if(torch.cuda.is_available()):
		device = "cuda"
		usegpu = True
		if __name__ == '__main__':
		    print("Using GPU", flush=True)
	else:
		device = "cpu"
		usegpu = False

torch.device(device)

if __name__ != '__main__':
	logging.getLogger('torch.hub').setLevel(logging.ERROR)

	try:
		print("Loading megadetector model...", flush=True)
		megadetector_model = torch.hub.load('ultralytics/yolov5', 'custom', path=MEGADETECTOR_MODEL, _verbose=True, verbose=True, trust_repo=True)

		print("Loading tiger model...", flush=True)
		tiger_model        = YOLO(TIGER_MODEL)

		print("Loading Florence-2 model...", flush=True)
		florence_model     = AutoModelForCausalLM.from_pretrained(FLORENCE_MODEL, trust_remote_code=True).eval()
		florence_processor = AutoProcessor.from_pretrained(FLORENCE_MODEL, trust_remote_code=True)

		print("Models loaded...", flush=True)
		
		print("Depoying all models to device: ", device, flush=True)

		megadetector_model.to(device)
		tiger_model.to(device)
		florence_model.to(device)

		print("Models deployed on device: ", device,  flush=True)

	except Exception as e:
		print(f"Problem loading models and/or deploying to the GPU: {e}", flush=True)
		sys.exit(-1)


def report(pid, report_list):


	filename,clip_path,fps,start_frame,end_frame,confidences = report_list

	min_conf  = min(confidences)
	max_conf  = max(confidences)
	mean_conf = 0 if len(confidences) == 0 else sum(confidences)/len(confidences)

	s = "\"%s\", \"%s\", %d, %.02f, %d, %.02f, %d, %.02f, %.02f, %.02f, %.02f\n" %(filename, clip_path, start_frame, start_frame/fps, end_frame, end_frame/fps, end_frame-start_frame, (end_frame-start_frame)/fps, min_conf, max_conf, mean_conf)

	try:
		report_file = open(args.report, "a")
		report_file.write(s)	
		report_file.flush()
		report_file.close()
	except:
		print("Warning:  Could not open report file %s for writing in report()" %(args.report), flush=True)




def label(img, frame, fps):
	s = "frame: %d, time: %s" %(frame, "{:0.3f}".format(frame/fps))
	cv2.putText(img, s, (200,100), cv2.FONT_HERSHEY_SIMPLEX, 1.75, (0,0,0), 6, cv2.LINE_AA) 	
	cv2.putText(img, s, (200,100), cv2.FONT_HERSHEY_SIMPLEX, 1.75, (255,255,255), 3, cv2.LINE_AA) 	
	return(img)


def clear_screen():
	os.system('cls' if os.name == 'nt' else 'clear')

def reset_screen():
	if(os.name != 'nt'):
		os.system('reset')

def human_size(bytes, units=[' bytes','KB','MB','GB','TB', 'PB', 'EB']):
	return str(bytes) + units[0] if bytes < 1024 else human_size(bytes>>10, units[1:])

def get_gpu_info():
	nvidia_smi.nvmlInit()

	deviceCount = nvidia_smi.nvmlDeviceGetCount()
	gpu_info = []
	gpu_info.append(deviceCount)
	for i in range(deviceCount):
		handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
		mem_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

		mem_free  = mem_info.free
		mem_total = mem_info.total
		mem_used  = mem_info.used

		gpu_info.append((mem_info.total, mem_info.used, mem_info.free))

		#print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*mem_info.free/mem_info.total, human_size(mem_info.total), human_size(mem_info.free), human_size(mem_info.used)))

	nvidia_smi.nvmlShutdown()

	return(gpu_info)



def chunks(filenames, n):
	if n <= 0:
		return []

	chunk_size = len(filenames) // n
    
	remainder = len(filenames) % n
    
	chunks = []
	start_index = 0

	for i in range(n):
		end_index = start_index + chunk_size + (1 if i < remainder else 0)
		if(end_index > start_index):
			chunks.append(filenames[start_index:end_index])
		start_index = end_index

	return chunks


def ensemble_detection(img):


	# Megadetector Detection
	megadetector_results = megadetector_model(img).pandas().xyxy[0]
	megadetector_classes = megadetector_results['class'].tolist()
	megadetector_results = 0 in megadetector_classes  # 0 = index of animal class for megadetector

        # Tiger Detection
	tiger_results = tiger_model(img)
	if isinstance(tiger_results, list):
		tiger_results = tiger_results[0]  # Assuming we take the first result if it's a list
	tiger_results = any(int(box.cls) == 0 for box in tiger_results.boxes)

	# Microsoft Florence-2 Detection
	florence_image = Image.fromarray(img)
	florence_results  = florence_task(florence_image, task_prompt="<OPEN_VOCABULARY_DETECTION>", text_input="tiger")
	florence_polygons = florence_results['<OPEN_VOCABULARY_DETECTION>']['polygons']
	if(len(florence_polygons)>0):
		florence_results = True
	else:
		florence_results = False


	print("Megadetector Results:")
	print(megadetector_results)

	print("Tiger Results:")
	print(tiger_results)

	print("Florence Results:")
	print(florence_results)


#
# retrieves chunk of video frames of size interval_sz
# returns: 
#     result as dict with keys:  {frame_buffer, detection (boolean), confidence score}
#     success (True/False) 
#
def get_video_chunk(invid, interval_sz, pu_lock):

	global chunk_idx

	#print("Getting chunk: %d" %chunk_idx)

	res = {}
	res["chunk_idx"] = chunk_idx

	buf = []
	for i in range(interval_sz):
		success, image = invid.read()
		if(success):
			buf.append(image)
		else:
			#print("Error:  Could not read frame chunk: %d" %chunk_idx)
			chunk_idx += 1
			return(None, False)
			

	inference_frame = cv2.resize(image, (640,640))    # is this needed?
	with pu_lock:
		results = ensemble_detection(inference_frame)
		try:
			results = ensemble_detection(inference_frame)

		except Exception as e:
			print("Error: Could not run model inference on frame from chunk index: %d" %chunk_idx)
			print(f"Exception: {e}", flush=True)
			sys.exit(-1)

#	if(results.boxes is not None):
#		cls = results.boxes.cls.cpu().numpy()
#		conf  = results.boxes.conf.cpu().numpy()
#		if(len(cls)):
#			cls  = cls[0]
#			conf = conf[0]
#			#print(cls, conf)

	conf = 0.5	
	if(results==True):
		detection  = True
		confidence = conf
	else:
		detection  = False
		confidence = None
    
	#print("----> Detection is [%s] for chunk index: %d" %(str(detection), chunk_idx))

	res["buffer"]	  = buf
	res["detection"]  = detection
	res["confidence"] = confidence

	chunk_idx+=1

	return(res, True)


def write_clip(clip, frame_chunk):

	global most_recent_written_chunk 


	if(frame_chunk["chunk_idx"] <= most_recent_written_chunk):
		print("***ALERT:  Trying to write the same chunk %d twice or out of order!!!  MOST RECENT CHUNK WRITTEN: %d" %(frame_chunk["chunk_idx"], most_recent_written_chunk))
		return


	#print("Writing: [%d, %s]" %(frame_chunk["chunk_idx"], str(frame_chunk["detection"])))

	most_recent_written_chunk = frame_chunk["chunk_idx"]
	
	for frame in frame_chunk["buffer"]:
		clip.write(frame)
		


def get_debug_buffer(frame_chunk):

	debug_info = ""
	for fc in frame_chunk: 
		debug_info += "[%d|%s] " %(fc["chunk_idx"],fc["detection"]) 

	return(debug_info)


#
# call the Microsoft multi-modal CV Florence model
#
def florence_task(image, task_prompt, text_input=None):
	if text_input is None:
		prompt = task_prompt
	else:
		prompt = task_prompt + text_input

	inputs = florence_processor(text=prompt, images=image, return_tensors="pt")
    
	# Move input tensors to the GPU
	input_ids = inputs["input_ids"].to(device)
	pixel_values = inputs["pixel_values"].to(device)
    
	generated_ids = florence_model.generate(
		input_ids=input_ids,
		pixel_values=pixel_values,
		max_new_tokens=1024,
		early_stopping=False,
		do_sample=False,
		num_beams=3,
	)
    
	# Move generated_ids back to CPU before processing
	generated_ids = generated_ids.to("cpu")
    
	generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
	parsed_answer = florence_processor.post_process_generation(
		generated_text,
		task=task_prompt,
		image_size=(image.width, image.height)
	)

	return parsed_answer



def process_chunk(pid, chunk, pu_lock, report_lock):

	global args
	global chunk_idx
	global most_recent_written_chunk

	# lets pace ourselves on startup to help avoid general race conditions
	time.sleep(pid*1)

	for fcnt, filename in enumerate(chunk):


		imageio_success = False
		for x in range(10):
			try:
				v=imageio.get_reader(filename,  'ffmpeg')
				nframes  = v.count_frames()
				metadata = v.get_meta_data()
				v.close()

				fps = metadata['fps']
				duration = metadata['duration']
				size = metadata['size']

				imageio_success = True
	
				break
			except:
				print("pid=%s: WARNING: imageio timeout %s.   Trying again...." %(str(pid).zfill(2), filename), flush=True)
				time.sleep(1)

		if(imageio_success == False):
			print("pid=%s: WARNING: imageio could not read %s.  Skipping!" %(str(pid).zfill(2), filename), flush=True)
			continue

		(width,height) = size

		try:
			invid = cv2.VideoCapture(filename)
		except:
			print("pid=%s: Could not read video file: ", filename, " skipping..." %(str(pid).zfill(2)), flush=True)
			continue

		DETECTION = 500
		SCANNING  = 501

		state = SCANNING

		interval_frames	    = int(args.interval*fps)
		padding_intervals   = math.ceil(args.padding*fps/interval_frames)
		nchunks		    = math.ceil(nframes/interval_frames)
		chunk_idx     = 0
		clip_number   = 0
		buffer_chunks = []
		forward_buf   = []
		confidences   = []

		most_recent_written_chunk = -1
	
		#print("NUMBER OF FRAMES: ", nframes)
		#print("NUMBER OF CHUNKS: ", nchunks)
		#print("FRAMES PER INTERVAL: ", interval_frames)	
		#print("PADDING INTERVALS: ", padding_intervals)

		#clear_screen()
		if(args.nobar):
			print("pid=%s Processing video %d/%d: %s" %(str(pid).zfill(2),fcnt+1,len(chunk),filename), flush=True)
		else:
			pbar = tqdm(total=nframes,position=pid,ncols=100,unit=" frames",leave=False,mininterval=0.5,file=sys.stdout)
			pbar.set_description("pid=%s Processing video %d/%d: %s" %(str(pid).zfill(2),fcnt+1,len(chunk),filename))

		frame_chunk, success = get_video_chunk(invid, interval_frames, pu_lock)
		if(frame_chunk["detection"] == True):
			confidences.append(frame_chunk["confidence"])
		
		if(not args.nobar):	
			pbar.update(interval_frames)

		while(success):
	
			#print("CHUNK_IDX: %d/%d" %(chunk_idx, nchunks))
			if(chunk_idx > nchunks):
				#print("OUT OF CHUNKS TO READ.  BAILING")
				return

			# state transition from SCANNING blanks to DETECTION
			if(state == SCANNING and frame_chunk["detection"] == True):
				#print("State transition from SCANNING blanks to DETECTION", flush=True)
				state = DETECTION

				# some debugging of buffers
				#debug_buffer = get_debug_buffer(buffer_chunks)
				#print("   Primary buffer: ", debug_buffer)
				#debug_buffer = get_debug_buffer(forward_buf)
				#print("   Forward Buffer: ", debug_buffer)

				# create a clip
				fn = os.path.basename(filename)
				clip_name = os.path.splitext(fn)[0] + "_{:03d}".format(clip_number) + ".mp4"
				clip_path = os.path.join(args.output, clip_name)
				fourcc = cv2.VideoWriter_fourcc(*'mp4v')	
				clip = cv2.VideoWriter(clip_path, fourcc, fps, (width,height))
				clip_number += 1
			
				# track the first frame of the clip for export to metadata report
				if(len(buffer_chunks)>0):
					clip_start_frame = buffer_chunks[0]["chunk_idx"]*interval_frames
				else:
					clip_start_frame = 0
				# flush the current sliding window buffer to the new clip
				for fc in buffer_chunks:
					write_clip(clip, fc)
				buffer_chunks = []
				write_clip(clip, frame_chunk)

			# possible state transition from DETECTION back to SCANNING
			elif(state == DETECTION and frame_chunk["detection"] == False):
    
				#print("state  == DETECTION, detection == False", flush=True)
				#
				# some debugging of buffers
				#debug_buffer = get_debug_buffer(buffer_chunks)
				#print("   Primary buffer: ", debug_buffer)
				#debug_buffer = get_debug_buffer(forward_buf)
				#print("   Forward Buffer: ", debug_buffer)
				#
				#print("grabbing forward_buffer...")
    
				# lets look into the future 2X to make sure we can split the clip
				forward_buf = []
				forward_buf.append(frame_chunk)
				forward_detection_flag = frame_chunk["detection"]
				for i in range(0,2*padding_intervals+1):   #SHENEMAN
					frame_chunk, success = get_video_chunk(invid, interval_frames, pu_lock)
					if(success and frame_chunk["detection"] == True):
						confidences.append(frame_chunk["confidence"])
					if(not args.nobar):
						pbar.update(interval_frames)
					if(success and frame_chunk["chunk_idx"]<=nchunks):
						forward_buf.append(frame_chunk)
						if(frame_chunk["detection"]):
							forward_detection_flag = True
				#	else:
				#		print("ELSE: success and frame_chunk[0]<=nchunks")
					

				if(forward_detection_flag == False):   # no positive detections in forward buffer
					#print("   NO positive detections in the forward buffer.", flush=True)
   
					# some debugging of buffers
					#debug_buffer = get_debug_buffer(buffer_chunks)
					#print("   Primary buffer: ", debug_buffer)
					#debug_buffer = get_debug_buffer(forward_buf)
					#print("   Forward Buffer: ", debug_buffer)

					#print("  Flushing primary buffer", flush=True)
					for f in buffer_chunks:
						write_clip(clip, f)
					buffer_chunks = []

					# flush the first part of the forward buffer disk up to padding_intervals
					if(len(forward_buf)>padding_intervals):
						extent = padding_intervals
					else:
						extent = len(forward_buf)	

					for i in range(extent):
						frame_chunk = forward_buf.pop(0)		
						write_clip(clip, frame_chunk)	

					# put whatever is left of the forward buffer onto the end of primary buffer
					buffer_chunks += forward_buf
					forward_buf = []

					## WRITE CLIP TO DISK AND LOG
					clip.release()	
					clip_end_frame = (most_recent_written_chunk * interval_frames) + interval_frames
					with report_lock:
						report(pid, [filename, clip_path, fps, clip_start_frame, clip_end_frame, confidences])
					#print("***WROTE CLIP TO DISK***")

					# some debugging of buffers
					#debug_buffer = get_debug_buffer(buffer_chunks)
					#print("   Primary buffer: ", debug_buffer)
					#debug_buffer = get_debug_buffer(forward_buf)
					#print("   Forward Buffer: ", debug_buffer)
					#
					#print("Changing state back to SCANNING...", flush=True)
					state = SCANNING     # complete state transition back to SCANNING

				else:   # positive detection in the forward buffer
    
					#print("  Positive detections in the forward buffer.", flush=True)

					# some debugging of buffers
					#debug_buffer = get_debug_buffer(buffer_chunks)
					#print("   Primary buffer: ", debug_buffer)
					#debug_buffer = get_debug_buffer(forward_buf)
					#print("   Forward Buffer: ", debug_buffer)
                                        #
					#print("  Flushing buffer", flush=True)
					for f in buffer_chunks:
						write_clip(clip, f)
					buffer_chunks = []

					#write_clip(clip, frame_chunk)

					last_forward_detection_idx = -1

					for i,f in enumerate(forward_buf):
						if(f["detection"]):
							last_forward_detection_idx = i

					#print("  Flushing all chunks in forward buffer up to and including the last_forward_detection_idx: ", last_forward_detection_idx)

					# some debugging of buffers
					#debug_buffer = get_debug_buffer(buffer_chunks)
					#print("   Primary buffer: ", debug_buffer)
					#debug_buffer = get_debug_buffer(forward_buf)
					#print("   Forward Buffer: ", debug_buffer)


					for i in range(last_forward_detection_idx+1):
						write_clip(clip, forward_buf[i])
   
					if(last_forward_detection_idx < len(forward_buf)-1): 
						forward_buf = forward_buf[last_forward_detection_idx+1:]
					else:
						forward_buf = []
				
					buffer_chunks = forward_buf	    
					forward_buf = []

					# some debugging of buffers
					#debug_buffer = get_debug_buffer(buffer_chunks)
					#print("   Primary buffer: ", debug_buffer)
					#debug_buffer = get_debug_buffer(forward_buf)
					#print("   Forward Buffer: ", debug_buffer)
                                        #
					#print("\n")


			elif(state == DETECTION and frame_chunk["detection"] == True):

				if(len(buffer_chunks)>0):
					#print("Flushing Primary Buffer...")		    
					for ch in buffer_chunks:
						write_clip(clip, ch)
					buffer_chunks = []

				#print("state == DETECTION, and detection == TRUE", flush=True)
				write_clip(clip, frame_chunk)
	
			else:   # state == SCANNING, frame_chunk["detection"] == FALSE
				#print("state == SCANNING, detection == FALSE.  Continuing to see nothing....", flush=True) 

				# add this new chunk to the sliding window
				#print("Adding new chunk to sliding window...", flush=True)
				buffer_chunks.append(frame_chunk)
				if(len(buffer_chunks)>padding_intervals):
					buffer_chunks.pop(0)
    
		
			frame_chunk, success = get_video_chunk(invid, interval_frames, pu_lock)
			if(success and frame_chunk["detection"] == True):
				confidences.append(frame_chunk["confidence"])
			if(not args.nobar):
				pbar.update(interval_frames)


		try:
			clip.release()
			clip_end_frame = (most_recent_written_chunk * interval_frames) + interval_frames
			with report_lock:
				report(pid, [filename, clip_path, fps, clip_start_frame, clip_end_frame, confidences])
		except:
			None
			 
		invid.release()

	if(not args.nobar):
		pbar.close()
	#clear_screen()



########################################
#
# Main Execution Section
#
#
def main():

	all_start_time = time.time()

	if(usegpu == True):
		gpu_info = get_gpu_info()
		print("Detected %d CUDA GPUs" %(gpu_info[0]))
		for g in range(1,len(gpu_info)):
			mem_total, mem_used, mem_free = gpu_info[g]
			print("GPU:{}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(g-1, 100*mem_free/mem_total, human_size(mem_total), human_size(mem_free), human_size(mem_used)))

	freeze_support()  # For Windows support - multiprocessing with tqdm

	try:
		report_file = open(args.report, "w")
	except:
		print("Error: Could not open report file %s in main()" %(args.report), flush=True)
		exit(-1)

	report_file.write("ORIGINAL, CLIP, START_FRAME, START_TIME, END_FRAME, END_TIME, NUM FRAMES, DURATION, MIN_CONF, MAX_CONF, MEAN_CONF\n")
	report_file.flush()
	report_file.close()


	print('''
	****************************************************
	*                       __,,,,_                    *
	*        _ __..-;''`--/'/ /.',-`-.                 *
	*    (`/' ` |  \ \ \\ / / / / .-'/`,_               *
	*   /'`\ \   |  \ | \| // // / -.,/_,'-,           *
	*  /<7' ;  \ \  | ; ||/ /| | \/    |`-/,/-.,_,/')  *
	* /  _.-, `,-\,__|  _-| / \ \/|_/  |    '-/.;.\'    *
	* `-`  f/ ;      / __/ \__ `/ |__/ |               *
	*      `-'      |  -| =|\_  \  |-' |               *
	*            __/   /_..-' `  ),'  //               *
	*           ((__.-'((___..-'' \__.'                *
	*                                                  *
	****************************************************
	''', flush=True)

	print("            BEGINNING PROCESSING          ")
	print("*********************************************")
	print("           INPUT_DIR: ", args.input)
	print("          OUTPUT_DIR: ", args.output)
	print("   SAMPLING INTERVAL: ", args.interval, "seconds")
	print("    PADDING DURATION: ", args.padding, "seconds")
	print("    CONCURRENT PROCS: ", args.jobs)
	print("DISABLE PROGRESS BAR: ", args.nobar)
	print("             USE GPU: ", usegpu)
	print("         REPORT FILE: ", args.report)
	print("*********************************************\n\n", flush=True)

	path = os.path.join(args.input, "*.mp4")
	files = glob.glob(path)
	random.shuffle(files)
	ch = chunks(files,args.jobs)

	manager = Manager()

	pu_lock     = manager.Lock()
	report_lock = manager.Lock()

	if(usegpu==True):
		torch.cuda.empty_cache()

	processes = []
	for pid,chunk in enumerate(ch):
		p = Process(target = process_chunk, args=(pid, chunk, pu_lock, report_lock))
		processes.append(p)
		p.start()


	while any(p.is_alive() for p in processes):
		for p in processes:
			if p.exitcode is not None and p.exitcode != 0:
				print(f"Terminating due to failure in process {p.pid}")

				for p in processes:
					p.terminate()

				time.sleep(2)
				clear_screen()
				reset_screen()

				print("\n")
				print("*****************************************************************************")
				print("SOMETHING WENT HORRIBLY WRONG:")
				print("Failure to run model within system resources (e.g. GPU RAM).")
				print("Please reduce the number of concurrent jobs (i.e., --jobs <n>) and try again!")
				print("*****************************************************************************")
				print("\n\n")

				return

		time.sleep(0.5)  # Check periodically

	#clear_screen()
	#reset_screen()

	print("Total time to process %d videos: %.02f seconds" %(len(files), time.time()-all_start_time))
	print("Report file saved to %s" %args.report)
	print("\nDONE\n")


if __name__ == '__main__':

	torch.multiprocessing.set_start_method('spawn')

	main()

