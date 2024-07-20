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

import os
import sys
import time
import pathlib
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

os.environ['YOLO_VERBOSE'] = 'False'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

if not os.path.exists(args.input):
	print(f"Error:  Could not find input directory path '{args.input}'", flush=True)
	parser.print_usage()
	sys.exit(-1)

if not os.path.exists(args.output):
	print(f"Could not find output directory path '{args.output}'...Creating Directory!", flush=True)
	os.makedirs(args.output)

if not os.path.exists(args.logging):
	print(f"Could not find logging directory path '{args.logging}'...Creating Directory!", flush=True)
	os.makedirs(args.logging)

if args.cpu:
	device = "cpu"
	usegpu = False
else:
	if torch.cuda.is_available():
		device = "cuda"
		usegpu = True
	else:
		device = "cpu"
		usegpu = False

torch.device(device)

def load_models(pid):
	print(f"PID={pid}: Loading Megadetector model...")
	megadetector_model = torch.hub.load('ultralytics/yolov5', 'custom', path=MEGADETECTOR_MODEL, _verbose=False, verbose=False, trust_repo=True)

	print(f"PID={pid}: Loading EnlightenGAN Tiger model...")
	tiger_model = YOLO(TIGER_MODEL)

	print(f"PID={pid}: Loading Florence-2 model...")
	florence_model = AutoModelForCausalLM.from_pretrained(FLORENCE_MODEL, trust_remote_code=True).eval()
	florence_processor = AutoProcessor.from_pretrained(FLORENCE_MODEL, trust_remote_code=True)

	print(f"PID={pid}: All models loaded.")

	print(f"PID={pid}: Deploying all models to device: {device}")

	megadetector_model.to(device)
	tiger_model.to(device)
	florence_model.to(device)

	print(f"PID={pid}: Models deployed on device: {device}")

	return megadetector_model, tiger_model, florence_model, florence_processor



def report(pid, report_list):
	filename, clip_path, fps, start_frame, end_frame, confidences = report_list

	min_conf = min(confidences) if confidences else 0
	max_conf = max(confidences) if confidences else 0
	mean_conf = sum(confidences) / len(confidences) if confidences else 0

	s = f'"{filename}", "{clip_path}", {start_frame}, {start_frame/fps:.02f}, {end_frame}, {end_frame/fps:.02f}, {end_frame-start_frame}, {(end_frame-start_frame)/fps:.02f}, {min_conf:.02f}, {max_conf:.02f}, {mean_conf:.02f}\n'

	try:
		with open(args.report, "a") as report_file:
			report_file.write(s)
	except:
		print(f"Warning:  Could not open report file {args.report} for writing in report()", flush=True)

def label(img, frame, fps):
	s = f"frame: {frame}, time: {frame/fps:.3f}"
	cv2.putText(img, s, (200,100), cv2.FONT_HERSHEY_SIMPLEX, 1.75, (0,0,0), 6, cv2.LINE_AA) 	
	cv2.putText(img, s, (200,100), cv2.FONT_HERSHEY_SIMPLEX, 1.75, (255,255,255), 3, cv2.LINE_AA) 	
	return img

def clear_screen():
	os.system('cls' if os.name == 'nt' else 'clear')

def reset_screen():
	if os.name != 'nt':
		os.system('reset')

def human_size(bytes, units=[' bytes','KB','MB','GB','TB', 'PB', 'EB']):
	return str(bytes) + units[0] if bytes < 1024 else human_size(bytes>>10, units[1:])

def get_gpu_info():
	nvidia_smi.nvmlInit()

	deviceCount = nvidia_smi.nvmlDeviceGetCount()
	gpu_info = [deviceCount]
	for i in range(deviceCount):
		handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
		mem_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
		gpu_info.append((mem_info.total, mem_info.used, mem_info.free))

	nvidia_smi.nvmlShutdown()

	return gpu_info

def chunks(filenames, n):
	if n <= 0:
		return []

	chunk_size = len(filenames) // n
	remainder = len(filenames) % n
	
	chunks = []
	start_index = 0

	for i in range(n):
		end_index = start_index + chunk_size + (1 if i < remainder else 0)
		if end_index > start_index:
			chunks.append(filenames[start_index:end_index])
		start_index = end_index

	return chunks

def contains_target_label(data):
	target_labels = {"tiger", "tigers", "cat", "wildcat", "animal"}
	labels = data.get("<OD>", {}).get("labels", [])
	return any(label in target_labels for label in labels)

def ensemble_detection(img, megadetector_model, tiger_model, florence_model, florence_processor):
	# Megadetector Detection
	megadetector_results = megadetector_model(img).pandas().xyxy[0]
	megadetector_classes = megadetector_results['class'].tolist()
	megadetector_confidences = megadetector_results['confidence'].tolist()
	megadetector_has_animal = 0 in megadetector_classes
	megadetector_max_confidence = max(megadetector_confidences) if megadetector_confidences else 0

	# Tiger Detection
	tiger_results = tiger_model(img, verbose=False)
	if isinstance(tiger_results, list):
		tiger_results = tiger_results[0]
	tiger_confidences = [box.conf.item() for box in tiger_results.boxes]
	tiger_has_tiger = any(int(box.cls) == 0 for box in tiger_results.boxes)
	tiger_max_confidence = max(tiger_confidences) if tiger_confidences else 0

	# Microsoft Florence-2 Detection
	florence_image = Image.fromarray(img)
	florence_results = florence_task(florence_image, "OD>", florence_model, florence_processor)
	florence_results = contains_target_label(florence_results)

	results = {
		"megadetector_detection": megadetector_has_animal,
		"megadetector_conf": megadetector_max_confidence,
		"tiger_detection": tiger_has_tiger,
		"tiger_confidence": tiger_max_confidence,
		"florence_detection": florence_results
	}

	return results 

def get_video_chunk(invid, interval_sz, pu_lock):
	global chunk_idx

	buf = []
	for _ in range(interval_sz):
		success, image = invid.read()
		if success:
			buf.append(image)
		else:
			chunk_idx += 1
			return None, False

	inference_frame = image
	with pu_lock:
		try:
			results = ensemble_detection(inference_frame, megadetector_model, tiger_model, florence_model, florence_processor)
		except Exception as e:
			print(f"Error: Could not run model inference on frame from chunk index: {chunk_idx}")
			print(f"Exception: {e}", flush=True)
			sys.exit(-1)

	res = {
		"chunk_idx": chunk_idx,
		"buffer": buf,
		"megadetector_detection": results.get("megadetector_detection"),
		"megadetector_conf": results.get("megadetector_conf", 0),
		"tiger_detection": results.get("tiger_detection"),
		"tiger_conf": results.get("tiger_conf", 0),
		"florence_detection": results.get("florence_detection"),
		"overall_detection": False,
		"overall_confidence": 0
	}
	
	total_models = 3
	detections = [
		results.get("megadetector_detection"),
		results.get("tiger_detection"),
		results.get("florence_detection")
	]
	
	confidences = [
		results.get("megadetector_conf", 0) if results.get("megadetector_detection") else 0,
		results.get("tiger_conf", 0) if results.get("tiger_detection") else 0
	]
	
	detection_count = sum(detections)
	res["overall_detection"] = detection_count >= 2
	
	if res["overall_detection"]:
		base_confidence = sum(confidences) / len(confidences) if confidences else 0
		non_detection_count = total_models - detection_count
		adjustment_factor = 1 - (non_detection_count / total_models)
		res["overall_confidence"] = base_confidence * adjustment_factor
	else:
		res["overall_confidence"] = 0

	chunk_idx += 1
	return res, True

def write_clip(clip, frame_chunk):
	global most_recent_written_chunk 

	if frame_chunk["chunk_idx"] <= most_recent_written_chunk:
		print(f"***ALERT:  Trying to write the same chunk {frame_chunk['chunk_idx']} twice or out of order!!!  MOST RECENT CHUNK WRITTEN: {most_recent_written_chunk}")
		return

	most_recent_written_chunk = frame_chunk["chunk_idx"]
	
	for frame in frame_chunk["buffer"]:
		clip.write(frame)

def get_debug_buffer(frame_chunk):
	return " ".join(f"[{fc['chunk_idx']}|{fc['detection']}]" for fc in frame_chunk)

def florence_task(image, task_prompt, florence_model, florence_processor):
	inputs = florence_processor(text=task_prompt, images=image, return_tensors="pt")
	
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
	
	generated_ids = generated_ids.to("cpu")
	
	generated_text = florence_processor.batch_decode(generated_ids, verbose=False, skip_special_tokens=False)[0]
	
	parsed_answer = florence_processor.post_process_generation(
		generated_text,
		task=task_prompt,
		image_size=(image.width, image.height),
	)

	return parsed_answer

def process_chunks(pid, chunk, pu_lock, report_lock):
	global megadetector_model, tiger_model, florence_model, florence_processor
	global chunk_idx, most_recent_written_chunk

	# Load models only once per process
	megadetector_model, tiger_model, florence_model, florence_processor = load_models(pid)

	print("PID=", pid, " CHUNK=", chunk)
	print("\n")

	for filename in chunk:
		#print("Trying to read file: ", filename)
		imageio_success = False
		for _ in range(10):
			try:
				v = imageio.get_reader(filename, 'ffmpeg')
				nframes = v.count_frames()
				metadata = v.get_meta_data()
				v.close()

				fps = metadata['fps']
				duration = metadata['duration']

				size = metadata['size']

				imageio_success = True
				break
			except:
				print(f"pid={str(pid).zfill(2)}: WARNING: imageio timeout {filename}.   Trying again....", flush=True)
				time.sleep(1)

		if not imageio_success:
			print(f"pid={str(pid).zfill(2)}: WARNING: imageio could not read {filename}.  Skipping!", flush=True)
			continue

		width, height = size

		try:
			invid = cv2.VideoCapture(filename)
		except:
			print(f"pid={str(pid).zfill(2)}: Could not read video file: {filename}, skipping...", flush=True)
			continue

		DETECTION = 500
		SCANNING  = 501

		state = SCANNING

		interval_frames = int(args.interval * fps)
		padding_intervals = math.ceil(args.padding * fps / interval_frames)
		nchunks = math.ceil(nframes / interval_frames)
		chunk_idx = 0
		clip_number = 0
		buffer_chunks = []
		forward_buf = []
		confidences = []

		most_recent_written_chunk = -1

		if args.nobar:
			print(f"pid={str(pid).zfill(2)} Processing video {chunk.index(filename)+1}/{len(chunk)}: {filename}", flush=True)
		else:
			pbar = tqdm(total=nframes, position=pid, ncols=100, unit=" frames", leave=False, mininterval=0.5, file=sys.stdout)
			pbar.set_description(f"pid={str(pid).zfill(2)} Processing video {chunk.index(filename)+1}/{len(chunk)}: {filename}")

		frame_chunk, success = get_video_chunk(invid, interval_frames, pu_lock)
		if frame_chunk["overall_detection"]:
			confidences.append(frame_chunk["overall_confidence"])
		
		if not args.nobar:    
			pbar.update(interval_frames)

		while success:
			if chunk_idx > nchunks:
				break

			# State transition from SCANNING blanks to DETECTION
			if state == SCANNING and frame_chunk["overall_detection"]:
				state = DETECTION

				fn = os.path.basename(filename)
				clip_name = f"{os.path.splitext(fn)[0]}_{clip_number:03d}.mp4"
				clip_path = os.path.join(args.output, clip_name)
				fourcc = cv2.VideoWriter_fourcc(*'mp4v')    
				clip = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
				clip_number += 1
			
				clip_start_frame = buffer_chunks[0]["chunk_idx"] * interval_frames if buffer_chunks else 0
				for fc in buffer_chunks:
					write_clip(clip, fc)
				buffer_chunks = []
				write_clip(clip, frame_chunk)

			# Possible state transition from DETECTION back to SCANNING
			elif state == DETECTION and not frame_chunk["overall_detection"]:
				forward_buf = [frame_chunk]
				forward_detection_flag = frame_chunk["overall_detection"]
				for _ in range(2 * padding_intervals + 1):
					frame_chunk, success = get_video_chunk(invid, interval_frames, pu_lock)
					if success and frame_chunk["overall_detection"]:
						confidences.append(frame_chunk["overall_confidence"])
					if not args.nobar:
						pbar.update(interval_frames)
					if success and frame_chunk["chunk_idx"] <= nchunks:
						forward_buf.append(frame_chunk)
						if frame_chunk["overall_detection"]:
							forward_detection_flag = True

				if not forward_detection_flag:
					for f in buffer_chunks:
						write_clip(clip, f)
					buffer_chunks = []

					extent = min(padding_intervals, len(forward_buf))
					for i in range(extent):
						frame_chunk = forward_buf.pop(0)        
						write_clip(clip, frame_chunk)    

					buffer_chunks += forward_buf
					forward_buf = []

					clip.release()    
					clip_end_frame = (most_recent_written_chunk * interval_frames) + interval_frames
					with report_lock:
						report(pid, [filename, clip_path, fps, clip_start_frame, clip_end_frame, confidences])

					state = SCANNING

				else:
					for f in buffer_chunks:
						write_clip(clip, f)
					buffer_chunks = []

					last_forward_detection_idx = max(i for i, f in enumerate(forward_buf) if f["overall_detection"])

					for i in range(last_forward_detection_idx + 1):
						write_clip(clip, forward_buf[i])

					if last_forward_detection_idx < len(forward_buf) - 1: 
						forward_buf = forward_buf[last_forward_detection_idx + 1:]
					else:
						forward_buf = []
			
					buffer_chunks = forward_buf        
					forward_buf = []

			elif state == DETECTION and frame_chunk["overall_detection"]:
				if buffer_chunks:
					for ch in buffer_chunks:
						write_clip(clip, ch)
					buffer_chunks = []

				write_clip(clip, frame_chunk)

			else:  # state == SCANNING, frame_chunk["overall_detection"] == FALSE
				buffer_chunks.append(frame_chunk)
				if len(buffer_chunks) > padding_intervals:
					buffer_chunks.pop(0)
	
			frame_chunk, success = get_video_chunk(invid, interval_frames, pu_lock)
			if success and frame_chunk["overall_detection"]:
				confidences.append(frame_chunk["overall_confidence"])
			if not args.nobar:
				pbar.update(interval_frames)

		try:
			clip.release()
			clip_end_frame = (most_recent_written_chunk * interval_frames) + interval_frames
		except:
			pass
		 
		invid.release()

	if not args.nobar:
		pbar.close()

def main():
	all_start_time = time.time()

	if usegpu:
		gpu_info = get_gpu_info()
		print(f"Detected {gpu_info[0]} CUDA GPUs")
		for g in range(1, len(gpu_info)):
			mem_total, mem_used, mem_free = gpu_info[g]
			print(f"GPU:{g-1}, Memory : ({100*mem_free/mem_total:.2f}% free): {human_size(mem_total)}(total), {human_size(mem_free)} (free), {human_size(mem_used)} (used)")

	freeze_support()  # For Windows support - multiprocessing with tqdm

	try:
		with open(args.report, "w") as report_file:
			report_file.write("ORIGINAL, CLIP, START_FRAME, START_TIME, END_FRAME, END_TIME, NUM FRAMES, DURATION, MIN_CONF, MAX_CONF, MEAN_CONF\n")
	except:
		print(f"Error: Could not open report file {args.report} in main()", flush=True)
		sys.exit(-1)

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
	ch = chunks(files, args.jobs)

	manager = Manager()

	pu_lock = manager.Lock()
	report_lock = manager.Lock()

	if usegpu:
		torch.cuda.empty_cache()

	processes = []
	for pid, chunk in enumerate(ch):
		p = Process(target=process_chunks, args=(pid, chunk, pu_lock, report_lock))
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

	print(f"Total time to process {len(files)} videos: {time.time()-all_start_time:.02f} seconds")
	print(f"Report file saved to {args.report}")
	print("\nDONE\n")

if __name__ == '__main__':
	torch.multiprocessing.set_start_method('spawn')
	main()
